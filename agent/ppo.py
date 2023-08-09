import os
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboard_logger import Logger as TbLogger
import numpy as np
import random

from utils import clip_grad_norms
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to
from utils.logger import log_to_tb_train
from agent.utils import validate,gather_tensor_and_concat
from problems.problem_cvrp import total_history

feasibility_history_base = [True] * (total_history)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = [] 
        self.obj = []
        self.context = []
        self.context2 = []
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obj[:]
        del self.context[:]
        del self.context2[:]

class PPO:
    def __init__(self, problem, opts):
        
        # figure out the options
        self.opts = opts
        
        # figure out the actor
        self.actor = Actor(
            problem = problem,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.actor_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            v_range = opts.v_range,
            seq_length = problem.size,
            k = opts.k,
            with_RNN = not opts.wo_RNN,
            with_feature1 = not opts.wo_feature1,
            with_feature3 = not opts.wo_feature3,
            with_simpleMDP = opts.wo_MDP
        )
        
        if not opts.eval_only:
            # figure out the critic
            self.critic = Critic(
                    embedding_dim = opts.embedding_dim,
                    hidden_dim = opts.hidden_dim,
                    n_heads = opts.critic_head_num,
                    n_layers = opts.n_encode_layers,
                    normalization = opts.normalization,
                    with_regular = not opts.wo_regular,
                    with_bonus = not opts.wo_bonus
                )
            
            self.optimizer = torch.optim.Adam(
                [{'params':  self.actor.parameters(), 'lr': opts.lr_model}] + 
                [{'params':  self.critic.parameters(), 'lr': opts.lr_critic}])
            
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        print(f'Distributed: {opts.distributed}')
        if opts.use_cuda and not opts.distributed:
            self.actor.to(opts.device)
            if not opts.eval_only: self.critic.to(opts.device)
                
    
    def load(self, load_path):
        
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
      
        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**load_data.get('actor', {})})
        
        if not self.opts.eval_only:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))
        
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    
    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only: 
            self.critic.eval()
        
    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only: 
            self.critic.train()
    
    def rollout(self, problem, T, val_m, stall_limit, batch, record = False, show_bar = False):
        bs, gs, _ = batch['coordinates'].size()
        batch = move_to(batch, self.opts.device)
        batch_aug_same = problem.augment(batch, val_m, only_copy=True)
        batch_aug = problem.augment(batch, val_m)
        batch_feature = problem.input_feature_encoding(batch_aug)
        
        solutions = move_to(problem.get_initial_solutions(batch_aug_same), self.opts.device)
        solution_best = solutions.clone()
        
        obj, context = problem.get_costs(batch_aug_same, solutions, get_context = True, check_full_feasibility = True)
        obj = torch.cat((obj[:,None], obj[:,None],obj[:,None]),-1).clone()
        context2 = torch.zeros(bs*val_m,9).to(solutions.device);context2[:,-1] = 1
        feasibility_history = torch.tensor(feasibility_history_base).view(-1,total_history).expand(bs*val_m, total_history).to(obj.device)
        
        solution_history = [solutions.clone()]
        solution_best_history = [solution_best.clone()]
        obj_history = [obj.clone()]        
        feasible_history_recorded = [feasibility_history[:,0]]
        action = None
        reward = []
        stall_cnt_ins = torch.zeros(bs * val_m).to(solution_best.device)

        for t in tqdm(range(T), disable = self.opts.no_progress_bar or not show_bar, desc = 'rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):       

            action = self.actor(problem,
                                batch_aug_same,
                                batch_feature,
                                solutions,
                                context,
                                context2,
                                action)[0]

            solutions, rewards, obj, feasibility_history, context, context2, info = problem.step(batch_aug_same, 
                                                                                                solutions,
                                                                                                action,
                                                                                                obj,
                                                                                                feasibility_history,
                                                                                                t,
                                                                                                weights = 0)
            index = rewards[:,0] > 0.0
            solution_best[index] = solutions[index].clone()

            # record informations
            reward.append(rewards[:,0].clone())
            obj_history.append(obj.clone())
            if record: 
                solution_history.append(solutions.clone())
                solution_best_history.append(solution_best.clone())
                feasible_history_recorded.append(feasibility_history[:,0].clone())
            
            # augment if stall>0 (checking every step)
            if stall_limit > 0:
                batch_aug_temp = problem.augment(batch, val_m)
                stall_cnt_ins = stall_cnt_ins * (1 - index.float()) + 1
                index_aug = stall_cnt_ins >= stall_limit
                batch_aug['coordinates'][index_aug] = batch_aug_temp['coordinates'][index_aug]
                batch_feature = problem.input_feature_encoding(batch_aug)
                stall_cnt_ins[index_aug] *= 0

        
        # assert
        best_length = problem.get_costs(batch_aug_same, solution_best, get_context = False, check_full_feasibility = True)
        assert (best_length - obj[:,1] < 1e-5).all(), (best_length, obj, best_length - obj[:,1])
        assert (problem.augment(batch, val_m, only_copy=True)['coordinates'] == batch_aug_same['coordinates']).all()
        out = (obj[:,1].reshape(bs, val_m).min(1)[0], # batch_size, 1
               torch.stack(obj_history,1).view(bs, val_m, T+1, -1).min(1)[0], # batch_size, T, 2/3
               torch.stack(reward,1).view(bs, val_m, T).max(1)[0], # batch_size, T
               None if not record else (solution_history, solution_best_history, feasible_history_recorded)
              )
        
        return out
    
    def start_inference(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:            
            mp.spawn(validate, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger, True))
        else:
            validate(0, problem, self, val_dataset, tb_logger, distributed = False)
            
    def start_training(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:
            mp.spawn(train, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger))
        else:
            train(0, problem, self, val_dataset, tb_logger)
            
def train(rank, problem, agent, val_dataset, tb_logger):
    
    opts = agent.opts
    warnings.filterwarnings("ignore")
    if opts.resume is None:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
        np.random.seed(opts.seed)
        
    if opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.critic.to(device)
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        
        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor,
                                                               device_ids=[rank])
        if not opts.eval_only: 
            agent.critic = torch.nn.parallel.DistributedDataParallel(agent.critic,
                                                               device_ids=[rank])
            
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    else:
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
                        
    if opts.distributed: dist.barrier()
    if rank == 0 and not opts.no_saving: agent.save('init')
    
    # Start the actual training loop
    batch_reward = None
    for epoch in range(opts.epoch_start, opts.epoch_end):
        
        agent.lr_scheduler.step(epoch)
        
        # Training mode
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'], 
                                                                                 agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)
        # prepare training data
        training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, DUMMY_RATE = opts.dummy_rate)
        if opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=False)
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size // opts.world_size, shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler)
        else:
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)
            
        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)  
        pbar = tqdm(total = (opts.K_epochs) * (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step) ,
                    disable = opts.no_progress_bar or rank!=0, desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch_id, batch in enumerate(training_dataloader):
            if batch_reward is None:
                batch_reward = []
                weights = 0
            else:
                batch_reward = torch.cat(batch_reward)
                if opts.distributed:
                    dist.barrier()
                    batch_reward = gather_tensor_and_concat(batch_reward.contiguous())
                    dist.barrier()
                weights = batch_reward.mean()
                batch_reward = []
            
            train_batch(rank,
                        problem,
                        agent,
                        epoch,
                        step,
                        batch,
                        tb_logger,
                        opts,
                        pbar,
                        batch_reward,
                        weights
                        )
            step += 1
            
        pbar.close()
        # save new model after one epoch  
        if rank == 0 and not opts.distributed: 
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
        elif opts.distributed and rank == 1:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
            
        
        # validate the new model   
        if rank == 0: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        
        # syn
        if opts.distributed: dist.barrier()

def train_batch(
        rank,
        problem,
        agent,
        epoch,
        step,
        batch,
        tb_logger,
        opts,
        pbar,
        batch_reward,
        weights
        ):
    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip

    # prepare the instances
    batch = move_to(batch, rank) if opts.distributed else move_to(batch, opts.device)# batch_size, graph_size, 2
    batch_feature = problem.input_feature_encoding(batch).cuda() if opts.distributed \
                        else move_to(problem.input_feature_encoding(batch), opts.device)
    bs, gs, _ = batch_feature.size()
        
    # initial solution
    solution = move_to(problem.get_initial_solutions(batch),rank) if opts.distributed \
                        else move_to(problem.get_initial_solutions(batch), opts.device)
    solution_best = solution.clone()
    
    # preapare input
    obj, context = problem.get_costs(batch, solution, get_context = True)
    obj = torch.cat((obj[:,None],obj[:,None],obj[:,None]),-1).clone()
    context2 = torch.zeros(bs,9).to(solution.device);context2[:,-1] = 1
    feasibility_history = torch.tensor(feasibility_history_base).view(-1,total_history).expand(bs, total_history).to(obj.device)
    action = None
    
    # CL stage
    agent.eval()
    problem.eval()
    for w in range(int(epoch // opts.warm_up)):
        # get model output	
        action  = agent.actor(problem,
                              batch,
                              batch_feature,
                              solution,
                              context,
                              context2,
                              action)[0]
        # state transient
        solution, rewards, obj, feasibility_history, context, context2, info = problem.step(batch, solution, action, obj, feasibility_history, w, weights = 0)
        index = rewards[:,0] > 0.0
        solution_best[index] = solution[index].clone()
            
    # re-init input
    solution = solution_best.clone()
    obj, context = problem.get_costs(batch, solution, get_context=True, check_full_feasibility=True)
    obj = torch.cat((obj[:,None],obj[:,None],obj[:,None]),-1).clone()
    context2 = torch.zeros(bs,9).to(solution.device);context2[:,-1] = 1
    feasibility_history = torch.tensor(feasibility_history_base).view(-1,total_history).expand(bs, total_history).to(obj.device)
    action = None
    initial_cost = obj.clone()
    
    agent.train()
    problem.train()    
    
    # sample trajectory
    t = 0
    memory = Memory()
    while t < T:
        
        t_s = t
        total_cost = 0
        total_cost_wo_feasible = 0
        entropy = []
        bl_val_detached = []
        bl_val = []
        memory.actions.append(action)
        
        while t - t_s < n_step and not (t == T):          
            
            # pass actor
            memory.states.append(solution.clone())
            if context is not None:
                memory.context.append(tuple(t.clone() for t in context))
                memory.context2.append(context2.clone())
            else:
                memory.context.append(None)
                memory.context2.append(None)
                
            action, log_lh, _to_critic, entro_p  = agent.actor(problem,
                                                               batch,
                                                               batch_feature,
                                                               solution,
                                                               context,
                                                               context2,
                                                               action,
                                                               require_entropy = True,
                                                               to_critic = True)
   
            memory.actions.append(action.clone())
            memory.logprobs.append(log_lh.clone())
            memory.obj.append(obj.clone())
            entropy.append(entro_p)
            
            # pass critic
            baseline_val_detached, baseline_val = agent.critic(_to_critic, obj, context2)
            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)
            
            # state transient
            solution, rewards, obj, feasibility_history, context, context2, info = problem.step(batch, 
                                                                                                solution, 
                                                                                                action, 
                                                                                                obj, 
                                                                                                feasibility_history,
                                                                                                t,
                                                                                                weights = weights)
            batch_reward.append(rewards[:,0].clone())
            memory.rewards.append(rewards)
            
            total_cost = total_cost + obj[:,1]
            total_cost_wo_feasible = total_cost_wo_feasible + obj[:,2]
            
            # next            
            t = t + 1
            
            
        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time
        total_cost_wo_feasible = total_cost_wo_feasible / t_time
        
        # convert list to tensor 
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        old_obj = torch.stack(memory.obj)
        old_value = None
                        
        # Optimize ppo policy for K mini-epochs:
        for _k in range(K_epochs):
            
            current_step = int(step * T / n_step * K_epochs + (t-1)//n_step * K_epochs  + _k)
            
            if _k == 0:
                
                logprobs = memory.logprobs
                
            else:
            
                # Evaluating old actions and values :
                logprobs = []  
                entropy = []
                bl_val_detached = []
                bl_val = []
                
                for tt in range(t_time):
                    # get new action_prob
                    _, log_p, _to_critic, entro_p = agent.actor(problem,
                                                                batch,
                                                                batch_feature,
                                                                memory.states[tt],
                                                                memory.context[tt],
                                                                memory.context2[tt],
                                                                last_action = memory.actions[tt],
                                                                fixed_action = memory.actions[tt+1],
                                                                require_entropy = True,# take same action
                                                                to_critic = True)
                    
                    assert (_ == memory.actions[tt+1]).all()
                    logprobs.append(log_p)
                    entropy.append(entro_p)
                    baseline_val_detached, baseline_val = agent.critic(_to_critic, old_obj[tt], memory.context2[tt])
                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)
            
            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached)
            bl_val = torch.stack(bl_val)

            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            c_cost_logger = torch.tensor(0.) if weights == 0 else (torch.stack(memory.rewards)[:,:,1].mean()/weights*-1)
            
            # estimate return
            new_to_critic = agent.actor(problem,batch,batch_feature,solution,context,context2,None,only_critic = True)
            R = agent.critic(new_to_critic, obj, context2)[0]
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R.clone())
            
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0) # n_step, bs, 3

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss:
            advantages = (Reward.sum(-1) - bl_val_detached.sum(-1)).view(-1)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages  
            
            reinforce_loss = - torch.min(surr1, surr2).mean()
            
            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).sum(-1).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                baseline_loss = torch.max(
                            ((bl_val - Reward) ** 2).sum(-1),
                            ((vpredclipped - Reward) ** 2).sum(-1)
                        ).mean()
                
            # check K-L divergence
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            
            # calculate loss
            loss = baseline_loss + reinforce_loss

            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradient norm and get (clipped) gradient norms for logging
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)
            
            # perform gradient descent
            agent.optimizer.step()
    
            # Logging to tensorboard            
            if(not opts.no_tb) and rank == 0:
                if (current_step + 1) % int(opts.log_step) == 0:
                    log_to_tb_train(tb_logger, agent, Reward, ratios, bl_val_detached, total_cost, total_cost_wo_feasible, grad_norms, entropy, approx_kl_divergence,
                       reinforce_loss, baseline_loss, c_cost_logger, weights, logprobs, initial_cost, info, current_step + 1)
                    
            if rank == 0: pbar.update(1)
        
        # end update
        memory.clear_memory()