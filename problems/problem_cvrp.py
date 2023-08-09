from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np
from utils import augmentation

CAPACITIES = {
    20: 30.,
    50: 40.,
    100: 50.,
    200: 70.,
}

EPSILON = {
    20: 0.33,
    50: 0.625,
    100: 1.0,
    200: 1.429
}

total_history = 25 # (T_ES in the paper)

class CVRP(object):

    NAME = 'cvrp'  # Capacitiated Vehicle Routing Problem
    
    def __init__(self, p_size, init_val_met = 'random', with_assert = False, DUMMY_RATE = 0.5, k = 5, with_bonus = True, with_regular = True):
        
        self.size = int(np.ceil(p_size * (1 + DUMMY_RATE)))   # the number of real nodes plus dummy nodes in cvrp
        self.real_size = p_size
        self.dummy_size = self.size - self.real_size
        self.init_val_met = init_val_met
        self.k_max = k
        self.state = 'eval'
        self.epsilon = EPSILON[p_size]
        self.with_bonus = with_bonus
        self.with_regular = with_regular
        self.with_assert = with_assert
        assert self.real_size + self.dummy_size == self.size
        print(f'CVRP with {p_size} nodes and {self.dummy_size} dummy depots (total {self.size}).\n', 
              f'Regulation: {self.with_regular} Bonus: {self.with_bonus} Do assert: {self.with_assert}.\n',
              f'MAX {self.k_max}-opt.\n')
    
    def train(self):
        self.state = 'train'
        
    def eval(self):
        self.state = 'eval'
    
    def augment(self, batch, val_m, only_copy=False):
        bs, gs, dim = batch['coordinates'].size()
        if only_copy:
            coordinates = batch['coordinates'].unsqueeze(1).expand(bs,val_m,gs,dim).clone().reshape(-1,gs,dim)
        else:
            coordinates = batch['coordinates'].unsqueeze(1).expand(bs,val_m,gs,dim).clone()
            coordinates = augmentation(coordinates, val_m).reshape(-1,gs,dim)
        demand = batch['demand'].unsqueeze(1).expand(bs,val_m,gs).clone().reshape(-1,gs)
        return {'coordinates': coordinates,
                'demand':demand}
    
    def input_feature_encoding(self, batch):
        return batch['coordinates'].clone()
    
    def get_initial_solutions(self, batch):
      
      batch_size = batch['coordinates'].size(0)
  
      def get_solution(methods):
          p_size = self.size
          
          if methods == 'random':
              
              candidates = torch.ones(batch_size,self.size).bool()
              candidates[:,:self.dummy_size] = False
              
              rec = torch.zeros(batch_size, self.size).long()
              selected_node = torch.zeros(batch_size, 1).long()
              cum_demand = torch.zeros(batch_size, 2)
              
              demand = batch['demand'].cpu()
              
              for i in range(self.size - 1):
                  
                  dists = torch.arange(p_size).view(-1, p_size).expand(batch_size, p_size).clone()
                  dists.scatter_(1, selected_node, 1e5)
                  dists[~candidates] = 1e5
                  dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                  dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                  
                  next_selected_node = dists.min(-1)[1].view(-1,1)
                  selected_demand = demand.gather(1,next_selected_node)
                  cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                  cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
    
                  rec.scatter_(1,selected_node, next_selected_node)
                  candidates.scatter_(1, next_selected_node, 0)
                  selected_node = next_selected_node  
                  
              return rec
          
          
          elif methods == 'greedy':

              candidates = torch.ones(batch_size,self.size).bool()
              candidates[:,:self.dummy_size] = False
              
              rec = torch.zeros(batch_size, self.size).long()
              selected_node = torch.zeros(batch_size, 1).long()
              cum_demand = torch.zeros(batch_size, 2)
              
              coor = batch['coordinates'].cpu()
              demand = batch['demand'].cpu()
              
              for i in range(self.size - 1):
                  
                  coor_now = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size, 2))
                  dists = (coor_now - coor).norm(p=2, dim=2)
                  
                  dists.scatter_(1, selected_node, 1e5)
                  dists[~candidates] = 1e5
                  dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                  dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                  
                  next_selected_node = dists.min(-1)[1].view(-1,1)
                  selected_demand = demand.gather(1,next_selected_node)
                  cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                  cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
                  
                  rec.scatter_(1,selected_node, next_selected_node)
                  candidates.scatter_(1, next_selected_node, 0)
                  selected_node = next_selected_node                         

              return rec
          
          else:
              raise NotImplementedError()

      return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
  
    def f(self, p): # The entropy measure in Eq.(5)
        return torch.clamp(1 - 0.5 * torch.log2(2.5*np.pi*np.e*p*(1-p)+ 1e-5), 0, 1)
    
    def step(self, batch, rec, action, obj, feasible_history, t, weights = 0):
        
        bs, gs = rec.size()
        pre_bsf = obj[:,1:].clone() # batch_size, 3 (current, bsf, tsp_bsf)
        feasible_history = feasible_history.clone() # bs, total_history 
        
        # k-opt step
        next_state = self.k_opt(rec, action)
        next_obj, context = self.get_costs(batch, next_state, True)
        
        # MDP step
        non_feasible_cost_total = torch.clamp_min(context[-1] - 1.00001, 0.0).sum(-1)
        feasible = non_feasible_cost_total <= 0.0
        soft_infeasible = (non_feasible_cost_total <= self.epsilon) & (non_feasible_cost_total > 0.)
        
        now_obj = pre_bsf.clone()
        now_obj[feasible,0] = next_obj[feasible].clone()
        now_obj[soft_infeasible,1] = next_obj[soft_infeasible].clone()
        now_bsf = torch.min(pre_bsf, now_obj)
        rewards = (pre_bsf - now_bsf) #bs,2(feasible_reward,infeasible_reward) 

        # feasible history step
        feasible_history[:,1:] = feasible_history[:,:total_history-1].clone()
        feasible_history[:,0] = feasible.clone()
        
        # compute the ES features
        feasible_history_pre = feasible_history[:,1:]
        feasible_history_post = feasible_history[:,:total_history-1]
        f_to_if = ((feasible_history_pre == True) & (feasible_history_post == False)).sum(1,True) / (total_history-1)
        f_to_f = ((feasible_history_pre == True) & (feasible_history_post == True)).sum(1,True) / (total_history-1)
        if_to_f = ((feasible_history_pre == False) & (feasible_history_post == True)).sum(1,True) / (total_history-1)
        if_to_if = ((feasible_history_pre == False) & (feasible_history_post == False)).sum(1,True) / (total_history-1)
        f_to_if_2 = f_to_if / (f_to_if + f_to_f + 1e-5)
        f_to_f_2 =  f_to_f / (f_to_if + f_to_f + 1e-5)
        if_to_f_2 =  if_to_f / (if_to_f + if_to_if + 1e-5)
        if_to_if_2 =  if_to_if / (if_to_f + if_to_if + 1e-5)

        # update info to decoder
        active = (t >= (total_history - 2))
        context2 = torch.cat((
                      (if_to_if * active),
                      (if_to_if_2 * active),
                      (f_to_f * active),
                      (f_to_f_2 * active),
                      (if_to_f * active),
                      (if_to_f_2 * active),
                      (f_to_if * active),
                      (f_to_if_2 * active),
                      feasible.unsqueeze(-1).float(),
                    ),-1) # 9 ES features
        
        # update regulation
        reg = self.f(f_to_f_2) + self.f(if_to_if_2)
        
        reward = torch.cat((rewards[:,:1], # reward
                            -1 * reg * weights * 0.05 * self.with_regular, # regulation, alpha = 0.05
                            rewards[:,1:2] * 0.05 * self.with_bonus, # bonus, beta = 0.05
                           ),-1)

        out = (next_state, 
               reward,
               torch.cat((next_obj[:,None], now_bsf),-1), 
               feasible_history,
               context,
               context2,
               (if_to_if,if_to_f,f_to_if,f_to_f,if_to_if_2,if_to_f_2,f_to_if_2,f_to_f_2)
               )
        
        return out

    def k_opt(self, rec, action):
        
        # action bs * (K_index, K_from, K_to)
        selected_index = action[:,:self.k_max]
        left = action[:,self.k_max:2*self.k_max]
        right = action[:,2*self.k_max:]
        
        # prepare
        rec_next = rec.clone()
        right_nodes = rec.gather(1,selected_index)
        argsort = rec.argsort()
        
        # new rec
        rec_next.scatter_(1,left,right)
        cur = left[:,:1].clone()
        for i in range(self.size - 2): # self.size - 2 is already correct
            next_cur = rec_next.gather(1,cur)
            pre_next_wrt_old = argsort.gather(1, next_cur)
            reverse_link_condition = ((cur!=pre_next_wrt_old) & ~((next_cur==right_nodes).any(-1,True)))
            next_next_cur = rec_next.gather(1,next_cur)
            rec_next.scatter_(1,next_cur,torch.where(reverse_link_condition, pre_next_wrt_old, next_next_cur))
            # if i >= self.size - 2: assert (reverse_link_condition == False).all()
            cur = next_cur
            
        return rec_next

    def get_order(self, rec, return_solution = False):
        
        bs,p_size = rec.size()
        visited_time = torch.zeros((bs,p_size),device = rec.device)
        pre = torch.zeros((bs),device = rec.device).long()
        for i in range(p_size - 1):
            visited_time[torch.arange(bs),rec[torch.arange(bs),pre]] = i + 1
            pre = rec[torch.arange(bs),pre]
        if return_solution:
            return visited_time.argsort() # return decoded solution in sequence
        else:
            return visited_time.long() # also return visited order

    def check_feasibility(self, rec, partial_sum_wrt_route_plan, basic = False):
        p_size = self.size
        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).sum(-1),"not visiting all nodes")
        
        real_solution = self.get_order(rec, True)
            
        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            real_solution.sort(1)[0]
        ).all(), ((
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            real_solution.sort(1)[0]
        ).sum(-1),"not valid tour")
            
        if not basic:   
            assert (partial_sum_wrt_route_plan <= 1 + 1e-5).all(), ("not satisfying capacity constraint", partial_sum_wrt_route_plan, partial_sum_wrt_route_plan.max())
    
    def get_costs(self, batch, rec, get_context = False, check_full_feasibility = False):
        
        coor = batch['coordinates']
        coor_next = coor.gather(1, rec.long().unsqueeze(-1).expand(*rec.size(), 2))
        cost = (coor - coor_next).norm(p=2, dim=2).sum(1)
        
        # check TSP feasibility if needed
        if self.with_assert:
            self.check_feasibility(rec, None, basic = True)
        
        # check full feasibility if needed
        if check_full_feasibility or get_context:
            context = self.preprocessing(rec, batch)
            if check_full_feasibility:
                self.check_feasibility(rec, context[-1], basic = False)          
        
        # get CVRP context
        if get_context:
            return cost, context
        else:
            return cost

    def get_dynamic_feature(self, rec, batch, context):
    
        route_plan_0x, visited_time, cum_demand, partial_sum_wrt_route_plan = context
        demand = batch['demand'].unsqueeze(-1)
        cum_demand = cum_demand.unsqueeze(-1)
        route_total_demand_per_node = partial_sum_wrt_route_plan.gather(-1, route_plan_0x).unsqueeze(-1)
        
        infeasibility_indicator_after_visit = torch.clamp_min(cum_demand - 1.00001, 0.0) > 0
        infeasibility_indicator_before_visit = torch.clamp_min((cum_demand - demand) - 1.00001, 0.0) > 0
        
        to_actor = torch.cat((
            cum_demand,  
            demand, 
            route_total_demand_per_node - cum_demand,
            (demand == 0).float(),
            infeasibility_indicator_before_visit,
            infeasibility_indicator_after_visit,
            ), -1) # the node features
        
        return visited_time, to_actor
        
    def preprocessing(self, solutions, batch):
        
        batch_size, seq_length = solutions.size()
        assert seq_length < 1000
        arange = torch.arange(batch_size)
        demand = batch['demand']
        
        pre = torch.zeros(batch_size, device = solutions.device).long()
        route = torch.zeros(batch_size, device = solutions.device).long()
        route_plan_visited_time = torch.zeros((batch_size,seq_length), device = solutions.device).long()
        cum_demand = torch.zeros((batch_size,seq_length), device = solutions.device)
        partial_sum_wrt_route_plan = torch.zeros((batch_size, self.dummy_size), device = solutions.device)
        
        for i in range(seq_length):
            next_ = solutions[arange,pre]
            next_is_dummy_node = next_ < self.dummy_size
            route[next_is_dummy_node] += 1
            route_plan_visited_time[arange,next_] = (route % self.dummy_size) * int(1e3) + (i+1) % self.size
            new_cum_demand = partial_sum_wrt_route_plan[arange,route % self.dummy_size] + demand[arange, next_]
            partial_sum_wrt_route_plan[arange,route % self.dummy_size] = new_cum_demand.clone()
            cum_demand[arange,next_] = new_cum_demand * (~next_is_dummy_node)
            
            pre = next_.clone()
    
        route_plan_0x = (route_plan_visited_time // int(1e3))
        
        out =  (route_plan_0x, # route plan 0xxxxx
                (route_plan_visited_time % int(1e3)), # visited time
                cum_demand.clone(), # cum_demand (inclusive)
                partial_sum_wrt_route_plan.clone()) # partial_sum_wrt_route_plan
        
        return out
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPDataset(*args, **kwargs)


class CVRPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE = None):
        
        super(CVRPDataset, self).__init__()
        
        self.data = []
        self.size = int(np.ceil(size * (1 + DUMMY_RATE))) # the number of real nodes plus dummy nodes in cvrp
        self.real_size = size # the number of real nodes in cvrp

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

        else:            
            self.data = [{'coordinates': torch.cat((torch.FloatTensor(1, 2).uniform_(0, 1).repeat(self.size - self.real_size,1), 
                                                    torch.FloatTensor(self.real_size, 2).uniform_(0, 1)), 0),
                          'demand': torch.cat((torch.zeros(self.size - self.real_size),
                                               torch.FloatTensor(self.real_size).uniform_(1, 10).long() / CAPACITIES[self.real_size]), 0)
                          } for i in range(num_samples)]
        
        self.N = len(self.data)
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        depot, loc, demand, capacity, *args = args
        
        depot = torch.FloatTensor(depot)
        loc = torch.FloatTensor(loc)
        demand = torch.FloatTensor(demand)
        
        return {'coordinates': torch.cat((depot.view(-1, 2).repeat(self.size - self.real_size,1), loc), 0),
                'demand': torch.cat((torch.zeros(self.size - self.real_size), demand / capacity), 0) }
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]