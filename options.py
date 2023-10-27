import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(description="NeuOpt-GIRE")

    ### overall settings
    parser.add_argument('--problem', default='tsp', choices = ['tsp', 'cvrp'], help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--dummy_rate', type=float, default=0.5) # 0.5, 0.4, 0.2 for CVRP20, 50, 100, respectively
    parser.add_argument('--eval_only', action='store_true', help='used only if to evaluate a model')
    parser.add_argument('--init_val_met', choices = ['greedy', 'random'], default = 'random', help='method to generate initial solutions while validation')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard')
    parser.add_argument('--no_saving', action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--no_DDP', action='store_true')
    parser.add_argument('--use_assert', action='store_true', help='Enable Assertion')
    parser.add_argument('--seed', type=int, default=6666, help='Random seed to use')
    
    ### NeuOpt configs
    parser.add_argument('--val_m', type=int, default=1) # number of augmentation, D2A=1 or D2A=5 in Table 1
    parser.add_argument('--stall_limit', type=int, default=10) # T_D2A in the paper, 0 means disable
    parser.add_argument('--k', type=int, default=4) # the maximum basis move number K
    parser.add_argument('--wo_regular', action='store_true') # to remove reward shaping term
    parser.add_argument('--wo_bonus', action='store_true') # to remove reward shaping term
    parser.add_argument('--wo_RNN', action='store_true') # to remove RNN
    parser.add_argument('--wo_feature1', action='store_true') # to remove VI featrues
    parser.add_argument('--wo_feature3', action='store_true')  # to remove ES featrues
    parser.add_argument('--wo_MDP', action='store_true', default=True) # always True (disabled function)
    
    ### resume and load models
    parser.add_argument('--load_path', default = None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None, help='Resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start at epoch # (relevant for learning rate decay)')
    
    ### training AND validation
    parser.add_argument('--K_epochs', type=int, default=3)
    parser.add_argument('--eps_clip', type=float, default=0.1)
    parser.add_argument('--T_train', type=int, default=200)
    parser.add_argument('--n_step', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_end', type=int, default=200, help='End at epoch #')
    parser.add_argument('--epoch_size', type=int, default=10240, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='Number of instances per batch used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default = None, help='Dataset file to use for validation')
    parser.add_argument('--lr_model', type=float, default=8e-5, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=2e-5, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.985, help='Learning rate decay per epoch')
    parser.add_argument('--warm_up', type=float, default=2) # the rho in the paper
    parser.add_argument('--max_grad_norm', type=float, default=0.05, help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    
    ### network
    parser.add_argument('--v_range', type=float, default=6.)
    parser.add_argument('--critic_head_num', type=int, default=4)
    parser.add_argument('--actor_head_num', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    parser.add_argument('--normalization', default='layer', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--gamma', type=float, default=0.999, help='decrease future reward')
    parser.add_argument('--T_max', type=int, default=1000, help='number of steps to swap')
    
    ### logs to tensorboard and screen
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=50,help='Log info every log_step steps')

    ### outputs
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)
    if opts.problem == 'tsp':
        opts.wo_feature1 = opts.wo_feature2 = opts.wo_feature3 = opts.wo_bonus = opts.wo_regular = True
    
    ### figure out whether to use distributed training
    opts.world_size = torch.cuda.device_count()
    opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) if not opts.no_saving else None
    return opts