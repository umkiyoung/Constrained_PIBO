import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from baselines.models.gfn import *


from baselines.models.flow import FlowModel
from baselines.functions.test_function import TestFunction
from baselines.models.value_functions import ProxyEnsemble, Proxy 
from baselines.utils import save_numpy_array, set_seed, get_value_based_weights, get_rank_based_weights
from baselines.gfn_folder.buffer import load_buffer
from baselines.gfn_folder.plot_utils import *
from baselines.gfn_folder.energies import *
from baselines.gfn_folder.gfn_train import *
from baselines.gfn_folder.gfn_utils import *


import wandb
from baselines.models.diffusion_sampler import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################################################################
    parser.add_argument('--lr_policy', type=float, default=1e-3)
    parser.add_argument('--lr_flow', type=float, default=1e-2)
    parser.add_argument('--lr_back', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--s_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--harmonics_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=300) #NOTE 100
    parser.add_argument('--epochs', type=int, default=25000)
    parser.add_argument('--buffer_size', type=int, default=300 * 1000 * 2) #NOTE 1000
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--subtb_lambda', type=int, default=2)
    parser.add_argument('--t_scale', type=float, default=5.)
    parser.add_argument('--log_var_range', type=float, default=4.)
    parser.add_argument('--energy', type=str, default='9gmm',
                        choices=('9gmm', '25gmm', 'hard_funnel', 'easy_funnel', 'many_well'))
    parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
    parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
    parser.add_argument('--both_ways', action='store_true', default=False)

    # For local search
    ################################################################
    parser.add_argument('--local_search', action='store_true', default=False)

    # How many iterations to run local search
    parser.add_argument('--max_iter_ls', type=int, default=200)

    # How many iterations to burn in before making local search
    parser.add_argument('--burn_in', type=int, default=100)

    # How frequently to make local search
    parser.add_argument('--ls_cycle', type=int, default=100)

    # langevin step size
    parser.add_argument('--ld_step', type=float, default=0.001)

    parser.add_argument('--ld_schedule', action='store_true', default=False)

    # target acceptance rate
    parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


    # For replay buffer
    ################################################################
    # high beta give steep priorization in reward prioritized replay sampling
    parser.add_argument('--beta', type=float, default=1.)

    # low rank_weighted give steep priorization in rank-based replay sampling
    parser.add_argument('--rank_weight', type=float, default=1e-2)

    # three kinds of replay training: random, reward prioritized, rank-based
    parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
    ################################################################

    parser.add_argument('--bwd', action='store_true', default=False)
    parser.add_argument('--exploratory', action='store_true', default=False)

    parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
    parser.add_argument('--langevin', action='store_true', default=False)  
    parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
    parser.add_argument('--conditional_flow_model', action='store_true', default=False)
    parser.add_argument('--learn_pb', action='store_true', default=False)
    parser.add_argument('--pb_scale_range', type=float, default=0.1)
    parser.add_argument('--learned_variance', action='store_true', default=False)
    parser.add_argument('--partial_energy', action='store_true', default=False)
    parser.add_argument('--exploration_factor', type=float, default=0.1)
    parser.add_argument('--exploration_wd', action='store_true', default=False)
    parser.add_argument('--clipping', action='store_true', default=False)
    parser.add_argument('--lgv_clip', type=float, default=1e2)
    parser.add_argument('--gfn_clip', type=float, default=1e4)
    parser.add_argument('--zero_init', action='store_true', default=False)
    parser.add_argument('--pis_architectures', action='store_true', default=False)
    parser.add_argument('--lgv_layers', type=int, default=3)
    parser.add_argument('--joint_layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--use_weight_decay', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
    
    set_seed(args.seed)
    
    def get_energy(args, device):
        if args.energy == '9gmm':
            energy = NineGaussianMixture(device=device)
        elif args.energy == '25gmm':
            energy = TwentyFiveGaussianMixture(device=device)
        elif args.energy == 'hard_funnel':
            energy = HardFunnel(device=device)
        elif args.energy == 'easy_funnel':
            energy = EasyFunnel(device=device)
        elif args.energy == 'many_well':
            energy = ManyWell(device=device)
        return energy
    
        
    eval_data_size = 2000
    final_eval_data_size = 2000
    plot_data_size = 2000
    final_plot_data_size = 2000
    
    if args.pis_architectures:
        args.zero_init = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

    if args.both_ways and args.bwd:
        args.bwd = False

    if args.local_search:
        args.both_ways = True

    name = get_name(args)
    energy = get_energy(args, device)
    eval_data = energy.sample(eval_data_size).to(device)
    
    args.dim = energy.data_ndim
    
    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="gfn-diffusion", config=config, name=name)
    
    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)
    
    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    print(gfn_model)
    metrics = dict()

    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    gfn_model.train()
    
    for i in trange(args.epochs+1):
        loss =  train_step(energy, gfn_model, i, args.exploratory,
                                           buffer, buffer_ls, args.exploration_factor, args.exploration_wd, args, device)    
        metrics['train/loss'] = loss.item()
        loss.backward()
        gfn_optimizer.step()
        if i % 100 == 0:
            metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False, eval_data_size = eval_data_size, device = device, args=args))
            images = plot_step(energy=energy, gfn_model=gfn_model, name=name, args=args, wandb=wandb, device=device, plot_data_size=plot_data_size)
            metrics.update(images)
            plt.close('all')
            wandb.log(metrics, step=i)                

    eval_results = final_eval(energy, gfn_model, final_eval_data_size, eval_step).to(device)
    metrics.update(eval_results)