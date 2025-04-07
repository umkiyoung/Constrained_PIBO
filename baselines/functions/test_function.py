from typing import Tuple

from botorch.test_functions import Ackley, Rastrigin, Levy, Rosenbrock
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.quasirandom import SobolEngine
from baselines.functions.rover_planning import Rover

class TestFunction(Dataset):
    def __init__(self, task: str, dim: int = 200, n_init: int = 200, seed: int = 0, dtype=torch.float64, device='cpu', negate=True,):
        self.task = task
        self.dim = dim
        self.n_init = n_init
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.lb, self.ub = None, None
        self.constraints = []
        #NOTE: Synthetic Functions
        if task == 'Ackley':
            # Constrain Settings:
            # c1(x) = ∑10  i=1 xi ≤ 0 and c2(x) = ‖x‖2 − 5 ≤ 0. (SCBO)
            self.fun = Ackley(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 10 #Following TurBO
            
            def c1(x):
                return torch.sum(x, dim=-1) - 0
            def c2(x):
                return torch.norm(x, p=2, dim=-1) - 5
            
            def eval_c1(x):
                return c1(unnormalize(x, self.fun.bounds))
            def eval_c2(x):
                return c2(unnormalize(x, self.fun.bounds))
            
            self.constraints.append((c1, eval_c1))
            self.constraints.append((c2, eval_c2))
            

        elif task == 'Rastrigin':
            self.fun = Rastrigin(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 5 #Following MCMC_BO
            assert False
            
        elif task == 'Levy':
            self.fun = Levy(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -10, 10 #Following LA-MCTS
            assert False

        elif task == 'Rosenbrock':
            self.fun = Rosenbrock(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 10 #Following LA-MCTS
            assert False

        elif task == 'RoverPlanning':
            self.fun = Rover(dim=dim, dtype=dtype, device=device)
            self.lb, self.ub = 0, 1
            assert False

        else:
            raise ValueError(f"Unknown task: {task}")
        
        if self.lb is not None and self.ub is not None:
            self.fun.bounds[0, :].fill_(self.lb)
            self.fun.bounds[1, :].fill_(self.ub)
            self.fun.bounds.to(dtype=dtype, device=device)
                
    def eval_objective(self, x):
        return self.fun(unnormalize(x, self.fun.bounds))
    
    def eval_constraints(self, x):
        c_list = []
        for c, eval_c in self.constraints:
            c_list.append(eval_c(x))
        c_list = torch.stack(c_list, dim=-1)
        return c_list.unsqueeze(0) if c_list.ndim == 1 else c_list
    
    def eval_objective_with_constraints(self, x):
        y = self.eval_objective(x)
        c_list = []
        for c, eval_c in self.constraints:
            c_list.append(eval_c(x))
        c_list = torch.stack(c_list, dim=-1)
        return y, c_list
    
    def eval_score(self, x):
        y = self.eval_objective(x)
        c_list = []
        for c, eval_c in self.constraints:
            c_list.append(eval_c(x))
        c_list = torch.stack(c_list, dim=-1)
        
        # if any constraint is violated, return a -inf
        if torch.any(c_list > 0):
            return -float('inf')
        else:
            return y.item()
        
    def get_initial_points(self):
        sobol = SobolEngine(self.dim, scramble=True, seed=self.seed)
        self.X = sobol.draw(n=self.n_init).to(self.dtype).to(self.device)
        self.Y = torch.tensor([self.eval_objective(x) for x in self.X], dtype=self.dtype, device=self.device).unsqueeze(-1)
        self.C = torch.cat([self.eval_constraints(x) for x in self.X], dim=0).to(self.dtype).to(self.device)
        return self.X, self.Y, self.C
    
    def normalizing_constraints(self, constraints, mean, std):
        normalized = (constraints - mean) / (std + 1e-7)
        zero_normalized = -mean / std
        return normalized - zero_normalized
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.C[idx]
    
if __name__ == "__main__":
    test_function = TestFunction(task='Ackley', dim=20, n_init=10, seed=0, dtype=torch.float64, device='cpu')
    test_function.get_initial_points()
    print(test_function.X, test_function.X.shape)
    print(test_function.Y, test_function.Y.shape)
    print(test_function.C, test_function.C.shape)