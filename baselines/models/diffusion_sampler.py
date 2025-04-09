from baselines.models import *
import torch
from baselines.gfn_folder.gfn_train import *



class DiffusionSampler():
    def __init__(self, Flow, Proxy_Ensemble, gfn_sampler, gfn_optimizer, buffer, buffer_ls, device, batch_size,  args, beta = 1):
        self.prior = Flow
        self.proxy = Proxy_Ensemble
        self.sampler = gfn_sampler
        self.beta = beta
        self.bsz = batch_size
        self.optimizer = gfn_optimizer
        self.buffer = buffer
        self.buffer_ls = buffer_ls
        self.args = args
        self.device = device
    def forward_tb(self,i):
        loss = train_step(self.proxy, self.sampler, self.optimizer , i, self.args.exploratory,
                                                self.buffer, self.buffer_ls, self.args.exploration_factor, self.args.exploration_wd, self.prior,self.args , self.device)
        return loss
    
  
    def sample(self, batch_size, fuck, step_size, track_gradient):
        # Sample from the gfn_sampler
        z = self.sampler.sample(batch_size, fuck)
        x = self.prior.sample_with_noise(z, step_size, track_gradient)
        return x
        
        
    
    # def forward_tb(self):
    #     states, log_pfs, log_pbs, log_fs = self.sampler.sample(self.bsz, self.proxy)
    #     x = states[:,-1]
    #     # tb loss
    #     with torch.no_grad():
    #         log_r = self.reward_call(x).detach()

    #     loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)
    #     return loss.mean()

    # def backward_tb():
    #     # tb backward
    #     self.gfn_sampler.backward_tb()


    def reward_call(self, z):
        reward = torch.distribution.Normal(loc=0, scale=1).log_prob(z)
        reward += torch.exp(self.beta * self.proxy(self.prior(z)))
        return reward
    
    # def sample(self, batch_size, step_szie):
    #     x = torch.randn(batch_size, self.x_dim, device=self.device, dtype=self.dtype)
    #     x = gfn_sampler.sample(x)
    #     x = self.prior(x)
    #     return x