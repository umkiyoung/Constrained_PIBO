from baselines.models import *
import torch
from baselines.gfn_folder.gfn_train import *


class DiffusionSampler:
    def __init__(self, energy, prior, gfn_sampler, buffer, buffer_ls, device, batch_size, args, beta=1):
        self.energy = energy
        self.prior = prior
        self.sampler = gfn_sampler
        self.beta = beta
        self.bsz = batch_size
        self.buffer = buffer
        self.buffer_ls = buffer_ls
        self.args = args
        self.device = device
        
    def train(self, i):
        loss = train_step(
            self.energy,
            self.sampler,
            i,
            self.args.exploratory,
            self.buffer,
            self.buffer_ls,
            self.args.exploration_factor,
            self.args.exploration_wd,
            self.args,
            self.device,
        )
        return loss

    def sample(self, batch_size, track_gradient):
        """
        Directly sample from z ~ sampler, return f(z)
        """
        z = self.sampler.sample(batch_size, self.energy.log_reward)
        x = self.prior.sample_with_noise(z, track_gradient)
        return x
    

class Energy():
    """
    Directly compute the r(f(z))
    """
    def __init__(self, proxy, prior, beta):
        self.proxy = proxy
        self.prior = prior
        self.beta = beta
        
    def log_reward(self, z):
        reward = reward = torch.distributions.Normal(loc=0, scale=1).log_prob(z).sum(dim=1)
        reward += self.proxy.log_reward(self.prior.sample_with_noise(z), beta=self.beta)
        return reward