import torch
import torch.nn as nn
import torch.nn.functional as F

class Proxy(nn.Module):
    def __init__(self, x_dim, hidden_dim=1024, num_hidden_layers = 1, output_dim=1, dtype = torch.float32):
        super(Proxy, self).__init__()
        self.gamma = 1.0
        self.dtype = dtype
        
        self.input_layer = nn.Linear(x_dim, hidden_dim, dtype=dtype)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, dtype=dtype) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=dtype)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.gelu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.gelu(hidden_layer(x))
        return self.output_layer(x)
    
    def uncertainty_forward(self, x):
        y = self(x)
        return y, torch.zeros_like(y)
    
    def compute_loss(self, x, y, c=None):
        pred = self(x)
        concat_yc = torch.cat([y, c], dim=-1) if c is not None else y
        loss = F.mse_loss(pred, concat_yc)
        return loss

    def log_reward(self, x, beta=1.0):
        pred_y = self(x).squeeze()
        r = pred_y * beta
        return r
    
    def score(self, x, beta=1.0):
        x = x.detach()
        x.requires_grad_(True)
        r = self.log_reward(x, beta=beta)
        score = torch.clamp(torch.autograd.grad(r.sum(), x)[0], -100, 100)
        return score.detach()

class ProxyEnsemble(nn.Module):
    def __init__(self, x_dim, hidden_dim=1024, num_hidden_layers=1, output_dim=1, n_ensembles=5, ucb_reward=False, constraint_formulation="Soft", lamb=1.0):
        super(ProxyEnsemble, self).__init__()
        self.n_ensembles = n_ensembles
        self.models = nn.ModuleList([Proxy(x_dim, hidden_dim, num_hidden_layers, output_dim) for _ in range(n_ensembles)])
        self.ucb_reward = ucb_reward
        self.gamma = 1.0
        self.lamb = lamb
        if self.n_ensembles == 1:
            self.ucb_reward = False
            self.gamma = 0.0
        
        # Soft/Log barrier/Lagrangian formulation
        self.constraint_formulation = constraint_formulation

        
    def forward(self, x):
        if self.n_ensembles == 1:
            return self.models[0](x)
        return torch.stack([model(x) for model in self.models], dim=0).mean(dim=0)
    
    def uncertainty_forward(self, x):
        stacked = torch.stack([model(x) for model in self.models], dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std
    
    def log_reward(self, x, beta=1.0):
        if self.ucb_reward:
            mean, std = self.uncertainty_forward(x)
            
            pred_y = mean[:, 0] + self.gamma * std[:, 0]
            pred_y = pred_y.squeeze()
            
            if self.constraint_formulation == "Soft":
                pred_c = self.soft_constraint(mean[:, 1:], self.lamb)
                pred_y = pred_y + pred_c
                
            elif self.constraint_formulation == "LogBarrier":
                pred_c = self.log_barrier_constraint(mean[:, 1:], self.lamb)
                pred_y = pred_y + pred_c
                
            elif self.constraint_formulation == "Lagrangian":
                pred_c = self.lagrangian_constraint(mean[:, 1:], self.lamb)
                pred_y = pred_y + pred_c
            
            else:
                raise ValueError(f"Unknown constraint formulation: {self.constraint_formulation}")
        
        else:
            pred_y = self(x).squeeze()
        
        r = pred_y * beta
        return r
    
    def score(self, x, beta=1.0):
        x = x.detach()
        x.requires_grad_(True)
        r = self.log_reward(x, beta=beta)
        score = torch.clamp(torch.autograd.grad(r.sum(), x)[0], -100, 100)
        return score.detach()
    
    def soft_constraint(self, mean, lamb):
        ##\mathbb{I}[\hat{g}_i(\mathbf{x})\leq0] \approx \sigma(-\lambda \cdot \hat{g}_i(\mathbf{x}))##
        return -torch.sum(1 - torch.sigmoid(-lamb * mean), dim=-1)
    
    def log_barrier_constraint(self, mean, lamb):
        return - torch.sum(torch.log(1 + torch.exp(lamb * mean)), dim=-1)
    
    def lagrangian_constraint(self, mean, lamb):
        return - torch.sum(lamb * torch.max(mean, torch.zeros_like(mean)), dim=-1)
        
    
class ProxyMCDropout(nn.Module):
    def __init__(self, x_dim, hidden_dim=1024, num_hidden_layers=1, dropout_rate=0.01, dtype = torch.float32):
        super(ProxyMCDropout, self).__init__()
        self.gamma = 1.0
        self.dtype = dtype
        
        self.input_layer = nn.Linear(x_dim, hidden_dim, dtype=dtype)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, dtype=dtype) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.gelu(self.input_layer(x))
        x = self.dropout(x)
        for hidden_layer in self.hidden_layers:
            x = self.gelu(hidden_layer(x))
            x = self.dropout(x)
        return self.output_layer(x)
    
    def compute_loss(self, x, y):
        pred_y = self(x)
        loss = F.mse_loss(pred_y, y)
        return loss
    
    def uncertainty_forward(self, x, n_samples=10):
        samples = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std
    
    def log_reward(self, x, beta=1.0):
        mean, std = self.uncertainty_forward(x)
        pred_y = mean + std
        pred_y = pred_y.squeeze()
        r = pred_y * beta
        return r
    
    def score(self, x, beta=1.0):
        x = x.detach()
        x.requires_grad_(True)
        r = self.log_reward(x, beta=beta)
        score = torch.clamp(torch.autograd.grad(r.sum(), x)[0], -100, 100)
        return score.detach()



        