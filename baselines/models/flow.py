import torch
from torch import nn, Tensor
class FlowModel(nn.Module):
    def __init__(self, x_dim: int = 2, hidden_dim: int = 512, device: str = 'cpu', dtype: torch.dtype = torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, x_dim)
        )
        self.x_dim = x_dim
        self.loss_function = nn.MSELoss()
        self.device = device
        self.dtype = dtype

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2,
                                          t_start + (t_end - t_start) / 2)
    
    def sample(self, batch_size: int, step_size: int, track_gradient: bool = False) -> Tensor:
        x = torch.randn(batch_size, self.x_dim, device=self.device, dtype=self.dtype)
        time_steps = torch.linspace(0, 1.0, step_size + 1, device=self.device, dtype=self.dtype)
        
        for i in range(step_size):
            x = self.step(x, time_steps[i], time_steps[i + 1])
            if track_gradient == False:
                x = x.detach()
        return x
    # add
    def sample_with_noise(self, z, step_size = 10, track_gradient: bool = False ):
        time_steps = torch.linspace(0, 1.0, step_size + 1, device=self.device, dtype=self.dtype)
        
        for i in range(step_size):
            nest_z = self.step(z, time_steps[i], time_steps[i + 1])
            if track_gradient == False:
                z = nest_z.detach()
        x = z
        return x

    def compute_loss(self, x_1: Tensor) -> Tensor:
        # Compute the loss for the flow model
        x_0 = torch.randn_like(x_1, device=self.device, dtype=self.dtype)
        t = torch.rand(len(x_1), 1, device=self.device, dtype=self.dtype)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        loss = self.loss_function(self(x_t, t), dx_t)
        return loss
    