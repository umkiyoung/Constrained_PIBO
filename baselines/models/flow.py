import torch
from torch import nn, Tensor
class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim)
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2,
                                          t_start + (t_end - t_start) / 2)
    

import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_swiss_roll
from tqdm import tqdm
import wandb  # wandb import 추가
from model import Flow
import argparse
import yaml
from util import sample_visual
# wandb 초기화
wandb.init(project="PIBO", name="Flow-matching")

import torch
import matplotlib.pyplot as plt

def sample_visual(flow, sample_size, dim, step_size):
    x = torch.randn(sample_size, dim)
    fig, axes = plt.subplots(1, step_size + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, step_size + 1)

    axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[0].set_title(f't = {time_steps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)

    for i in range(step_size):
        x = flow.step(x, time_steps[i], time_steps[i + 1])
        axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

    plt.tight_layout()
    plt.show()
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--train_sample_size', type=int, default=256)
    parser.add_argument('--test_sample_size', type=int, default=300)
    parser.add_argument('--step_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--dataset', type=str, choices=["moons", "swiss"], default="moons")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--config', default=None, type=str, help='path to config file,'
                        ' and command line arguments will be overwritten by the config file')
    
    args = parser.parse_args()

    if args.config:
        with open("config.yaml", "r") as f:
            cfg_dict = yaml.safe_load(f)

            for key, val in cfg_dict.items():
                assert hasattr(args, key), f'Unknown config key: {key}'
                setattr(args, key, val)
            f.seek(0)
            print(f'Config file: {args.config}', )
            for line in f.readlines():
                print(line.rstrip())

    return args


if __name__ == '__main__':

    args = parse_arguments()
    # training
    flow = Flow()
    optimizer = torch.optim.Adam(flow.parameters(), args.learning_rate)
    loss_fn = nn.MSELoss()

    for _ in tqdm(range(args.epochs)):
        train_data = make_moons(args.train_sample_size, noise=0.05)[0] # shape : (count, dim)
        x_1 = Tensor(train_data)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(len(x_1), 1)

        x_t = (1-t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        optimizer.zero_grad()
        loss = loss_fn(flow(x_t, t), dx_t)
        loss.backward()
        
        # logging
        wandb.log({"loss": loss.item(), "epoch": _})
        optimizer.step()

    # save model
    torch.save(flow.state_dict(), "./result/flow_model.pth")

    # sampling visualization
    sample_visual(flow, args.test_sample_size, train_data.shape[1], args.step_size)

    wandb.finish()