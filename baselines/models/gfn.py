import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from baselines.gfn_folder.gfn_utils import gaussian_params
import torch
import numpy as np
from einops import rearrange
from torch import nn
import math
logtwopi = math.log(2 * math.pi)

class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel, dtype=dtype)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel, dtype=dtype)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel, dtype=dtype),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel, dtype=dtype),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim, dtype=dtype),
        )

        # Initialize the last layer's weights and biases
        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.to(self.dtype)) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.to(self.dtype)) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(
            self,
            in_shape=2,
            out_shape=2,
            num_layers=2,
            channels=128,
            zero_init=True,
            dtype=torch.float32,
    ):
        super().__init__()

        self.dtype = dtype
        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels, dtype=dtype)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels, dtype=dtype)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels, dtype=dtype)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels, dtype=dtype),
            nn.GELU(),
            nn.Linear(channels, channels, dtype=dtype),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels, dtype=dtype), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape)), dtype=dtype),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.to(self.dtype)) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.to(self.dtype)) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64, dtype=torch.float32):
        super(TimeEncoding, self).__init__()

        self.dtype = dtype
        pe = torch.arange(1, harmonics_dim + 1, dtype=dtype).unsqueeze(0) * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, dtype=dtype),
            nn.GELU()
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = (t.to(self.dtype) * self.pe).sin()
        t_cos = (t.to(self.dtype) * self.pe).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncoding(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64, dtype=torch.float32):
        super(StateEncoding, self).__init__()

        self.dtype = dtype
        self.x_model = nn.Sequential(
            nn.Linear(s_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim, dtype=dtype),
            nn.GELU()
        )

    def forward(self, s):
        return self.x_model(s.to(self.dtype))


class JointPolicy(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 zero_init: bool = False, dtype=torch.float32):
        super(JointPolicy, self).__init__()
        self.dtype = dtype
        if out_dim is None:
            out_dim = 2 * s_dim

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, dtype=dtype)
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(torch.cat([s.to(self.dtype), t.to(self.dtype)], dim=-1))
