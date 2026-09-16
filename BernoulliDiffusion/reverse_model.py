import math

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReverseModel(nn.Module):
    def __init__(self, sequence_length, T, hidden_size=50, num_hidden_layers=2, time_features=32, skip_init=5.0):
        super().__init__()

        self.T = T
        self.time_features = time_features

        # The timestep is fed in as sinusoidal features rather than selecting one of T output heads.
        # Training samples a random t per example, so a per-t head would only rarely receive gradient;
        # a shared network conditioned on t generalizes across neighboring timesteps.
        self.time_layers = torch.nn.Sequential(
            torch.nn.Linear(time_features, hidden_size),
            torch.nn.ReLU()
        )

        self.input_layer = torch.nn.Linear(sequence_length, hidden_size)
        hidden_layers = []
        for _ in range(num_hidden_layers - 1):
            hidden_layers += [torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size)]
        self.shared_layers = torch.nn.Sequential(*hidden_layers, torch.nn.ReLU())
        self.output_layer = torch.nn.Linear(hidden_size, sequence_length)
        self.outputs_sigmoid = torch.nn.Sigmoid()

        # Skip connection: each reverse step is nearly the identity (bit flip probabilities are ~1/T),
        # which the MLP can't represent through its bottleneck. A learned, t-dependent scale on x_t makes
        # copying the input the default, so the network only has to learn corrections.
        # skip_init=5.0 starts each step at sigmoid(5) ~= 99.3% probability of keeping each bit.
        self.skip_layers = torch.nn.Sequential(
            torch.nn.Linear(time_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
        nn.init.zeros_(self.skip_layers[-1].weight)
        nn.init.constant_(self.skip_layers[-1].bias, skip_init)

    def embed_time(self, t):
        '''Sinusoidal features of the timestep, shape (batch, time_features)'''
        half = self.time_features // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, x, t):
        '''t is either an int shared by the whole batch or a tensor with one timestep per example'''
        if not torch.is_tensor(t):
            t = torch.full((x.size(0),), t, device=x.device)
        t_emb = self.embed_time(t)

        h = self.shared_layers(self.input_layer(x) + self.time_layers(t_emb))
        logits = self.output_layer(h) + self.skip_layers(t_emb) * (2.0 * x - 1.0)
        return self.outputs_sigmoid(logits)
