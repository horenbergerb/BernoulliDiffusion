import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReverseModel(nn.Module):
    def __init__(self, sequence_length, T, skip_init=5.0):
        super().__init__()

        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(sequence_length, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU()
        )

        self.output_layers = nn.ModuleList([torch.nn.Linear(50, sequence_length) for x in range(T)])
        self.outputs_sigmoid = torch.nn.Sigmoid()

        # Skip connection: each reverse step is nearly the identity (bit flip probabilities are ~1/T),
        # which the MLP can't represent through its bottleneck. A learned per-step scale on x_t makes
        # copying the input the default, so the network only has to learn corrections.
        # skip_init=5.0 starts each step at sigmoid(5) ~= 99.3% probability of keeping each bit.
        self.skip_scale = nn.Parameter(torch.full((T,), skip_init))

    def forward(self, x, t):
        logits = self.output_layers[t-1](self.shared_layers(x))
        logits = logits + self.skip_scale[t-1] * (2.0 * x - 1.0)
        return self.outputs_sigmoid(logits)
