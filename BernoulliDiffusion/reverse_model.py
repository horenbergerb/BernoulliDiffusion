import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReverseModel(nn.Module):
    def __init__(self, sequence_length, T):
        super().__init__()

        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(sequence_length, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU()
        )

        self.output_layers = nn.ModuleList([torch.nn.Linear(50, sequence_length) for x in range(T)])
        self.outputs_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, t):
        x = self.shared_layers(x)
        x = self.output_layers[t-1](x)
        x = self.outputs_sigmoid(x)
        return x
