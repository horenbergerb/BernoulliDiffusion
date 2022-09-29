import torch
from numpy.random import default_rng


rng = default_rng()


def sample_heartbeat(period: int = 5, sequence_length: int = 20) -> torch.Tensor:
    '''
    Creates random 'Heartbeat' data sequences. Here are two examples with period=5 and sequence_length=20:
    [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]
    [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0]
    '''
    sample = torch.zeros(sequence_length)
    offset = rng.integers(period)
    sample[offset::period] = 1
    return sample


def generate_batch(num_samples: int = 10,
                     period: int = 5,
                     sequence_length: int = 20) -> torch.Tensor:
    '''Creates a batch of heartbeat sequences'''
    x_0 = [sample_heartbeat(period, sequence_length) for index in range(num_samples)]
    return torch.vstack(x_0)
