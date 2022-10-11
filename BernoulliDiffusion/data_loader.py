import torch
from torch.distributions import Binomial

import os

from BernoulliDiffusion.utils.data_utils import load_data_from_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataLoader:
    '''Loads training data and generates minibatches to
    be iterated over during training.'''
    def __init__(self, working_dir, cfg):
        self.cfg = cfg

        self.data = load_data_from_file(os.path.join(working_dir, 'train.txt'))
        self.num_data = self.data.size()[0]
        self.sequence_length = self.data.size()[1]

        # permute so that we can grab random minibatches
        self.permutation = torch.randperm(self.num_data)

        self.cur_ind = 0

    def next_minibatch(self):
        if self.cur_ind < self.num_data:
            indices = self.permutation[self.cur_ind:self.cur_ind+self.cfg.batch_size]
            self.cur_ind += self.cfg.batch_size
            return self.data[indices]
        else:
            self.cur_ind = 0
            return None

    def generate_random_data(self):
        '''Generates a random tensor with the same shape as training data.
        Used for initialization'''
        m = Binomial(1, torch.zeros((self.cfg.batch_size, self.sequence_length)).fill_(0.5))
        return m.sample().to(device)

    def get_sequence_length(self):
        return self.sequence_length