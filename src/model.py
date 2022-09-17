import torch
from torch import nn
from torch.distributions import Binomial
import torch.optim as optim
from numpy.random import default_rng


rng = default_rng()

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


class BinomialDiffusion(nn.Module):
    def __init__(self, model, sequence_length, num_sample_steps, T):
        super().__init__()

        self.model = model
        
        self.sequence_length = sequence_length
        self.num_sample_steps = num_sample_steps

        self.T = T

    def p_bit_flip_prob(self, x, t):
        '''Returns probability that each bit of x will flip during reverse process from t to t-1'''
        return self.model(x, t)

    def p_sample(self, x, t):
        '''Performs reverse process on x from t to t-1. Returns result and the result's
        respective probability'''
        bit_flip_prob = self.p_bit_flip_prob(x, t)
        m = Binomial(1, bit_flip_prob)
        return m.sample().to(device), torch.abs(1 - x - bit_flip_prob)

    def p_prob(self, x_next, x_prev, t):
        '''Returns the probability of observing x_prev as output when x_next is input to the reverse process
        while going from t to t-1'''
        return torch.prod(torch.abs(1.0 - x_prev - self.p_bit_flip_prob(x_next, t)), dim=1)

    def p_sample_loop(self, batch_size):
        '''Performs complete reverse process on a batch of noise'''
        bit_flip_prob = torch.empty((batch_size, self.sequence_length)).fill_(0.5).to(device)
        m = Binomial(1, bit_flip_prob)
        x = m.sample().to(device)

        for t in range(2000, 0):
            x = self.p_sample(x, t)

        return x

    def q_bit_flip_prob(self, x, t):
        '''Returns probability that each bit of x will flip during forward process from t to t+1'''
        beta_t = 1.0/(self.T-t+1.0)
        return (x * (1.0 - beta_t)) + 0.5*beta_t

    def q_sample(self, x, t):
        ''''Performs forward process on x from t to t+1. Returns result and the result's
        respective probability'''
        bit_flip_prob = self.q_bit_flip_prob(x, t)
        m = Binomial(1, bit_flip_prob)
        x = m.sample().to(device)
        return x, torch.prod(torch.abs(1.0 - x - bit_flip_prob), dim=1)

    def forward(self, x):
        '''Approximates loss via equation 11 in Deep Unsupervised Learning using Nonequilibrium Thermodynamics
        using samples from the reverse process'''
        #avg_loss is the monte carlo approximation of loss
        avg_loss = torch.zeros((x.size(dim=0),)).to(device)
        for n in range(self.num_sample_steps):
            print('n: {}'.format(n))
            cur_loss = torch.zeros((x.size(dim=0),)).fill_(1.0).to(device)
            x_prev = x
            for t in range(0, self.T):
                # sample forward using q
                x_next, q_prob = self.q_sample(x_prev, t)
                cur_loss *= self.p_prob(x_next, x_prev, t+1)/q_prob
                x_prev = x_next
            avg_loss += cur_loss
        avg_loss = avg_loss/self.num_sample_steps
        # Negative sign so we can minimize loss
        return -1.0 * torch.log(avg_loss)

    def funky_forward(self, x):
        '''Similar to the other forward function, but it just uses one random p/q value
        per sample step instead of taking the product of all of them'''
        #avg_loss is the monte carlo approximation of loss
        avg_loss = torch.zeros((x.size(dim=0),)).to(device)
        for n in range(self.num_sample_steps):
            print('n: {}'.format(n))
            cur_loss = torch.zeros((x.size(dim=0),)).fill_(1.0).to(device)
            chosen_t = rng.integers(1, self.T)
            x_prev = x
            for t in range(0, self.T):
                # sample forward using q
                x_next, q_prob = self.q_sample(x_prev, t)
                if t == chosen_t:
                    avg_loss += self.p_prob(x_next, x_prev, t+1)/q_prob
                    break
                x_prev = x_next
        avg_loss = avg_loss/self.num_sample_steps
        # Negative sign so we can minimize loss
        return -1.0 * torch.log(avg_loss)
