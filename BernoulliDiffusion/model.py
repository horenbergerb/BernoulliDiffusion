import torch
from torch import nn
from torch.distributions import Binomial
import torch.optim as optim
from numpy.random import default_rng
import numpy as np

from BernoulliDiffusion.utils.math_utils import kl_div, entropy_of_q_conditional, entropy_of_prior

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


class BernoulliDiffusionModel(nn.Module):
    def __init__(self, sequence_length, num_sample_steps, T):
        super().__init__()

        self.model = ReverseModel(sequence_length, T).to(device)
        
        self.sequence_length = sequence_length
        self.num_sample_steps = num_sample_steps

        self.T = T

        # calculate the array of beta_tilde_t values from t=1 to t=T
        self.beta_tilde_t = [torch.zeros((self.sequence_length)).to(device)]
        for cur_t in range(1, self.T+1):
            self.beta_tilde_t.append((self.beta_tilde_t[cur_t-1] + self.beta_t(cur_t) - (self.beta_t(cur_t)*self.beta_tilde_t[cur_t-1])))
        self.beta_tilde_t = torch.stack(self.beta_tilde_t)

        self.H_start = entropy_of_q_conditional(self.sequence_length, self.beta_tilde_t[1,0].item())
        self.H_end = entropy_of_q_conditional(self.sequence_length, self.beta_tilde_t[self.T, 0].item())
        self.H_prior = entropy_of_prior(self.sequence_length)

    def p_conditional_prob(self, x_t, t):
        '''Returns the probabilities of the Bernoulli variables for the reverse process from t to t-1,
        i.e. p(x_(t-1)|x_t)'''
        return self.model(x_t, t)

    @torch.no_grad()
    def p_step(self, x, t):
        '''Performs reverse process on x from t to t-1'''
        return torch.bernoulli(self.model(x, t))

    @torch.no_grad()
    def p_sample(self, batch_size, x=None):
        '''Performs complete reverse process on a batch of noise'''
        if x is None:
            init_prob = torch.empty((batch_size, self.sequence_length)).fill_(0.5).to(device)
            x = torch.bernoulli(init_prob)

        for cur_t in range(self.T, 0, -1):
            x = torch.bernoulli(self.p_conditional_prob(x, cur_t))
        return x

    def beta_t(self, t):
        '''The bit flip probability at step t of the forward process'''
        return 1.0/(self.T-t+1)

    def q_conditional_prob(self, x_t, t):
        '''Returns the probabilities of the Bernoulli variables for the forward process
        with x_t as input while going from t to t+1, i.e. q(x_(t+1)|x_t)'''
        # had to change the beta_t equation here to keep indexing consistent
        return (x_t * (1.0 - self.beta_t(t+1))) + 0.5 * self.beta_t(t+1)

    def q_conditional_prob_wrt_x_0(self, x_0, t):
        '''Returns the probabilities of the Bernoulli variables for observing samples of x_t
        given x_0, i.e. q(x_t|x_0)'''
        beta_tilde_t = self.beta_tilde_t[t].expand(x_0.size())
        return ((x_0 * (1.0 - beta_tilde_t)) + 0.5 * beta_tilde_t)

    @torch.no_grad()
    def q_step(self, x, t):
        '''Performs forward process on x from t to t+1'''
        probs = self.q_conditional_prob(x, t)
        return torch.bernoulli(probs)

    def q_sample(self, x_0, t):
        '''Returns a sample x_t given input x_0'''
        return torch.bernoulli(self.q_conditional_prob_wrt_x_0(x_0, t))
    
    def forward(self, x_0):
        '''Approximates the loss via equation 13 in Deep Unsupervised Learning using Nonequilibrium Thermodynamics
        using samples from the reverse process.'''
        # the monte carlo sampling is performed using the minibatch
        total_loss = torch.zeros((x_0.size(dim=0),)).to(device)
        for t in range(1, self.T + 1):
            x_t = self.q_sample(x_0, t)
            beta_t = self.beta_t(t)
            posterior = x_0*(1-self.beta_tilde_t[t-1]) + 0.5*self.beta_tilde_t[t-1]
            posterior *= x_t * (1-0.5*beta_t) + (1 - x_t) * (1.5*beta_t)
            kl_divergence = kl_div(posterior,
                                   self.p_conditional_prob(x_t, t))

            total_loss += kl_divergence + self.H_start - self.H_end + self.H_prior
        return torch.mean(total_loss)
