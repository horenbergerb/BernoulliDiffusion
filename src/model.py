import torch
from torch import nn
from torch.distributions import Binomial
import torch.optim as optim
from numpy.random import default_rng
import numpy as np

# KL Divergence of multivariate bernoulli distributions:
# https://math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution


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


class BernoulliDiffusion(nn.Module):
    def __init__(self, model, sequence_length, num_sample_steps, T):
        super().__init__()

        self.model = model
        
        self.sequence_length = sequence_length
        self.num_sample_steps = num_sample_steps

        self.T = T

        self.beta_tilde_t = [torch.zeros((self.sequence_length)).to(device)]
        # self.beta_tilde_t = [torch.ones((self.sequence_length)).to(device)]
        for cur_t in range(1, self.T+1):
            self.beta_tilde_t.append((self.beta_tilde_t[cur_t-1] + self.beta_t(cur_t) - (self.beta_t(cur_t)*self.beta_tilde_t[cur_t-1])))
        self.beta_tilde_t = torch.stack(self.beta_tilde_t)

    def p_conditional_prob(self, x_t, t):
        '''Returns the probabilities of the Bernoulli variables for the reverse process from t to t-1,
        i.e. p(x_(t-1)|x_t)'''
        return self.model(x_t, t)

    def p_step(self, x, t):
        '''Performs reverse process on x from t to t-1'''
        return torch.bernoulli(self.model(x, t))

    def p_sample(self, batch_size):
        '''Performs complete reverse process on a batch of noise'''
        init_prob = torch.empty((batch_size, self.sequence_length)).fill_(0.5).to(device)
        x = torch.bernoulli(init_prob)

        for cur_t in range(self.T, 0, -1):
            x = torch.bernoulli(self.p_conditional_prob(x, cur_t))
        return x

    def beta_t(self, t):
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

    def q_step(self, x, t):
        '''Performs forward process on x from t to t+1'''
        probs = self.q_conditional_prob(x, t)
        return torch.bernoulli(probs)

    def q_sample(self, x_0, t):
        '''Returns a sample x_t given input x_0'''
        return torch.bernoulli(self.q_conditional_prob_wrt_x_0(x_0, t))

    def kl_div(self, q, p):
        q_clamp = torch.clamp(q, min=0.000001, max=0.999999)
        p_clamp = torch.clamp(p, min=0.000001, max=0.999999)
        return -1.0*torch.sum(q_clamp * torch.log(q_clamp/p_clamp) + (1-q_clamp) * torch.log((1.0-q_clamp)/(1.0-p_clamp)), dim=1)

    def entropy(self, X):
        X_clamp = torch.clamp(X, min=0.000001, max=0.999999)
        return -1.0*torch.sum(X_clamp * torch.log(X_clamp), dim=1)


    def forward(self, x_0):
        '''Approximates the loss via equation 13 in Deep Unsupervised Learning using Nonequilibrium Thermodynamics
        using samples from the reverse process.'''
        # the monte carlo sampling is performed using the minibatch
        total_loss = torch.zeros((x_0.size(dim=0),)).to(device)
        for t in range(1, self.T + 1):
            x_t = self.q_sample(x_0, t)
            beta_t = self.beta_t(t)
            left_term = x_0*(1-self.beta_tilde_t[t-1]) + 0.5*self.beta_tilde_t[t-1]
            left_term *= x_t * (1-0.5*beta_t) + (1 - x_t) * (1.5*beta_t)
            kl_divergence = self.kl_div(left_term,
                                        self.p_conditional_prob(x_t, t))
            q_start = self.q_conditional_prob_wrt_x_0(x_0, 1)
            H_start = torch.sum((-1.0*q_start*torch.log2(q_start)) - ((1.0-q_start)*torch.log2(1.0-q_start)), dim=1)
            q_end = self.q_conditional_prob_wrt_x_0(x_0, self.T)
            H_end = torch.sum((-1.0*q_end*torch.log2(q_end)) - ((1.0-q_end)*torch.log2(1.0-q_end)), dim=1)
            H_prior = (-1.0*0.5*np.log2(0.5)) - ((1.0-0.5)*np.log2(1.0-0.5))
            total_loss += kl_divergence + H_start - H_end + H_prior
        # mult by -1 so we can minimize
        return -1.0 * torch.mean(total_loss)

    def eq_11_forward(self, x):
        '''Approximates loss via equation 11 in Deep Unsupervised Learning using Nonequilibrium Thermodynamics
        using samples from the reverse process.
        In practice, the product term ends up vanishing for each sample,
        so this loss isn't really useful'''
        #avg_loss is the monte carlo approximation of loss
        avg_loss = torch.zeros((x.size(dim=0),)).to(device)
        for n in range(self.num_sample_steps):
            print('n: {}'.format(n))
            cur_loss = torch.zeros((x.size(dim=0),)).fill_(1.0).to(device)
            x_prev = x
            for t in range(0, self.T):
                # sample forward using q
                x_next, q_prob = self.q_sample(x_prev, t)
                print(self.p_prob(x_next, x_prev, t+1))
                cur_loss *= self.p_prob(x_next, x_prev, t+1)/q_prob
                print(cur_loss)
                x_prev = x_next
            avg_loss += cur_loss
        avg_loss = avg_loss/self.num_sample_steps
        # Negative sign so we can minimize loss
        return -1.0 * torch.log(avg_loss)