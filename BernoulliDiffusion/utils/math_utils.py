import torch
from math import comb, pow, log2

eps = 1e-30

def kl_div(q, p):
    '''KL Divergence of two multivariate Bernoulli distributions'''
    q = torch.clip(q, min=1e-10, max=1-(1e-7))
    p = torch.clip(p, min=1e-10, max=1-(1e-7))
    return torch.sum((q * torch.log2((q/p) + eps)) + ((1.0-q) * torch.log2(((1.0-q)/(1.0-p)) + eps)), dim=1)


def entropy_of_q_conditional(sequence_length, beta_tilde_t):
    total_entropy = 0.0
    for k in range(0, sequence_length+1):
        n_choose_k = comb(sequence_length, k)
        prob = pow((1.0-(0.5*beta_tilde_t)), k) * pow(0.5*beta_tilde_t, sequence_length-k)
        cur_entropy = n_choose_k * prob * log2(prob + eps)
        total_entropy += cur_entropy
    return -1.0 * total_entropy


def entropy_of_prior(sequence_length):
    '''Assuming all Bernoulli distributions in prior have prob 0.5.
    Fun fact: this basically just returns float(sequence_length)'''
    total_entropy = 0.0
    for k in range(0, sequence_length+1):
        n_choose_k = comb(sequence_length, k)
        prob = pow((1.0-0.5), k) * pow(0.5, sequence_length-k)
        cur_entropy = n_choose_k * prob * log2(prob + eps)
        total_entropy += cur_entropy
    return -1.0 * total_entropy
