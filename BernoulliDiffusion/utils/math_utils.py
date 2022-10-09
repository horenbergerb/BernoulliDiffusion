import torch
from math import comb, pow, log10


def kl_div(q, p):
    '''KL Divergence of two multivariate Bernoulli distributions'''
    q_clamp = torch.clamp(q, min=0.000001, max=0.999999)
    p_clamp = torch.clamp(p, min=0.000001, max=0.999999)
    return -1.0*torch.sum(q_clamp * torch.log(q_clamp/p_clamp) + (1-q_clamp) * torch.log((1.0-q_clamp)/(1.0-p_clamp)), dim=1)


def entropy_of_q_conditional(sequence_length, beta_tilde_t):
    total_entropy = 0.0
    for k in range(0, sequence_length+1):
        n_choose_k = comb(sequence_length, k)
        prob = pow((1-0.5*beta_tilde_t), k) * pow(0.5*beta_tilde_t, sequence_length-k)
        prob = max(prob, 0.000001)
        cur_entropy = n_choose_k * prob * log10(prob)
        total_entropy += cur_entropy
    return -1.0 * total_entropy

def entropy_of_prior(sequence_length):
    total_entropy = 0.0
    for k in range(0, sequence_length+1):
        n_choose_k = comb(sequence_length, k)
        prob = pow((1-0.5*0.5), k) * pow(0.5*0.5, sequence_length-k)
        prob = max(prob, 0.000001)
        cur_entropy = n_choose_k * prob * log10(prob)
        total_entropy += cur_entropy
    return -1.0 * total_entropy
