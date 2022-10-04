import torch


def kl_div(q, p):
    '''KL Divergence of two multivariate Bernoulli distributions'''
    q_clamp = torch.clamp(q, min=0.000001, max=0.999999)
    p_clamp = torch.clamp(p, min=0.000001, max=0.999999)
    return -1.0*torch.sum(q_clamp * torch.log(q_clamp/p_clamp) + (1-q_clamp) * torch.log((1.0-q_clamp)/(1.0-p_clamp)), dim=1)


def entropy(q):
    '''Calculate the entropy of a multivariate Bernoulli distribution
    TODO: This calculation is WRONG right now. Working on an efficient solution'''

    return torch.sum((-1.0*q*torch.log2(q)) - ((1.0-q)*torch.log2(1.0-q)), dim=1)