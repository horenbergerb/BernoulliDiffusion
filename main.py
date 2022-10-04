import torch

from BernoulliDiffusion.config import load_config
from BernoulliDiffusion.model import BernoulliDiffusionModel, ReverseModel
from BernoulliDiffusion.train import train_diffusion_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    cfg = load_config('config.yaml')

    reverse_model = ReverseModel(cfg.sequence_length, cfg.T).to(device)
    diffusion_model = BernoulliDiffusionModel(reverse_model, cfg.sequence_length, cfg.num_sample_steps, cfg.T).to(device)

    print('Batch of samples from reverse trajectory before training:')
    pre_sample = diffusion_model.p_sample(cfg.batch_size)
    print(pre_sample)
    print('Percent of sample digits which are 1: {}'.format(pre_sample.sum()/torch.numel(pre_sample)))

    diffusion_model = train_diffusion_model(cfg, diffusion_model)

    print('Batch of samples from reverse trajectory after training:')
    post_sample = diffusion_model.p_sample(cfg.batch_size)
    print(post_sample)
    print('Percent of sample digits which are 1: {}'.format(post_sample.sum()/torch.numel(post_sample)))