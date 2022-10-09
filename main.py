from numpy import diff
import torch

from BernoulliDiffusion.config import load_config
from BernoulliDiffusion.data import DataLoader
from BernoulliDiffusion.model import BernoulliDiffusionModel
from BernoulliDiffusion.train import Trainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    model_cfg, training_cfg, data_cfg = load_config('config.yaml')
    
    data_loader = DataLoader(data_cfg)
    sequence_length = data_loader.get_sequence_length()

    diffusion_model = BernoulliDiffusionModel(sequence_length,
                                              model_cfg.num_sample_steps,
                                              model_cfg.T).to(device)

    trainer = Trainer(sequence_length, training_cfg, diffusion_model, data_loader)
    diffusion_model = trainer.train()

    print('Batch of 10 samples from reverse trajectory after training:')
    post_sample = diffusion_model.p_sample(10)
    print(post_sample)
    print('Percent of sample digits which are 1: {}'.format(post_sample.sum()/torch.numel(post_sample)))