from numpy import diff
import torch

import yaml
import os
import shutil

from BernoulliDiffusion.config import load_config
from BernoulliDiffusion.data_loader import DataLoader
from BernoulliDiffusion.diffusion_model import BernoulliDiffusionModel
from BernoulliDiffusion.trainer import Trainer
from BernoulliDiffusion.validator import Validator

import sys
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_config(working_dir, output_dir):
    '''Copies the config.yaml and the reverse model to the output directory of the training'''
    shutil.copyfile(os.path.join(working_dir, 'config.yaml'), os.path.join(output_dir, 'config.yaml'))
    shutil.copyfile(os.path.join('BernoulliDiffusion/reverse_model.py'), os.path.join(output_dir, 'reverse_model.py'))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please provide a working directory, for example:\n`python {} runs/multiples_of_5`'.format(sys.argv[0]))
        exit()

    working_dir = sys.argv[1]

    model_cfg, training_cfg, data_cfg = load_config(os.path.join(working_dir, 'config.yaml'))
    
    data_loader = DataLoader(working_dir, data_cfg)

    validator = None
    if os.path.exists(os.path.join(working_dir, 'val.txt')):
        validator = Validator(working_dir, data_cfg)

    sequence_length = data_loader.get_sequence_length()

    diffusion_model = BernoulliDiffusionModel(sequence_length,
                                              model_cfg.num_sample_steps,
                                              model_cfg.T).to(device)

    trainer = Trainer(working_dir, training_cfg, diffusion_model, data_loader, validator)
    save_config(working_dir, trainer.get_output_dir())

    diffusion_model = trainer.train()
    
    print('Batch of 10 samples from reverse trajectory after training:')
    post_sample = diffusion_model.p_sample(10)
    print(post_sample)
    print('Percent of sample digits which are 1: {}'.format(post_sample.sum()/torch.numel(post_sample)))