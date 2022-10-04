import torch
import torch.optim as optim
from torch.distributions import Binomial

import os
import json

from BernoulliDiffusion.config import Config
from BernoulliDiffusion.model import BernoulliDiffusionModel
from BernoulliDiffusion.data import generate_batch
from BernoulliDiffusion.utils.plotting_utils import plot_loss, plot_evolution

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer:
    '''Given a diffusion model and config file, this class
    trains the model from scratch and then saves the result.'''
    def __init__(self, cfg: Config, diffusion_model, num_examples = 10):
        self.cfg = cfg
        self.diffusion_model = diffusion_model
        self.num_examples = num_examples

        self.optimizer = optim.Adam(diffusion_model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.losses = []

        self.examples_per_epoch = []
        self.example_seed = Binomial(1, torch.zeros((self.num_examples, self.cfg.sequence_length)).fill_(0.5)).sample().to(device)

    def initialize_output_dir_(self):
        if not os.path.isdir(self.cfg.output_dir):
            os.mkdir(self.cfg.output_dir)
        if not os.path.isdir(os.path.join(self.cfg.output_dir, 'models')):
            os.mkdir(os.path.join(self.cfg.output_dir, 'models'))

    def initialize_new_training_(self):
        # Zero grad, input a random tensor, then backprop
        self.diffusion_model.zero_grad()
        m = Binomial(1, torch.zeros((self.cfg.batch_size, self.cfg.sequence_length)).fill_(0.5))
        input = m.sample().to(device)
        out = self.diffusion_model(input)
        self.diffusion_model.zero_grad()
        out.sum().backward()

    def train_step_(self):
        avg_loss = 0.0
        for batch in self.batches:
            self.optimizer.zero_grad()
            output = self.diffusion_model(batch)
            output.backward()
            self.optimizer.step()
            avg_loss += output.sum()
        avg_loss = avg_loss / (self.cfg.batch_size * self.cfg.num_batches)
        self.losses.append(avg_loss.item())
        return avg_loss

    def print_training_info_(self):
        print('Epoch: {}'.format(self.epoch))
        print('Losses per epoch: {}'.format(self.losses))
        print('Sample from reverse trajectory:')
        post_sample = self.diffusion_model.p_sample(self.num_examples, self.example_seed)
        self.examples_per_epoch.append(post_sample)
        print(post_sample)
        print('Percent of sample digits which are 1: {}'.format(post_sample.sum()/torch.numel(post_sample)))

    def dump_data_to_json(self):
        results = {'losses': self.losses,
                   'examples_per_epoch': [x.tolist() for x in self.examples_per_epoch]}
        with open(os.path.join(self.cfg.output_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp)

    def make_plots(self):
        plot_loss([x for x in range(0, self.cfg.epochs)],
                  self.losses,
                  self.cfg.output_dir)
        numpy_examples = [x.cpu().detach().numpy() for x in self.examples_per_epoch]
        plot_evolution(numpy_examples,
                       self.cfg.output_dir,
                       step_name = 'Epoch',
                       filename='sample_evolution_throughout_training.gif')

    def train(self):
        self.initialize_output_dir_()
        self.initialize_new_training_()

        # generate the training data, i.e. batches of heartbeat data
        self.batches = [generate_batch(num_samples=self.cfg.batch_size,
                                       period=self.cfg.period,
                                       sequence_length=self.cfg.sequence_length).to(device=device)
                                       for i in range(0, self.cfg.num_batches)]

        for self.epoch in range(0, self.cfg.epochs):
            self.train_step_()
            if self.epoch%self.cfg.training_info_freq == 0:
                self.print_training_info_()
            if self.epoch%self.cfg.save_every_n_epochs == 0:
                torch.save(self.diffusion_model, os.path.join(self.cfg.output_dir, 'models', 'model_epoch_{}.pt'.format(self.epoch)))
        
        torch.save(self.diffusion_model, os.path.join(self.cfg.output_dir, 'models', 'model_epoch_{}.pt'.format(self.epoch)))
        self.dump_data_to_json()
        self.make_plots()

        return self.diffusion_model