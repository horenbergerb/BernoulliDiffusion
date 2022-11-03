import torch
import torch.optim as optim
from torch.distributions import Binomial

from datetime import datetime
import os
import json

from BernoulliDiffusion.utils.plotting_utils import plot_loss, plot_evolution, plot_validation_proportions

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer:
    '''Given a diffusion model and config file, this class
    trains the model from scratch and then saves the result.'''
    def __init__(self, working_dir, cfg, diffusion_model, data_loader, validator=None):
        self.working_dir = working_dir
        self.cfg = cfg
        self.diffusion_model = diffusion_model
        self.data_loader = data_loader
        self.validator = validator

        self.optimizer = optim.AdamW(diffusion_model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.result_dir = None

        self.initialize_output_dir_()
        self.initialize_new_training_()

        self.example_seed = self.data_loader.generate_random_data(self.cfg.num_examples)


    def initialize_output_dir_(self):
        '''Creates the file where results of training will be stored'''
        now = datetime.now().strftime('d_%m_%d_%Y_t_%H_%M')
        all_results_dir = os.path.join(self.working_dir, 'results')
        self.result_dir = os.path.join(all_results_dir, now)
        if not os.path.isdir(all_results_dir):
            os.mkdir(all_results_dir)
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not os.path.isdir(os.path.join(self.result_dir, 'checkpoints')):
            os.mkdir(os.path.join(self.result_dir, 'checkpoints'))

    def get_output_dir(self):
        return self.result_dir

    def initialize_new_training_(self):
        # Zero grad, input a random tensor, then backprop
        self.diffusion_model.zero_grad()
        input = self.data_loader.generate_random_data()
        out = self.diffusion_model(input)
        out.backward()

        # reset all metrics of training
        self.losses = []
        self.examples_per_epoch = []

        self.proportions = {'train': [], 'val':[], 'other':[]}

    def train_step_(self):
        '''Actual training occurs here'''
        avg_loss = 0.0
        batch_count = 0
        batch = self.data_loader.next_minibatch()
        while batch is not None:
            self.optimizer.zero_grad()
            output = self.diffusion_model(batch)
            if torch.isnan(output):
                raise Exception('Loss is NaN. Is your learning rate too high?')
            output.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.cfg.clip_thresh)
            self.optimizer.step()
            batch = self.data_loader.next_minibatch()
            batch_count += 1
            avg_loss += output.detach()
        avg_loss = avg_loss / batch_count
        self.losses.append(avg_loss.item())
        return avg_loss

    def log_data_(self):
        '''Calculate and store various values of interest. Done after each epoch.'''
        post_sample = self.diffusion_model.p_sample(self.cfg.num_examples, self.example_seed)
        self.examples_per_epoch.append(post_sample)
        if self.validator is not None:
            train_count, val_count, other_count = self.validator.validate(self.diffusion_model,
                                                                          self.cfg.num_val_samples,
                                                                          self.cfg.val_batch_size)
            self.proportions['train'].append(train_count/self.cfg.num_val_samples)
            self.proportions['val'].append(val_count/self.cfg.num_val_samples)
            self.proportions['other'].append(other_count/self.cfg.num_val_samples)

    def print_training_info_(self):
        print('Epoch: {}: Loss: {} Train Proportion: {} Val Proportion: {}'.format(self.epoch,
                                                                                   self.losses[-1],
                                                                                   self.proportions['train'][-1],
                                                                                   self.proportions['val'][-1]))

    def dump_data_to_json_(self):
        results = {'losses': self.losses,
                   'examples_per_epoch': [x.tolist() for x in self.examples_per_epoch],
                   'proportions': self.proportions}
        with open(os.path.join(self.result_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp)

    def make_plots_(self):
        epochs_plotted = [x for x in range(0, self.cfg.epochs)]
        plot_loss(epochs_plotted,
                  self.losses,
                  self.result_dir)
        numpy_examples = [x.cpu().detach().numpy() for x in self.examples_per_epoch]
        plot_evolution(epochs_plotted,
                       numpy_examples,
                       self.result_dir,
                       step_name = 'Epoch',
                       filename='sample_evolution_throughout_training.gif')
        plot_validation_proportions(epochs_plotted,
                                    self.proportions,
                                    self.result_dir)

    def train(self):
        for self.epoch in range(0, self.cfg.epochs):
            self.train_step_()
            self.log_data_()
            self.print_training_info_()
            if self.epoch%self.cfg.save_every_n_epochs == 0 and self.epoch > 0:
                torch.save(self.diffusion_model, os.path.join(self.result_dir, 'checkpoints', 'model_epoch_{}.pt'.format(self.epoch)))
        
        torch.save(self.diffusion_model, os.path.join(self.result_dir, 'checkpoints', 'model_epoch_{}.pt'.format(self.epoch)))
        self.dump_data_to_json_()
        self.make_plots_()

        return self.diffusion_model