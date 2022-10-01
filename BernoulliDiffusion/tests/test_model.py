import torch

import unittest
import statistics

from BernoulliDiffusion.data import sample_heartbeat, generate_batch
from BernoulliDiffusion.model import ReverseModel, BernoulliDiffusionModel
from BernoulliDiffusion.config import load_config, Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            
class TestBinomialDiffusion(unittest.TestCase):
    '''Tests functions from the data.py file'''

    def setUp(self):
        '''This is done at the start of every test'''
        self.cfg = load_config('BernoulliDiffusion/tests/test_config.yaml')
        self.reverse_model = ReverseModel(self.cfg.sequence_length, self.cfg.T).to(device)
        self.diffusion_model = BernoulliDiffusionModel(self.reverse_model, self.cfg.sequence_length, self.cfg.num_sample_steps, self.cfg.T).to(device)

    def test_known_values_of_beta_tilde(self):
        '''We expect that beta_tilde_t[T] will always be 1.0
        and beta_tilde_t[1] will be beta_t(1)'''

        for T in [10,1000,2000,5000]:
            self.cfg.T = T
            reverse_model = ReverseModel(self.cfg.sequence_length, self.cfg.T).to(device)
            diffusion_model = BernoulliDiffusionModel(reverse_model, self.cfg.sequence_length, self.cfg.num_sample_steps, self.cfg.T).to(device)
            self.assertAlmostEqual(1.0, diffusion_model.beta_tilde_t[T][0].item(), 5)
            self.assertAlmostEqual(diffusion_model.beta_t(1), diffusion_model.beta_tilde_t[1][0].item(), 5)

    def test_sampling_wrt_x_0(self):
        '''p(1) for a digit of x_0 is 20/100. Bit flip prob for x_t is 0.5*beta_tilde_t.
        p(1->1): 1/5*(1-0.5*beta_tilde_t)
        p(0->1): 4/5*(0.5*beta_tilde_t)
        p(1) for digits of x_t = (20/100*(1-0.5*beta_tilde_t)) + (80/100*(0.5*beta_tilde_t))'''

        target_t = 1000

        beta_tilde_t = self.diffusion_model.beta_tilde_t[target_t][0].item()
        expectation = (20.0/100.0*(1.0-0.5*beta_tilde_t)) + (80.0/100.0*(0.5*beta_tilde_t))
        results = []

        for cur_sample in range(1000):
            x_0 = generate_batch(num_samples=self.cfg.batch_size,
                    period=self.cfg.period,
                    sequence_length=self.cfg.sequence_length).to(device)
            result = self.diffusion_model.q_sample(x_0, target_t)
            result = torch.mean(result).item()
            results.append(result)

        err_msg = 'beta_tilde_t: {}, {} and {} are not almost equal'.format(beta_tilde_t, result, expectation)
        self.assertAlmostEqual(statistics.fmean(results), expectation, 4, err_msg)

            
    def test_sampling_methods_agree(self):
        '''The proportion of digits which are 1 should be approximately equivalent whether we iterate sampling
        from x_0->x_1->...->x_t or sample directly from x_t'''

        target_t = 500

        results1 = []
        results2 = []

        for cur_sample in range(1000):
            x_0 = generate_batch(num_samples=self.cfg.batch_size,
                                period=self.cfg.period,
                                sequence_length=self.cfg.sequence_length).to(device)

            result1 = x_0
            for t in range(0, target_t):
                # print('beta_tilde_{}: {}'.format(t, diffusion_model.beta_tilde_t[t][0].item()))
                # print('beta_t_{}: {}'.format(t, diffusion_model.beta_t(t)))
                result1 = self.diffusion_model.q_step(result1, t)

            result2 = self.diffusion_model.q_sample(x_0, target_t)

            result1 = torch.mean(result1).item()
            result2 = torch.mean(result2).item()
            results1.append(result1)
            results2.append(result2)
        
        err_msg = '{} and {} are not almost equal'.format(statistics.fmean(results1), statistics.fmean(results2))
        self.assertAlmostEqual(statistics.fmean(results1), statistics.fmean(results2), 4, err_msg)
                        

if __name__ == '__main__':
    unittest.main()
