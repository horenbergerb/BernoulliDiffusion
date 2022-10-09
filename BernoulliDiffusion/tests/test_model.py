import torch

import unittest
import statistics

from BernoulliDiffusion.data import DataLoader
from BernoulliDiffusion.model import ReverseModel, BernoulliDiffusionModel
from BernoulliDiffusion.config import load_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            
class TestBinomialDiffusion(unittest.TestCase):
    '''Tests functions from the data.py file'''

    def setUp(self):
        '''This is done at the start of every test'''
        self.model_cfg, self.training_cfg, self.data_cfg = load_config('BernoulliDiffusion/tests/test_config.yaml')
        self.data_loader = DataLoader(self.data_cfg)
        self.sequence_length = self.data_loader.get_sequence_length()

        self.diffusion_model = BernoulliDiffusionModel(self.sequence_length,
                                                       self.model_cfg.num_sample_steps,
                                                       self.model_cfg.T).to(device)

    def test_known_values_of_beta_tilde(self):
        '''We expect that beta_tilde_t[T] will always be 1.0
        and beta_tilde_t[1] will be beta_t(1)'''

        for T in [10,1000,2000,5000]:
            diffusion_model = BernoulliDiffusionModel(self.sequence_length,
                                                      self.model_cfg.num_sample_steps,
                                                      T).to(device)
            self.assertAlmostEqual(1.0, diffusion_model.beta_tilde_t[T][0].item(), 5)
            self.assertAlmostEqual(diffusion_model.beta_t(1), diffusion_model.beta_tilde_t[1][0].item(), 5)

    def test_sampling_wrt_x_0(self):
        '''p(1) for a digit of x_0 is 1/period (when period divides sequence_length). Bit flip prob for x_t is 0.5*beta_tilde_t.
        p(1->1): (1/period)*(1-0.5*beta_tilde_t)
        p(0->1): (1 - 1/period)*(0.5*beta_tilde_t)
        p(1) for digits of x_t = ((1/period)*(1-0.5*beta_tilde_t)) + ((1 - 1/period)(0.5*beta_tilde_t))'''

        target_t = 1000
        period = 100
        prob_1 = 1/period

        beta_tilde_t = self.diffusion_model.beta_tilde_t[target_t][0].item()
        expectation = (prob_1*(1.0-0.5*beta_tilde_t)) + ((1-prob_1)*(0.5*beta_tilde_t))
        results = []

        for cur_sample in range(10000):
            x_0 = self.data_loader.next_minibatch()
            while x_0 is not None:
                result = self.diffusion_model.q_sample(x_0, target_t)
                result = torch.mean(result).item()
                results.append(result)
                x_0 = self.data_loader.next_minibatch()


        err_msg = 'beta_tilde_t: {}, {} and {} are not almost equal'.format(beta_tilde_t, result, expectation)
        self.assertAlmostEqual(statistics.fmean(results), expectation, 4, err_msg)

            
    def test_sampling_methods_agree(self):
        '''The proportion of digits which are 1 should be approximately equivalent whether we iterate sampling
        from x_0->x_1->...->x_t or sample directly from x_t'''

        target_t = 500

        results1 = []
        results2 = []

        for cur_sample in range(1000):
            x_0 = self.data_loader.next_minibatch()
            while x_0 is not None:
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
                x_0 = self.data_loader.next_minibatch()

        
        err_msg = '{} and {} are not almost equal'.format(statistics.fmean(results1), statistics.fmean(results2))
        self.assertAlmostEqual(statistics.fmean(results1), statistics.fmean(results2), 3, err_msg)
                        

if __name__ == '__main__':
    unittest.main()
