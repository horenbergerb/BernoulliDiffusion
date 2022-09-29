import torch

import unittest

from src.data import sample_heartbeat, generate_batch
from src.model import ReverseModel, BernoulliDiffusion
from src.config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestData(unittest.TestCase):
    '''Tests functions from the data.py file'''

    def test_sample_heartbeat(self):
        # 10 random samples is a pretty thorough check
        for i in range(10):
            sample = sample_heartbeat(period=5, sequence_length=20)

            self.assertEqual(sample.size(), torch.Size([20]))
            # There should only be 3 or 4 occurences of 1 in any sample
            self.assertLessEqual(sample.sum(), 4)
            self.assertGreaterEqual(sample.sum(),3)

        # todo: check that the data is actually random heartbeats
        # i gave it an ocular patdown

    def test_generate_batch(self):
        # 10 random samples is a pretty thorough check
        for i in range(10):
            batch = generate_batch(num_samples=10, period=5, sequence_length=20)

            self.assertEqual(batch.size(), torch.Size([10,20]))
            # There should only be 3 or 4 occurences of 1 in any sample
            self.assertLessEqual(batch.sum(), 4*10)
            self.assertGreaterEqual(batch.sum(),3*10)


class TestBinomialDiffusion(unittest.TestCase):
    '''Tests functions from the data.py file'''

    def setUp(self):
        '''This is done at the start of every test'''
        self.cfg = Config(sequence_length=20,
                          period=5,
                          T=2000,
                          batch_size=100000,
                          num_batches=1,
                          num_sample_steps=10,
                          epochs=10,
                          lr=0.01,
                          training_info_freq=1,
                          filename='unittest_model.pt')
        self.reverse_model = ReverseModel(cfg.sequence_length, cfg.T).to(device)
        self.diffusion_model = BernoulliDiffusion(reverse_model, cfg.sequence_length, cfg.num_sample_steps, cfg.T).to(device)

    def test_beta_tilde_T_is_correct(self):
        '''We expect that beta_tilde_T will always be 0.5'''

        for T in [10,1000,2000,5000]:
            self.cfg.T = T
            reverse_model = ReverseModel(self.cfg.sequence_length, self.cfg.T).to(device)
            diffusion_model = BernoulliDiffusion(reverse_model, self.cfg.sequence_length, self.cfg.num_sample_steps, self.cfg.T).to(device)
            print('T: {} beta_tilde_T: {}'.format(T, diffusion_model.beta_tilde_t[t][0].item()))
            self.assertEqual(0.5, diffusion_model.beta_tilde_t[t][0].item())
    
    def test_sampling_methods_agree(self):
        x_0 = generate_batch(num_samples=self.cfg.batch_size,
                               period=self.cfg.period,
                               sequence_length=self.cfg.sequence_length).to(device)

        target_t = 500
        result1 = x_0
        for t in range(0, target_t):
            # print('beta_tilde_{}: {}'.format(t, diffusion_model.beta_tilde_t[t][0].item()))
            # print('beta_t_{}: {}'.format(t, diffusion_model.beta_t(t)))
            result1 = self.diffusion_model.q_step(result1, t)

        result2 = self.diffusion_model.q_sample(x_0, target_t)

        result1 = torch.mean(result1)
        result2 = torch.mean(result2)
        
        err_msg = '{} and {} are not almost equal'.format(result1, result2)

        self.assertAlmostEqual(result1, result2, 5, err_msg)
                
        # print(diffusion_model.beta_tilde_t[target_t][0])
        

if __name__ == '__main__':
    unittest.main()
