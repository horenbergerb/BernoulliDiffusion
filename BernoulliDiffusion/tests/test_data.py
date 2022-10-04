import torch
import unittest
import statistics

from BernoulliDiffusion.data import sample_heartbeat, generate_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestData(unittest.TestCase):
    '''Tests functions from the data.py file'''

    def test_sample_heartbeat(self):
        '''Deterministic sanity checks'''
        # 10 random samples is a pretty thorough check
        for i in range(10):
            sample = sample_heartbeat(period=5, sequence_length=20)

            self.assertEqual(sample.size(), torch.Size([20]))
            # There should only be 3 or 4 occurences of 1 in any sample
            self.assertLessEqual(sample.sum(), 4)
            self.assertGreaterEqual(sample.sum(),3)

    def test_generate_batch(self):
        '''Deterministic sanity checks'''
        # 10 random samples is a pretty thorough check
        for i in range(10):
            batch = generate_batch(num_samples=10, period=5, sequence_length=20)

            self.assertEqual(batch.size(), torch.Size([10,20]))
            # There should only be 3 or 4 occurences of 1 in any sample
            self.assertLessEqual(batch.sum(), 4*10)
            self.assertGreaterEqual(batch.sum(),3*10)
            
    def test_expected_ratio_of_ones(self):
        '''Stochastic; the probability of 1 should approach 20/100 for large batches
        of period=5, sequence_length=20'''
        batch = generate_batch(num_samples=100000, period=5, sequence_length=20)
        self.assertAlmostEqual(torch.mean(batch).item(), 20.0/100.0)