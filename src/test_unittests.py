import torch

import unittest

from data import sample_heartbeat, generate_batch


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
        pass


if __name__ == '__main__':
    unittest.main()