import unittest
from math import log2

import torch
import numpy as np

from BernoulliDiffusion.utils.math_utils import kl_div, entropy_of_prior, entropy_of_q_conditional

def kl_div_single_bernoulli_distributions(p1, p2):
    p1 = torch.Tensor([p1])
    p2 = torch.Tensor([p2])
    return (p1 * torch.log2(p1/p2)) + ((1-p1) * torch.log2((1-p1)/(1-p2)))

class TestMathUtils(unittest.TestCase):
    '''Tests functions from the math_utils.py file'''

    def test_known_values_of_entropy_of_prior(self):
        '''Entropy of the prior when the prob of each Bernoulli
        distribution is 0.5 should equal sequence_length, i.e.
        one bit of entropy per Bernoulli distribution'''
        H_prior = entropy_of_prior(1)
        self.assertEqual(1.0, H_prior)

        H_prior = entropy_of_prior(2)
        self.assertEqual(2.0, H_prior)

        H_prior = entropy_of_prior(3)
        self.assertEqual(3.0, H_prior)

        H_prior = entropy_of_prior(4)
        self.assertEqual(4.0, H_prior)

        H_prior = entropy_of_prior(32)
        self.assertEqual(32.0, H_prior)

    def test_known_values_of_entropy_of_q_conditional(self):

        # when beta_tilde_t = 1.0, then q=0.5 and this is equivalent to prior
        H = entropy_of_q_conditional(1, 1.0)
        self.assertEqual(1.0, H)

        H = entropy_of_q_conditional(2, 1.0)
        self.assertEqual(2.0, H)

        H = entropy_of_q_conditional(3, 1.0)
        self.assertEqual(3.0, H)

        # this is the single variable case where q=0.25 worked out by hand
        H = entropy_of_q_conditional(1, 0.5)
        real_answer = -1.0*(0.25 * log2(0.25) + 0.75 * log2(0.75))
        self.assertEqual(real_answer, H)

    def test_known_values_of_kl_div(self):

        # kl_div is zero when the two distributions are equal
        div = kl_div(torch.Tensor([[0.5]]), torch.Tensor([[0.5]])).detach().cpu().numpy()
        real_answer = np.array([0.0])
        self.assertTrue(np.array_equal(div, real_answer))

        div = kl_div(torch.Tensor([[0.25, 0.3, 0.75, 0.69]]), torch.Tensor([[0.25, 0.3, 0.75, 0.69]])).detach().cpu().numpy()
        real_answer = np.array([0.0])
        self.assertTrue(np.array_equal(div, real_answer))

        # same as above but with multiple samples
        div = kl_div(torch.Tensor([[0.5], [0.25], [0.75]]), torch.Tensor([[0.5], [0.25], [0.75]])).detach().cpu().numpy()
        real_answer = np.array([0.0, 0.0, 0.0])
        self.assertTrue(np.array_equal(div, real_answer))

        div = kl_div(torch.Tensor([[0.25, 0.3, 0.75, 0.69], [0.25, 0.3, 0.75, 0.40]]), torch.Tensor([[0.25, 0.3, 0.75, 0.69], [0.25, 0.3, 0.75, 0.40]])).detach().cpu().numpy()
        real_answer = np.array([0.0, 0.0])
        self.assertTrue(np.array_equal(div, real_answer))

        # testing an actual calculation with multiple samples where the kl_div is not 0
        div = kl_div(torch.Tensor([[0.4], [0.8]]), torch.Tensor([[0.8], [0.4]]))
        real_answer1 = kl_div_single_bernoulli_distributions(0.4, 0.8)
        real_answer2 = kl_div_single_bernoulli_distributions(0.8, 0.4)
        real_answer = torch.hstack([real_answer1, real_answer2])
        self.assertTrue(torch.equal(div, real_answer))


if __name__ == '__main__':
    unittest.main()
