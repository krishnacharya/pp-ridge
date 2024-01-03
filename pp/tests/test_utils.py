import sys
sys.path.append('../')
import unittest
import numpy as np
from src.utils import *


class TestUtils(unittest.TestCase):
    def test_generate_linear_data(self):
        n, d = 10, 3 # 10 datapoints in 3 dimensions
        X, y = generate_linear_data(n=n, d=d, sigma=0)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape, (n,d))
        self.assertEqual(y.shape, (n,))
        self.assertTrue(0 <= np.min(X) and np.max(X) <= 1) # we are given that each feature is in [0,1]
        self.assertTrue(-1 <= np.min(y) and np.max(y) <= 1) # each y label is in [-1, 1]

    def test_weighted_rls_solution(self):
        n, d = 100, 10
        X = np.random.rand(n, d) # shape (n, d)
        theta_true = np.random.rand(d) # shape (d,)
        y = X @ theta_true # shape (n,)
        thetabar = weighted_rls_solution(weights = np.ones(n)/n, X=X, y=y, lamb = 0)
        self.assertTrue(thetabar.shape, (d, 1))
        np.testing.assert_allclose(thetabar.ravel(), theta_true)
    
    def test_evaluate_weighted_rls_objective(self):
        X = np.array([[2,1], [3,2]])
        y = np.array([3, 5]) # theta that generated this is X @ (1,1)

        theta_1 = np.array([1,1])
        ls_on_theta1 = evaluate_weighted_rls_objective(theta = theta_1, weights = np.ones(2)/2, X=X, y=y, lamb=0)
        np.testing.assert_allclose(ls_on_theta1, 0.0)

        theta_2 = np.array([0, 0])
        ls_on_theta2 = evaluate_weighted_rls_objective(theta = theta_2, weights = np.ones(2)/2, X=X, y=y, lamb=0)
        np.testing.assert_allclose(ls_on_theta2, ((3**2 + 5**2)/2)**0.5)
    
    def test_set_epsilons(self):
        N_train = 100
        epsilons = set_epsilons(N_train, f_c=0.34, f_m=0.43, eps_c=0.01, eps_m=0.2, eps_l=1.0)
        self.assertEqual(epsilons.shape, (N_train,))
        np.testing.assert_allclose(np.sum((epsilons > 0.01) & (epsilons <= 0.2)) / N_train, 0.34, atol = 0.05)
        np.testing.assert_allclose(np.sum((epsilons > 0.2) & (epsilons < 1)) / N_train, 0.43, atol = 0.05)
        np.testing.assert_allclose(np.sum((epsilons >=1) / N_train), 0.23, atol = 0.05)

if __name__ == '__main__':
    unittest.main()