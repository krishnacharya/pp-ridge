from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import sys
sys.path.append('../')
# from src utils
# from src import estimator

from src.utils import generate_linear_data
from src.estimator import pp_estimator, jorgensen_private_estimator

Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
N = [10,50,100,200]
D = [10,20,30,40]
runs = 10000

# N = 100 # synthetic dataset size
# d = 10 # dimensionality

list_of_results = []

for n in N:
    for d in D:

        theta = np.random.uniform(0, 1, size=d)
        theta = normalize(theta.reshape(d, -1), axis=0, norm='l2')

        theta = theta.reshape((d,))

        X, y = generate_linear_data(n = N, theta = theta, sigma=0)
        assert np.linalg.norm(y) <= 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        N_train, N_test = len(X_train), len(X_test)

        # Privacy Level Distributions

        epsilons = np.zeros(N_train)# training datas agents privacy levels, want 3 privacy levels

        # values chosen from this paper : "Utility-aware Exponential Mechanism for Personalized Differential Privacy"
        epsilons[:int(0.54*N_train)] = 0.1 # 54% care abt privacy (FUNDAMENTALISTS)
        epsilons[int(0.54*N_train) : int(0.91 * N_train)] = 0.5 # 37% care little abt privacy (PRAGMATISTS)
        epsilons[int(0.91 * N_train) : ] = 1.0 # 9% dont care (UNCONCERNED)
        # from collections import Counter
        # print(Counter(epsilons))

        for lamb in Lamb:
            
            pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, N_train, N_test)
            jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, N_train, N_test)

            di = {"n": n,
                "d": d,
                "lamb": lamb,
                "pp_unw_train_mean": pp_unw_train_mean,
                "pp_unw_train_std": pp_unw_train_std,
                "pp_w_test_mean": pp_w_test_mean,
                "pp_w_test_std": pp_w_test_std,
                "jorg_unw_train_mean": jorg_unw_train_mean,
                "jorg_unw_train_std": jorg_unw_train_std,
                "jorg_w_test_mean": jorg_w_test_mean,
                "jorg_w_test_std": jorg_w_test_std}
            
            list_of_results.append(di)

df = pd.DataFrame(list_of_results)

df.to_csv('../plevel_54_37_9_result.csv', encoding='utf-8', index=False)


