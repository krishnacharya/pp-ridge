from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math

import sys
sys.path.append('../')
# from src utils
# from src import estimator

from src.utils import generate_linear_data, set_epsilons
from src.estimator import pp_estimator, jorgensen_private_estimator

# Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
N = [10,50,100,200]
D = [10,20,30,40]
runs = 10000
Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 7, 10, 25, 50, 75, 100]

list_of_results = []
i = 0
for d in D:
    # theta = np.random.uniform(0, 1, size=d)
    # theta = normalize(theta.reshape(d, -1), axis=0, norm='l2')
    # theta = theta.reshape((d,))
    for n in N:
        X, y = generate_linear_data(n = n, d = d, sigma = 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
        N_train, N_test = len(X_train), len(X_test)

        # Privacy Levels
        # epsilons = epsilons_34_43_23(N_train)
        epsilons = set_epsilons(N_train, f_c=0.34, f_m=0.43, eps_c=0.01, eps_m=0.2, eps_l=1.0)

        print(f"d: {d}, n: {n}")

        for lamb in Lamb:
            lamb = lamb * d
            pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs)
            jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs)

            di = {"d": d,
                "n": n,
                "lamb": lamb,
                "pp_train_mean": pp_unw_train_mean,
                "pp_train_std": pp_unw_train_std,
                "pp_test_mean": pp_w_test_mean,
                "pp_test_std": pp_w_test_std,
                "jorg_train_mean": jorg_unw_train_mean,
                "jorg_train_std": jorg_unw_train_std,
                "jorg_test_mean": jorg_w_test_mean,
                "jorg_test_std": jorg_w_test_std}
            i += 1
            print(f"Expt {i} done, lambda {lamb}")
            list_of_results.append(di)

df = pd.DataFrame(list_of_results)

df.to_csv('../plevel_34_43_23_result_findlambs.csv', encoding='utf-8', index=False)
