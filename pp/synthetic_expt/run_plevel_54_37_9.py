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

N = [10,50,100,200]
D = [10,20,30,40]
runs = 10000

list_of_results = []
check_test_vals = []
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
        # epsilons = epsilons_54_37_9(N_train)
        epsilons = set_epsilons(N_train, f_c=0.54, f_m=0.37, eps_c=0.01, eps_m=0.2, eps_l=1.0)

        Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 7, 10, 25, 50, 75, 100]
        print(f"d: {d}, n: {n}")

        c, p = 0, 0
        for lamb in Lamb:
            # pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, N_train, N_test, runs)
            # jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, N_train, N_test, runs)
            # if lamb >= 1000:
            #     break
            if p >= 3:
                break
            lamb = lamb * d
            pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs)
            jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs)
            
            check_test_vals.append(pp_w_test_mean)
            if len(check_test_vals) >= 2:
                if check_test_vals[c-1] <= check_test_vals[c]:
                    p += 1
                    Lamb.insert(c+1, Lamb[c]*2)
            
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
            c += 1
            print(f"Expt {i} done, lambda {lamb}")
            list_of_results.append(di)
        list_of_results.sort()

df = pd.DataFrame(list_of_results)

df.to_csv('../plevel_54_37_9_result_findlambs.csv', encoding='utf-8', index=False)