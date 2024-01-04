from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math

import os, sys, argparse
sys.path.append('../')
# from src utils
# from src import estimator

from src.utils import generate_linear_data, set_epsilons
from src.estimator import pp_estimator, jorgensen_private_estimator

# Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]

def run(N, D, lambds, runs=10000):

    list_of_results = []
    check_pp_test_vals, check_jorg_test_vals = [], []
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
            epsilons = set_epsilons(N_train, f_c=0.54, f_m=0.37, eps_c=0.01, eps_m=0.2, eps_l=1.0)

            Lamb = lambds
            print(f"d: {d}, n: {n}")
    
            c, p = 0, 0
            for lamb in Lamb:
                # if lamb >= 1000:
                #     break 
                
                # lamb = lamb * d

                
                pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
                jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
                
                # check_jorg_test_vals.append(jorg_w_test_mean)

                # check_pp_test_vals.append(pp_w_test_mean)
                # if len(check_pp_test_vals) >= 2:
                #     if check_pp_test_vals[c-1] <= check_pp_test_vals[c]:
                #         p += 1
                #         if Lamb[c]*2 not in Lamb:
                #             Lamb.insert(c+1, Lamb[c]*2)
                

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
                
                print(f"Expt {i} done, lambda {lamb}")
                list_of_results.append(di)

                # if (lamb >= 1e10) or (abs(check_pp_test_vals[c]-check_jorg_test_vals[c])<=0.01):
                #     break
                i += 1
                c += 1

            

    df = pd.DataFrame(list_of_results)

    if not os.path.exists("../result"):
        os.mkdir("../result")

    df.to_csv(f'../specific_{d}_{n}_plevel_34_43_23_result.csv', encoding='utf-8', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("d", default=10, type=int)
    parser.add_argument("n", default=100, type=int)
    args = parser.parse_args()

    N = [args.n]
    D = [args.d]

    lambds = [10, 25, 50, 100, 200, 300, 400, 500]
    runs = 10000

    run(N, D, lambds, runs)