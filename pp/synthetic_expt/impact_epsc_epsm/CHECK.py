from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math

import os, sys, argparse
sys.path.append('../../../')
# from src utils
# from src import estimator

from src.utils import generate_linear_data, set_epsilons
from src.estimator import pp_estimator, jorgensen_private_estimator, nonpriv_solution

# Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]

def run(N, D, lambds, epsc, epsm, runs=10000):

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
            for eps_c in epsc:
                for eps_m in epsm:

                    if (eps_m == 0.5):
                        # (eps_c == 0.01) or :
                        epsilons = set_epsilons(N_train, f_c=0.54, f_m=0.37, eps_c=eps_c, eps_m=eps_m, eps_l=1.0)

                        jorg_thresh_max, jorg_thresh_mean = max(epsilons), np.mean(epsilons)

                        Lamb = lambds
                        print(f"d: {d}, n: {n}")

                        for lamb in Lamb:
                            # if lamb >= 1000:
                            #     break 
                            
                            # lamb = lamb * d

                            # just for sanity check 
                            _, _, unreg_pp_baseline_test_mean, unreg_pp_baseline_test_std, _ = pp_estimator(epsilons, X_train, y_train, X_test, y_test, 0, runs, eval_lamb=0, non_personalized=True)
                            # just for sanity check 
                            jorg_thresh = min(epsilons)
                            _, _, unreg_jorg_baseline_test_mean, unreg_jorg_baseline_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh, X_train, y_train, X_test, y_test, 0, runs, eval_lamb=0)
                            _, _, jorg_baseline_test_mean, jorg_baseline_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)



                            # 4.1 Type1
                            unreg_pp_train_mean, unreg_pp_train_std, unreg_pp_test_mean, unreg_pp_test_std, unreg_theta_hat_pp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
                            _, _, unreg_nonpp_test_mean, unreg_nonpp_test_std, unreg_theta_hat_nonpp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0, non_personalized=True)
                            
                            # 4.1 Type2
                            reg_pp_train_mean, reg_pp_train_std, reg_pp_test_mean, reg_pp_test_std, reg_theta_hat_pp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb)
                            _, _, reg_nonpp_test_mean, reg_nonpp_test_std, reg_theta_hat_nonpp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb, non_personalized=True)

                            # 4.3 Type1
                            unreg_jorg_max_train_mean, unreg_jorg_max_train_std, unreg_jorg_max_test_mean, unreg_jorg_max_test_std, unreg_theta_hat_jorg_max_norm = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
                            unreg_jorg_avg_train_mean, unreg_jorg_avg_train_std, unreg_jorg_avg_test_mean, unreg_jorg_avg_test_std, unreg_theta_hat_jorg_avg_norm = jorgensen_private_estimator(epsilons, jorg_thresh_mean, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
                            type1_nonpriv_loss = nonpriv_solution(N_train, N_test, X_train, y_train, X_test, y_test, lamb=0, eval_lamb=0)

                            # 4.3 Type2
                            reg_jorg_max_train_mean, reg_jorg_max_train_std, reg_jorg_max_test_mean, reg_jorg_max_test_std, reg_theta_hat_jorg_max_norm = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb)
                            reg_jorg_avg_train_mean, reg_jorg_avg_train_std, reg_jorg_avg_test_mean, reg_jorg_avg_test_std, reg_theta_hat_jorg_avg_norm = jorgensen_private_estimator(epsilons, jorg_thresh_mean, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb)
                            type2_nonpriv_loss = nonpriv_solution(N_train, N_test, X_train, y_train, X_test, y_test, lamb, eval_lamb=lamb)
                            
                            
                            di = {"d": d,
                                "n": n,
                                "theta_"
                                "eps_c": eps_c, 
                                "eps_m": eps_m,
                                "lamb": lamb,
                                "unreg_pp_test_mean": unreg_pp_test_mean,
                                "unreg_nonpp_test_mean": unreg_nonpp_test_mean,
                                "reg_pp_test_mean": reg_pp_test_mean,
                                "reg_nonpp_test_mean": reg_nonpp_test_mean,
                                "unreg_jorg_max_test_mean": unreg_jorg_max_test_mean,
                                "unreg_jorg_avg_test_mean": unreg_jorg_avg_test_mean,
                                "type1_nonpriv_loss": type1_nonpriv_loss,
                                "reg_jorg_max_test_mean": reg_jorg_max_test_mean,
                                "reg_jorg_avg_test_mean": reg_jorg_avg_test_mean,
                                "type2_nonpriv_loss": type2_nonpriv_loss,
                                "unreg_pp_test_std": unreg_pp_test_std,
                                "unreg_nonpp_test_std": unreg_nonpp_test_std,
                                "unreg_jorg_max_test_std": unreg_jorg_max_test_std,
                                "unreg_jorg_avg_test_std": unreg_jorg_avg_test_std,
                                "reg_pp_test_std": reg_pp_test_std,
                                "reg_nonpp_test_std": reg_nonpp_test_std,
                                "reg_jorg_max_test_std": reg_jorg_max_test_std,
                                "reg_jorg_avg_test_std": reg_jorg_avg_test_std,
                                # "pp_baseline_test_mean": pp_baseline_test_mean,
                                # "pp_train_mean": pp_train_mean,
                                # "pp_test_mean": pp_test_mean,
                                # "jorg_max_train_mean": jorg_max_train_mean,
                                # "jorg_max_test_mean": jorg_max_test_mean,
                                # "jorg_avg_train_mean": jorg_avg_train_mean,
                                # "jorg_avg_test_mean": jorg_avg_test_mean,
                                # "pp_train_std": pp_train_std,
                                # "pp_test_std": pp_test_std,
                                # "jorg_max_train_std": jorg_max_train_std,
                                # "jorg_max_test_std": jorg_max_test_std,
                                # "jorg_avg_train_std": jorg_avg_train_std,
                                # "jorg_avg_test_std": jorg_avg_test_std,
                                # "pp_baseline_test_std": pp_baseline_test_std,
                                "unreg_pp_baseline_test_mean": unreg_pp_baseline_test_mean,
                                "unreg_jorg_baseline_test_mean": unreg_jorg_baseline_test_mean,
                                "jorg_baseline_test_mean": jorg_baseline_test_mean,
                                "unreg_pp_baseline_test_std": unreg_pp_baseline_test_std,
                                "unreg_jorg_baseline_test_std": unreg_jorg_baseline_test_std,
                                "jorg_baseline_test_std": jorg_baseline_test_std,
                                #######
                                "unreg_theta_hat_pp_norm": unreg_theta_hat_pp_norm,
                                "unreg_theta_hat_nonpp_norm": unreg_theta_hat_nonpp_norm,
                                "reg_theta_hat_pp_norm": reg_theta_hat_pp_norm,
                                "reg_theta_hat_nonpp_norm": reg_theta_hat_nonpp_norm,
                                "unreg_theta_hat_jorg_max_norm": unreg_theta_hat_jorg_max_norm,
                                "unreg_theta_hat_jorg_avg_norm": unreg_theta_hat_jorg_avg_norm,
                                "reg_theta_hat_jorg_max_norm": reg_theta_hat_jorg_max_norm,
                                "reg_theta_hat_jorg_avg_norm": reg_theta_hat_jorg_avg_norm
                                }
                            
                            print(f"Expt {i} done, lambda {lamb}, eps_c {eps_c}, eps_m {eps_m}")
                            list_of_results.append(di)

                            # if (lamb >= 1e10) or (abs(check_pp_test_vals[c]-check_jorg_test_vals[c])<=0.01):
                            #     break
                            i += 1

            

    df = pd.DataFrame(list_of_results)

    if not os.path.exists("../csv_outputs"):
        os.mkdir("../csv_outputs")

    df.to_csv(f'../csv_outputs/forplots_specific_{d}_{n}_eps_c_eps_m_CHECK.csv', encoding='utf-8', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("d", default=10, type=int)
    parser.add_argument("n", default=100, type=int)
    args = parser.parse_args()

    N = [args.n]
    D = [args.d]
    
    # lambds = [0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000]
    lambds = [50]
    # new_lambs = [10000 + i for i in range(0, 41000, 1000)]

    # f = [(0.1, 0.7), (0.3, 0.5), (0.5, 0.3), (0.7, 0.1), (0.9, 0.05)]
    eps_c = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    eps_m = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    runs = 10000

    # run(N, D, lambds, runs)
    run(N, D, lambds, eps_c, eps_m, runs)