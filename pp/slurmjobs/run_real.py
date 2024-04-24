import sys
sys.path.append('../')

from src.preprocessing import *
from src.utils import *
from src.estimator import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_exp(X_train, y_train, frac_trainset:float, lamb:float, \
            f_c : float, f_m : float, eps_c : float, eps_m : float, eps_l=1.0, seed = 21):
    '''
        X_train, y_train : the training split of the dataset
        frac_trainset: fraction of X_train, y_train to be used
    '''
    if frac_trainset == 1.0: # sklearn can't split when trainset frac = 1.0, REFACTOR to utils
        X_tr_frac, y_tr_frac = X_train, y_train
    else:
        X_tr_frac, _ , y_tr_frac, _ = train_test_split(X_train, y_train, train_size = frac_trainset, random_state = seed)
    
    N_train_frac = len(X_tr_frac)
    epsilons = set_epsilons(N_train_frac, f_c = fc, f_m = f_m, eps_c = eps_c, eps_m = eps_m, eps_l = eps_l)
    jorg_thresh_max, jorg_thresh_mean = max(epsilons), np.mean(epsilons)
    # TODO change

    # 4.1 Type1
    unreg_pp_train_mean, unreg_pp_train_std, unreg_pp_test_mean, unreg_pp_test_std, unreg_theta_hat_pp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0) # with personalized privacy
    _, _, unreg_nonpp_test_mean, unreg_nonpp_test_std, unreg_theta_hat_nonpp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0, non_personalized=True) # standard DP
    
    # 4.1 Type2
    reg_pp_train_mean, reg_pp_train_std, reg_pp_test_mean, reg_pp_test_std, reg_theta_hat_pp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb) # with personalized privacy
    _, _, reg_nonpp_test_mean, reg_nonpp_test_std, reg_theta_hat_nonpp_norm = pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb, non_personalized=True) # standard DP

    # 4.3 Type1 Jorgensen with max and mean threshold
    unreg_jorg_max_train_mean, unreg_jorg_max_train_std, unreg_jorg_max_test_mean, unreg_jorg_max_test_std, unreg_theta_hat_jorg_max_norm = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
    unreg_jorg_avg_train_mean, unreg_jorg_avg_train_std, unreg_jorg_avg_test_mean, unreg_jorg_avg_test_std, unreg_theta_hat_jorg_avg_norm = jorgensen_private_estimator(epsilons, jorg_thresh_mean, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0)
    type1_nonpriv_loss = nonpriv_solution(N_train, N_test, X_train, y_train, X_test, y_test, lamb=0, eval_lamb=0)

    # 4.3 Type2 Jorgensen with max and mean threshold
    reg_jorg_max_train_mean, reg_jorg_max_train_std, reg_jorg_max_test_mean, reg_jorg_max_test_std, reg_theta_hat_jorg_max_norm = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb)
    reg_jorg_avg_train_mean, reg_jorg_avg_train_std, reg_jorg_avg_test_mean, reg_jorg_avg_test_std, reg_theta_hat_jorg_avg_norm = jorgensen_private_estimator(epsilons, jorg_thresh_mean, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=lamb)
    type2_nonpriv_loss = nonpriv_solution(N_train, N_test, X_train, y_train, X_test, y_test, lamb, eval_lamb=lamb)
    
    

