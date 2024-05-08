import sys
sys.path.append('../')

from src.preprocessing import *
from src.utils import *
from src.estimator import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_real_data(runs:int, ttsplit: float, lamb:float, frac_train: float, \
                    f_c:float, f_m:float, eps_c:float, eps_m:float, eps_l:float, seed:int):

    np.random.seed(seed = seed) # set seed for data generation below, and for sklearn randomness in test train split
    # Preprocessing
    df_medical = pd.read_csv('../../datasets/insurance.csv')
    numeric_all = ['age', 'bmi', 'children', 'charges']
    cat_all = ['sex', 'smoker', 'region']
    df_medical_mm = numeric_scaler(df_medical, numeric_all) # minmax scaling for all numeric columns, so all elements in [0,1]
    df_medical_mm_oh = one_hot(df_medical_mm, cat_all)
    df_medical_mm_oh.drop(cat_all, axis = 1, inplace=True) # drop the categorics that were used to one hot encode
    df_medical_mm_oh = df_medical_mm_oh * 1.0 # make bool true, false into 1.0, 0.0

    X = df_medical_mm_oh.drop('charges', axis=1)
    X['intercept'] = 1.0
    X = X.to_numpy() # now (n, d+1) dimensional, linear regression in d+1 is affine in d
    y = df_medical_mm_oh['charges'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ttsplit, random_state = seed)
    # print("Training data x, y shapes", X_train.shape, y_train.shape)
    # print("Test data x, y shapes", X_test.shape, y_test.shape)

    X_tr_frac, y_tr_frac = split_training_data(X_tr = X_train, y_tr = y_train, f = frac_train, seed = seed)
    N_train_frac, N_test = len(X_tr_frac), len(X_test)
    epsilons = set_epsilons(N_train_frac, f_c=f_c, f_m=f_m, eps_c=eps_c, eps_m=eps_m, eps_l=eps_l)

    jorg_thresh_max, jorg_thresh_avg = np.max(epsilons), np.mean(epsilons)
    di = {"runs":runs, "ttsplit":ttsplit, "lamb": lamb, "frac_train":frac_train, \
        "f_c":f_c, "f_m":f_m, "eps_c":eps_c, "eps_m":eps_m, "eps_l":eps_l} # just the values for this experiment
    # Type 1/Unregularized test loss for 1) our algorithm, 2) standard DP, 3,4) Jorgensen (max and avg thresh)
    di['type1_nonpriv_loss'] = nonpriv_solution(N_train_frac, N_test, X_tr_frac, y_tr_frac, X_test, y_test, lamb=0, eval_lamb=0) # test loss of the non private training data solution, equal weighted, no regularization

    # Our algorithm
    di['t1_oa_train_mean'], di['t1_oa_train_std'], di['t1_oa_test_mean'], di['t1_oa_test_std'], di['t1_thetahat_oa_mean'], di['t1_thetahat_oa_std'], \
    di['t1_thetadiff_oa_mean'], di['t1_thetadiff_oa_std']  = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0) 
    
    # Standard DP
    di['t1_sdp_train_mean'], di['t1_sdp_train_std'], di['t1_sdp_test_mean'], di['t1_sdp_test_std'], di['t1_thetahat_sdp_mean'], di['t1_thetahat_sdp_std'], \
    di['t1_thetadiff_sdp_mean'], di['t1_thetadiff_sdp_std']  = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0, non_personalized=True)

    # Jorgensen with max and average epsilon threshold 
    di['t1_jorgmax_train_mean'], di['t1_jorgmax_train_std'], di['t1_jorgmax_test_mean'], di['t1_jorgmax_test_std'], di['t1_thetahat_jorgmax_mean'], di['t1_thetahat_jorgmax_std'], \
    di['t1_thetadiff_jorgmax_mean'], di['t1_thetadiff_jorgmax_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0)
    
    di['t1_jorgavg_train_mean'], di['t1_jorgavg_train_std'], di['t1_jorgavg_test_mean'], di['t1_jorgavg_test_std'], di['t1_thetahat_jorgavg_mean'], di['t1_thetahat_jorgavg_std'], \
    di['t1_thetadiff_jorgavg_mean'], di['t1_thetadiff_jorgavg_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_avg, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0)
  
    # Type 2/ Regularized test loss for  1) our algorithm, 2) standard DP, 3,4) Jorgensen (max and avg thresh)

    # Type 2
    # Our algorithm
    di['t2_oa_train_mean'], di['t2_oa_train_std'], di['t2_oa_test_mean'], di['t2_oa_test_std'], \
    di['t2_thetahat_oa_mean'], di['t2_thetahat_oa_std'], \
    di['t2_thetadiff_oa_mean'], di['t2_thetadiff_oa_std'] = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb)


     # Standard DP
    di['t2_sdp_train_mean'], di['t2_sdp_train_std'], di['t2_sdp_test_mean'], di['t2_sdp_test_std'], \
    di['t2_thetahat_sdp_mean'], di['t2_thetahat_sdp_std'], \
    di['t2_thetadiff_sdp_mean'], di['t2_thetadiff_sdp_std']  = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb, non_personalized=True)

    # Jorgensen with max epsilon threshold 
    di['t2_jorgmax_train_mean'], di['t2_jorgmax_train_std'], di['t2_jorgmax_test_mean'], di['t2_jorgmax_test_std'], \
    di['t2_thetahat_jorgmax_mean'], di['t2_thetahat_jorgmax_std'], \
    di['t2_thetadiff_jorgmax_mean'], di['t2_thetadiff_jorgmax_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb)
    
    # Jorgensen with average epsilon threshold 
    di['t2_jorgavg_train_mean'], di['t2_jorgavg_train_std'], di['t2_jorgavg_test_mean'], di['t2_jorgavg_test_std'], \
    di['t2_thetahat_jorgavg_mean'], di['t2_thetahat_jorgavg_std'], \
    di['t2_thetadiff_jorgavg_mean'], di['t2_thetadiff_jorgavg_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_avg, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb)

    return di



def run_linear_synth(N:int, d:int, sigma:float, runs:int, ttsplit: float, lamb:float, frac_train: float, \
                    f_c:float, f_m:float, eps_c:float, eps_m:float, eps_l:float, seed:int) -> dict:
    '''
        N: synthetic dataset size
        d: dimension of synthetic features
        sigma: std of gaussian noise added to label
        
        runs: number of runs of private estimator computation on the same data
        ttsplit: desired fraction of test set for train test split

        lamb: regularization for ridgereg
        frac_train: fraction of training set to use for private estimator
        f_c, f_m, eps_c, eps_m, eps_l : see paper

        seed: seed for numpy
    '''
    np.random.seed(seed = seed) # set seed for data generation below, and for sklearn randomness in test train split
    X, y, theta_star = generate_linear_data(n = N, d = d, sigma = sigma) # generate data once use multiple times over the runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ttsplit, random_state = seed)
    
    X_tr_frac, y_tr_frac = split_training_data(X_tr = X_train, y_tr = y_train, f = frac_train, seed = seed)
    N_train_frac, N_test = len(X_tr_frac), len(X_test)
    epsilons = set_epsilons(N_train_frac, f_c=f_c, f_m=f_m, eps_c=eps_c, eps_m=eps_m, eps_l=eps_l)

    jorg_thresh_max, jorg_thresh_avg = np.max(epsilons), np.mean(epsilons)
    di = {"N":N, "d":d, "sigma": sigma, "runs":runs, "ttsplit":ttsplit, "lamb": lamb, "frac_train":frac_train, \
        "f_c":f_c, "f_m":f_m, "eps_c":eps_c, "eps_m":eps_m, "eps_l":eps_l} # just the values for this experiment
    # Type 1/Unregularized test loss for 1) our algorithm, 2) standard DP, 3,4) Jorgensen (max and avg thresh)
    di['type1_nonpriv_loss'] = nonpriv_solution(N_train_frac, N_test, X_tr_frac, y_tr_frac, X_test, y_test, lamb=0, eval_lamb=0) # test loss of the non private training data solution, equal weighted, no regularization

    # Our algorithm
    di['t1_oa_train_mean'], di['t1_oa_train_std'], di['t1_oa_test_mean'], di['t1_oa_test_std'], di['t1_thetahat_oa_mean'], di['t1_thetahat_oa_std'], \
    di['t1_thetadiff_oa_mean'], di['t1_thetadiff_oa_std']  = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0, theta_star=theta_star) 
    
    # Standard DP
    di['t1_sdp_train_mean'], di['t1_sdp_train_std'], di['t1_sdp_test_mean'], di['t1_sdp_test_std'], di['t1_thetahat_sdp_mean'], di['t1_thetahat_sdp_std'], \
    di['t1_thetadiff_sdp_mean'], di['t1_thetadiff_sdp_std']  = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0, non_personalized=True, theta_star=theta_star)

    # Jorgensen with max and average epsilon threshold 
    di['t1_jorgmax_train_mean'], di['t1_jorgmax_train_std'], di['t1_jorgmax_test_mean'], di['t1_jorgmax_test_std'], di['t1_thetahat_jorgmax_mean'], di['t1_thetahat_jorgmax_std'], \
    di['t1_thetadiff_jorgmax_mean'], di['t1_thetadiff_jorgmax_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0, theta_star=theta_star)
    
    di['t1_jorgavg_train_mean'], di['t1_jorgavg_train_std'], di['t1_jorgavg_test_mean'], di['t1_jorgavg_test_std'], di['t1_thetahat_jorgavg_mean'], di['t1_thetahat_jorgavg_std'], \
    di['t1_thetadiff_jorgavg_mean'], di['t1_thetadiff_jorgavg_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_avg, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0, theta_star=theta_star)
  
    # Type 2/ Regularized test loss for  1) our algorithm, 2) standard DP, 3,4) Jorgensen (max and avg thresh)
    # TODO

    # Type 2
    # Our algorithm
    di['t2_oa_train_mean'], di['t2_oa_train_std'], di['t2_oa_test_mean'], di['t2_oa_test_std'], \
    di['t2_thetahat_oa_mean'], di['t2_thetahat_oa_std'], \
    di['t2_thetadiff_oa_mean'], di['t2_thetadiff_oa_std'] = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb, theta_star=theta_star)


     # Standard DP
    di['t2_sdp_train_mean'], di['t2_sdp_train_std'], di['t2_sdp_test_mean'], di['t2_sdp_test_std'], \
    di['t2_thetahat_sdp_mean'], di['t2_thetahat_sdp_std'], \
    di['t2_thetadiff_sdp_mean'], di['t2_thetadiff_sdp_std']  = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb, non_personalized=True, theta_star=theta_star)

    # Jorgensen with max epsilon threshold 
    di['t2_jorgmax_train_mean'], di['t2_jorgmax_train_std'], di['t2_jorgmax_test_mean'], di['t2_jorgmax_test_std'], \
    di['t2_thetahat_jorgmax_mean'], di['t2_thetahat_jorgmax_std'], \
    di['t2_thetadiff_jorgmax_mean'], di['t2_thetadiff_jorgmax_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb, theta_star=theta_star)
    
    # Jorgensen with average epsilon threshold 
    di['t2_jorgavg_train_mean'], di['t2_jorgavg_train_std'], di['t2_jorgavg_test_mean'], di['t2_jorgavg_test_std'], \
    di['t2_thetahat_jorgavg_mean'], di['t2_thetahat_jorgavg_std'], \
    di['t2_thetadiff_jorgavg_mean'], di['t2_thetadiff_jorgavg_std'] = jorgensen_private_estimator(epsilons, jorg_thresh_avg, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb, theta_star=theta_star)

    return di

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
    epsilons = set_epsilons(N_train_frac, f_c = f_c, f_m = f_m, eps_c = eps_c, eps_m = eps_m, eps_l = eps_l)
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
    
    pass