import sys
sys.path.append('../../')
from src.preprocessing import *
from src.utils import *
from src.estimator import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing


seed = 21 # random seed

def run():

    # Preprocessing
    # df_medical = pd.read_csv('../../../../datasets/insurance.csv')
    # numeric_all = ['age', 'bmi', 'children', 'charges']
    # cat_all = ['sex', 'smoker', 'region']
    
    # fetch dataset 
    housing = fetch_california_housing()
    
    # data (as pandas dataframes) 
    california = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target

    # numeric_all = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    # cat_all = ['Sex']

    # X = one_hot(X, cat_all)
    # X.drop(cat_all, axis = 1, inplace=True) # drop the categorics that were used to one hot encode
    # X = X * 1.0 # make bool true, false into 1.0, 0.0
    
    # X = np.c_[X, np.ones(X.shape[0])] # now (n, d+1) dimensional, linear regression in d+1 is affine in d
    X = california.to_numpy() 
    # label = 1*y
    # y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
    print("Training data x, y shapes", X_train.shape, y_train.shape)
    print("Test data x, y shapes", X_test.shape, y_test.shape)

    # N_test = len(y_test)
    # Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
    # Lamb = np.arange(0, 100.5, 0.5)[1:] #0.5, ... 100
    Lamb = [0.01, 0.05, 1, 3, 5, 7, 10, 15, 20, 30, 40, 50, 100, 150, 200]
    frac_of_train_dataset = np.arange(0.1, 1.1, 0.1) # fraction of training dataset used [0.1, ... 1.0]
    runs = 10000

    list_of_results = []
    i = 0
    for f in frac_of_train_dataset: # varying the number of trianing data points used by using fractions
        if f == 1.0: # sklearn can't split when trainset frac = 1.0
            X_tr_frac, y_tr_frac = X_train, y_train
        else:
            X_tr_frac, _ , y_tr_frac, _ = train_test_split(X_train, y_train, train_size = f, random_state = seed)
        N_train_frac, N_test = len(X_tr_frac), len(X_test)
        
        epsilons = set_epsilons(N_train_frac, f_c=0.54, f_m=0.37, eps_c=0.01, eps_m=0.2, eps_l=1.0)

        jorg_thresh_max, jorg_thresh_mean = max(epsilons), np.mean(epsilons)


        print(f"frac: {f}")
        for lamb in Lamb:
            
            # just for sanity check 
            _, _, unreg_pp_baseline_test_mean, unreg_pp_baseline_test_std, _ = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, 0, runs, eval_lamb=0, non_personalized=True)
            # just for sanity check 
            jorg_thresh = min(epsilons)
            _, _, unreg_jorg_baseline_test_mean, unreg_jorg_baseline_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh, X_tr_frac, y_tr_frac, X_test, y_test, 0, runs, eval_lamb=0)
            _, _, jorg_baseline_test_mean, jorg_baseline_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0)



            # 4.1 Type1
            unreg_pp_train_mean, unreg_pp_train_std, unreg_pp_test_mean, unreg_pp_test_std, _ = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0)
            _, _, unreg_nonpp_test_mean, unreg_nonpp_test_std, _ = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0, non_personalized=True)
            
            # 4.1 Type2
            reg_pp_train_mean, reg_pp_train_std, reg_pp_test_mean, reg_pp_test_std, _ = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb)
            _, _, reg_nonpp_test_mean, reg_nonpp_test_std, _ = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb, non_personalized=True)

            # 4.3 Type1
            unreg_jorg_max_train_mean, unreg_jorg_max_train_std, unreg_jorg_max_test_mean, unreg_jorg_max_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0)
            unreg_jorg_avg_train_mean, unreg_jorg_avg_train_std, unreg_jorg_avg_test_mean, unreg_jorg_avg_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh_mean, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=0)
            type1_nonpriv_loss = nonpriv_solution(N_train_frac, N_test, X_tr_frac, y_tr_frac, X_test, y_test, lamb=0, eval_lamb=0)

            # 4.3 Type2
            reg_jorg_max_train_mean, reg_jorg_max_train_std, reg_jorg_max_test_mean, reg_jorg_max_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh_max, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb)
            reg_jorg_avg_train_mean, reg_jorg_avg_train_std, reg_jorg_avg_test_mean, reg_jorg_avg_test_std, _ = jorgensen_private_estimator(epsilons, jorg_thresh_mean, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs, eval_lamb=lamb)
            type2_nonpriv_loss = nonpriv_solution(N_train_frac, N_test, X_tr_frac, y_tr_frac, X_test, y_test, lamb, eval_lamb=lamb)
            
            
            di = {"N_train_frac": N_train_frac,
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
                }
            print(f"Expt {i} done, N_train {N_train_frac}, lambda {lamb}")
            list_of_results.append(di)
            i += 1

    df = pd.DataFrame(list_of_results)
    df.to_csv(f'../forplots_california_housing_data_impact_n_plevel_54_37_9_affine.csv', encoding='utf-8', index=False)


if __name__ == "__main__":

    run()