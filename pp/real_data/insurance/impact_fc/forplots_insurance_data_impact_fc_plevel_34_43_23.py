import sys
sys.path.append('../../../')
from src.preprocessing import *
from src.utils import *
from src.estimator import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 21 # random seed

def run(lambds, f, runs=10000):

    # Preprocessing
    df_medical = pd.read_csv('../../../../datasets/insurance.csv')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
    print("Training data x, y shapes", X_train.shape, y_train.shape)
    print("Test data x, y shapes", X_test.shape, y_test.shape)

    # N_test = len(y_test)
    # Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
    # Lamb = np.arange(0, 100.5, 0.5)[1:] #0.5, ... 100
    # frac_of_train_dataset = np.arange(0.1, 1.1, 0.1) # fraction of training dataset used [0.1, ... 1.0]

    list_of_results = []
    i = 0
    
    N_train, N_test = len(X_train), len(X_test)


    for fc in f:
        epsilons = set_epsilons(N_train, f_c=fc, f_m=0.37, eps_c=0.01, eps_m=0.2, eps_l=1.0)

        jorg_thresh_max, jorg_thresh_mean = max(epsilons), np.mean(epsilons)

        Lamb = lambds
        
        for lamb in Lamb:
            
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
            
            
            
            di = {"f_c": fc,
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
                "reg_theta_hat_pp_norm": reg_theta_hat_nonpp_norm,
                "unreg_theta_hat_jorg_max_norm": unreg_theta_hat_jorg_max_norm,
                "unreg_theta_hat_jorg_avg_norm": unreg_theta_hat_jorg_avg_norm,
                "reg_theta_hat_jorg_max_norm": reg_theta_hat_jorg_max_norm,
                "reg_theta_hat_jorg_avg_norm": reg_theta_hat_jorg_avg_norm
                }
            print(f"Expt {i} done, f_c {fc}, lambda {lamb}")
            list_of_results.append(di)
            i += 1

    df = pd.DataFrame(list_of_results)
    df.to_csv(f'../../../csv_outputs/forplots_insurance_data_impact_fc_affine.csv', encoding='utf-8', index=False)


if __name__ == "__main__":

    fc = [0.1*i for i in range(1, 7)]

    Lamb = [0.01, 0.05, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 10]
    runs = 10000

    run(Lamb, fc, runs)