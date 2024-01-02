import sys
sys.path.append('../')
from src.preprocessing import *
from src.utils import *
from src.estimator import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 21 # random seed

# Preprocessing
df_medical = pd.read_csv('../../datasets/insurance.csv')
numeric_all = ['age', 'bmi', 'children', 'charges']
cat_all = ['sex', 'smoker', 'region']
df_medical_mm = numeric_scaler(df_medical, numeric_all) # minmax scaling for all numeric columns, so all elements in [0,1]
df_medical_mm_oh = one_hot(df_medical_mm, cat_all)
df_medical_mm_oh.drop(cat_all, axis = 1, inplace=True) # drop the categorics that were used to one hot encode
df_medical_mm_oh = df_medical_mm_oh * 1.0 # make bool true, false into 1.0, 0.0

X = df_medical_mm_oh.drop('charges', axis=1).to_numpy()
y = df_medical_mm_oh['charges'].to_numpy()
# X = normalize(X, norm='l2') # each row is L2 normalized, no need to do this, each x_i in [0,1] works with the new sensitivity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = seed)
print("Training data x, y shapes", X_train.shape, y_train.shape)
print("Test data x, y shapes", X_test.shape, y_test.shape)
dim = X_train.shape[1] # d
# N_test = len(y_test)
Lamb = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]
frac_of_train_dataset = np.arange(0.1, 1.1, 0.1) # fraction of training dataset used [0.1, ... 1.0]
runs = 10000

list_of_results = []
i = 0
for f in frac_of_train_dataset: # varying the number of trianing data points used by using fractions
    if f == 1.0: # sklearn can't split when trainset frac = 1.0
        X_tr_frac, y_tr_frac = X_train, y_train
    else:
        X_tr_frac, _ , y_tr_frac, _ = train_test_split(X_train, y_train, train_size = f, random_state = seed)
    N_train_frac = len(X_tr_frac) 
    epsilons = epsilons_54_37_9(N_train_frac)
    print(f"frac: {f}")
    for lamb in Lamb:
        lamb = lamb * (dim**0.5)
        pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs)
        jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, runs)
        di = {"frac_train": f,
            "N_train": N_train_frac,
            "lamb": lamb,
            "pp_train_mean": pp_unw_train_mean,
            "pp_train_std": pp_unw_train_std,
            "pp_test_mean": pp_w_test_mean,
            "pp_test_std": pp_w_test_std,
            "jorg_train_mean": jorg_unw_train_mean,
            "jorg_train_std": jorg_unw_train_std,
            "jorg_test_mean": jorg_w_test_mean,
            "jorg_test_std": jorg_w_test_std
            }
        print(f"Expt {i} done, lambda {lamb}")
        list_of_results.append(di)
        i += 1
df = pd.DataFrame(list_of_results)
df.to_csv('../insurance_data_plevel_54_37_9_sqrtdlambda.csv', encoding='utf-8', index=False)