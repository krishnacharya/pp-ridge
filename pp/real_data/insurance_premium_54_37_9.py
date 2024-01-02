import sys
sys.path.append('../')

from src.preprocessing import *
from src.utils import *
from src.estimator import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 21 # random seed

# Preprocessing
df_medical = pd.read_csv('../../datasets/insurance.csv')
numeric_all = ['age', 'bmi', 'children', 'charges']
cat_all = ['sex', 'smoker', 'region']
df_medical_mm = numeric_scaler(df_medical, numeric_all) # minmax scaling for all numeric columns
df_medical_mm_oh = one_hot(df_medical_mm, cat_all)
df_medical_mm_oh.drop(cat_all, axis = 1, inplace=True) # drop the categorics that were used to one hot encode
df_medical_mm_oh = df_medical_mm_oh * 1.0 # make bool true, false into 1.0, 0.0

# make X rows L2 norm 1, y is already in [0,1] from the above
X = df_medical_mm_oh.drop('charges', axis=1).to_numpy()
y = df_medical_mm_oh['charges'].to_numpy()
dim = X.shape[1] # dimensionality of the feature
X  = X / (dim**0.5)
# X = normalize(X, norm='l2') # each row is L2 normalized

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = seed)
print("Training data x, y shapes", X_train.shape, y_train.shape)
print("Test data x, y shapes", X_test.shape, y_test.shape)

N_test = len(y_test)
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
    for lamb in Lamb:
        pp_unw_train_mean, pp_unw_train_std, pp_w_test_mean, pp_w_test_std = pp_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, N_train_frac, N_test, runs)
        jorg_unw_train_mean, jorg_unw_train_std, jorg_w_test_mean, jorg_w_test_std = jorgensen_private_estimator(epsilons, X_tr_frac, y_tr_frac, X_test, y_test, lamb, N_train_frac, N_test, runs)
        di = {"frac_train": f,
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
        print(f"Expt {i} done, fraction {f}, lambda {lamb}")
        list_of_results.append(di)
        i += 1
df = pd.DataFrame(list_of_results)
df.to_csv('../insurance_data_plevel_54_37_9_sqrtd.csv', encoding='utf-8', index=False)



