import pandas as pd
from run import run_real_data
from tqdm import tqdm

# lambs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 7, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000]
lambs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 25, 50, 75, 100]
# lambs = [1e-2, 5e-2, 1e-1, 0.5, 1, 3, 5]
runs = 100

f_c=0.34
f_m=0.43
# eps_c=0.01
# eps_m=0.2
# eps_l=1.0
eps_c = 0.1
eps_m = 0.5
eps_l = 1.0

seed = 21

tot = len(lambs)
res = []
with tqdm(total = tot) as pbar:
    for lamb in lambs:
        di = run_real_data(runs=runs, ttsplit = 0.1, lamb = lamb, frac_train = 1.0, \
                    f_c=f_c, f_m=f_m, eps_c=eps_c, eps_m=eps_m, eps_l=eps_l, seed=seed)
        res.append(di)
        pbar.update(1)

df = pd.DataFrame(res)
df.to_pickle('lambda-var-real.pkl')