import numpy as np
import sys
sys.path.append('../')

from src.utils import weighted_rls_solution, compute_eta, compute_private_estimator, evaluate_weighted_rls_objective, dataset_mask_jorgensen

## NON-PRIVATE SOLUTION

def nonpriv_solution(N_train, N_test, X_train, y_train, X_test, y_test, lamb, eval_lamb=0):

  uniform_weight_train = np.ones(N_train) / N_train
  exact_soln = weighted_rls_solution(uniform_weight_train, X_train, y_train, lamb)
  uniform_weight_test = np.ones(N_test) / N_test
  exact_loss_ridge = evaluate_weighted_rls_objective(exact_soln, uniform_weight_test, X_test, y_test, eval_lamb)

  return exact_loss_ridge


## PP-ESTIMATOR

def pp_estimator(epsilons, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0, non_personalized=False):
  '''
    X_train: np.ndarray of shape (n, d)
    epsilons: must be a numpy array of shape (len(X_train),)
  '''
  N_train, d =  X_train.shape
  N_test =  len(X_test)
  if non_personalized:
    epsilons = np.array([min(epsilons)]*N_train)
  tot_epsilon = np.sum(epsilons)
  weights_pp = epsilons / tot_epsilon # weights used in the ridge regression for personalized privacy
  sol_exact_ridge_pp = weighted_rls_solution(weights_pp, X_train, y_train, lamb) # weighted ridge on training set
  # sanity check
  if lamb == 0:
    eta_pp = 0
  else:
    eta_pp = compute_eta(lamb = lamb, tot_epsilon=tot_epsilon, d = d)
  # print("eta for pp", eta_pp)
  # Do runs, in each calculate loss on unweighted train, unweighted test set loss of the private estimator; 1000 runs for randomness in L2 laplce dp noise
  unweighted_train = []
  unweighted_test = []
  uniform_weight_train = np.ones(N_train) / N_train
  uniform_weight_test = np.ones(N_test) / N_test

  # print("pluggin exact soln back into weighted ridge", evaluate_weighted_rls_objective(sol_exact_ridge_pp, uniform_weight_test, X_test, y_test, eval_lamb))
  # exact_loss_ridge = []
  # weighted_erm = []
  theta_hat_pp_norm = []

  for _ in range(runs):
    # exact_loss_ridge.append(evaluate_weighted_rls_objective(sol_exact_ridge_pp, uniform_weight_test, X_test, y_test, eval_lamb))
    theta_hat_pp = compute_private_estimator(sol_exact_ridge_pp, eta_pp) # exact solution on weighted training + noise
    theta_hat_pp_norm.append(np.linalg.norm(theta_hat_pp))
    unweighted_train.append(evaluate_weighted_rls_objective(theta_hat_pp, uniform_weight_train, X_train, y_train, eval_lamb)) # evaluate with lambda = 0, don't add regularizer for evaluation!
    unweighted_test.append(evaluate_weighted_rls_objective(theta_hat_pp, uniform_weight_test, X_test, y_test, eval_lamb))
  
  return np.mean(unweighted_train), np.std(unweighted_train), np.mean(unweighted_test), np.std(unweighted_test), np.mean(theta_hat_pp_norm)

# JORGENSEN PRIVATE ESTIMATOR

def jorgensen_private_estimator(epsilons, thresh, X_train, y_train, X_test, y_test, lamb, runs, eval_lamb=0):
  '''
    epsilons: must be a numpy array of shape (len(X_train),)
    thresh : np.mean(epsilons) OR max(epsilons) -- Acc to Jorgensen (using max as of now)
    X_train: np.ndarray of shape (n, d)
    To ensure personalized privacy Jorgensen et.al first subsample from the training data acc personalized privacy levels
    then add noise
  '''
  N_train, d =  X_train.shape
  N_test =  len(X_test)
  # thresh = max(epsilons) #global threshold used in jorgensen sampling
  # to loop the part below
  uniform_weight_train = np.ones(N_train) / N_train
  uniform_weight_test = np.ones(N_test) / N_test

  unweighted_train = []
  unweighted_test = []
  theta_hat_norm = []

  for _ in range(runs):
    mask = dataset_mask_jorgensen(epsilons, thresh) # which datapoint in X_train, y_train to mask, shape (N_train)
    X_samp = X_train[mask.astype(bool)]
    y_samp = y_train[mask.astype(bool)]
    N_samp = len(y_samp)
    unif_weight_samp = np.ones(N_samp) / N_samp
    tot_epsilon = thresh * N_samp # Use global threshold epsilon as privacy level for each sampled datapoint
    # now do DP with global threshold thresh, on the sampled data, using our framewor/sensitivity calculations
    theta_bar = weighted_rls_solution(unif_weight_samp, X_samp, y_samp, lamb) # unweighted soln with sampled data
    if lamb == 0:
      eta = 0
    else:
      eta = compute_eta(lamb = lamb, tot_epsilon=tot_epsilon, d = d)
    theta_hat = compute_private_estimator(theta_bar, eta)
    theta_hat_norm.append(np.linalg.norm(theta_hat))
    unweighted_train.append(evaluate_weighted_rls_objective(theta_hat, uniform_weight_train, X_train, y_train, eval_lamb))
    unweighted_test.append(evaluate_weighted_rls_objective(theta_hat, uniform_weight_test, X_test, y_test, eval_lamb))
  return np.mean(unweighted_train), np.std(unweighted_train), np.mean(unweighted_test), np.std(unweighted_test), np.mean(theta_hat_norm)