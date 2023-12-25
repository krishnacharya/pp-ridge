from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def weighted_rls_solution(weights:np.ndarray, X: np.ndarray, y:np.ndarray, lamb:float = 1.0) -> np.ndarray:
  """
    Inputs
      weights: one for each datapoint, shape (n,1) or (n,)
      X: design matrix shape (n, d)
      y: yvalues shape (n,1) or (n,)
      lamb: L2 regularization parameter, default 1.0
    Returns
      dx1 solution theta
  """
  n, d = X.shape
  weights = weights.reshape(n,-1)
  X_w = X * (weights**0.5)
  y_w = y.reshape(n, -1) * (weights**0.5)
  return np.linalg.solve(lamb * np.identity(d) + X_w.T @ X_w, X_w.T @ y_w)

def evaluate_weighted_rls_objective(theta: np.ndarray, weights:np.ndarray, X: np.ndarray, y:np.ndarray, lamb:float = 1.0) -> float:
  """
    Inputs
      theta: candidate vector for which we want to evaluate the objective value, shape (d, 1) or (d,)
      weights: one for each datapoint, shape (n,1) or (n,)
      X: design matrix shape (n, d)
      y: yvalues shape (n,1) or (n,)
      lamb: L2 regularization parameter, default 1.0
    Note: If you want to evaluate the objective on equal weighted rls objective just pass weights as 1/n...1/n
  """
  n, d = X.shape
  theta = theta.reshape(d, -1) # now shape (d, 1)
  weights = weights.reshape(n, -1) #now shape (n, 1)
  X_w = X * (weights**0.5) # shape (n, d)
  y_w = y.reshape(n, -1) * (weights**0.5) # shape (n, 1)
  return np.sum((X_w @ theta - y_w)**2) + lamb * np.linalg.norm(theta)**2

def sample_l2lap(beta:float, d:int) -> np.array:
  """
    Returns
      d dimensional noise sampled from `L2 laplace'
      https://math.stackexchange.com/questions/3801271/sampling-from-a-exponentiated-multivariate-distribution-with-l2-norm
  """
  R = np.random.gamma(d, scale = 1.0/beta)
  Z = np.random.normal(0, 1, size = d)
  return R * (Z / np.linalg.norm(Z)) #shape is (d,) one dimensional

def compute_beta(lamb, tot_epsilon):
  '''
    lamb: regularization parameter for ridge
    tot_epsilon: sum of all the agents privacy requirements \sum_i \varepsilon_i]
    Returns
      beta used for L2 laplace central noise
  '''
  return (lamb/2) * (lamb**0.5 / (1 + lamb**0.5)) * tot_epsilon

def compute_private_estimator(minimizer:np.ndarray, beta:float) -> np.ndarray:
  '''
    Private estimate after adding L2 laplace noise, note beta is calculated using epsilon required by each agent
    
    minimizer: shape (d,1), minimizer of ridge regression
  '''
  d = len(minimizer)
  return minimizer + sample_l2lap(beta, d).reshape(d, -1)

def generate_linear_data(n:int, theta:np.array, sigma:float):
  '''
    Input:
      n: number of datapoints
      theta: the true theta used to generate the data, shape (d,), one dimensional array
      sigma: std for gaussian noise for the synthetic linear data
    returns
      X the design matrix shape is (n x d)
      y the associated labels shape is (n,)
  '''
  X = np.random.rand(n, len(theta))
  y = X @ theta + np.random.normal(0, sigma, size=n)
  return X, y

def one_hot(df, cols): # idk if sklearns one-hot encoder is similar
    """
    df: pandas DataFrame
    param: cols a list of columns to encode 
    return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

def numeric_scaler(df, cols):
    '''
    df: pandas dataframe
    numeric_cols: (array of strings) column names for numeric variables

    no return: does inplace operation
    '''
    df_new = df.copy()
    mmscaler = MinMaxScaler()
    df_new[cols] = mmscaler.fit_transform(df_new[cols])
    return df_new

def dataset_mask_jorgensen(epsilons:np.ndarray, thresh:float) -> np.ndarray:
    '''
    randomized sampling of dataset using definition 9 from
    Conservative or Liberal? Personalized Differential 
    Privacy  https://wrap.warwick.ac.uk/67370/7/WRAP_Cormode_pdp.pdf

    epsilons: privacy requirement of each agent, shape (n,)
    thres: global threshold in defn 9

    Returns:
        binary array representing with 1 meaning pick that datapoint, 0 meaning don't pick
    '''
    p = np.clip((np.exp(epsilons) - 1) / (np.exp(thresh) - 1), None, 1.0) # shape (n,); bernoulli probabilities for each datapoint
    select = (np.random.random(len(p)) < p).astype(np.uint32) # binary array shape (n,); 1 means select that datapoint
    return select
    # return X[select.astype(bool)]