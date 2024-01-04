from sklearn.preprocessing import normalize, MinMaxScaler
import numpy as np

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

def evaluate_weighted_rls_objective(theta: np.ndarray, weights:np.ndarray, X: np.ndarray, y:np.ndarray, lamb:float = 0.0) -> float:
  """
    Inputs
      theta: candidate vector for which we want to evaluate the objective value, shape (d, 1) or (d,)
      weights: one for each datapoint, shape (n,1) or (n,)
      X: design matrix shape (n, d)
      y: yvalues shape (n,1) or (n,)
      lamb: L2 regularization parameter, default 0.0
    Note: If you want to evaluate the objective on equal weighted rls objective just pass weights as 1/n...1/n
    Returns MSE of weighted sse + regularization
  """
  n, d = X.shape
  theta = theta.reshape(d, -1) # now shape (d, 1)
  weights = weights.reshape(n, -1) #now shape (n, 1)
  X_w = X * (weights**0.5) # shape (n, d)
  y_w = y.reshape(n, -1) * (weights**0.5) # shape (n, 1)
  return np.sum((X_w @ theta - y_w)**2) + lamb * np.linalg.norm(theta)**2

def sample_l2lap(eta:float, d:int) -> np.array:
  """
    Returns
      d dimensional noise sampled from `L2 laplace'
      https://math.stackexchange.com/questions/3801271/sampling-from-a-exponentiated-multivariate-distribution-with-l2-norm
  """
  R = np.random.gamma(d, scale = 1.0/eta)
  Z = np.random.normal(0, 1, size = d)
  return R * (Z / np.linalg.norm(Z)) #shape is (d,) one dimensional

def compute_eta(lamb:float, tot_epsilon:float, d:int, B_lambda = None):
  '''
    lamb: regularization parameter for ridge
    tot_epsilon: sum of all the agents privacy requirements \sum_i epsilon_i
    d: dimensionality of the data, i.e.,  x_i lies in [0,1]^d
    B_lambda: upper bound for norm of thetabar 

    Returns
      eta used for L2 laplace central noise
  ''' 
  if B_lambda is None: # Unless explicitly given a bound B just use 1/sqrt(lambda)
    B_lambda = min(lamb**(-0.5), d**0.5 / lamb)
  Dr = 2*(d*B_lambda + d**0.5) # denominator in eta expresion
  return lamb * tot_epsilon / Dr

def compute_private_estimator(minimizer:np.ndarray, eta:float) -> np.ndarray:
  '''
    Private estimate after adding L2 laplace noise, 
    note eta is calculated using epsilon required by each agent
    minimizer: shape (d,1), minimizer of ridge regression
  '''
  d = len(minimizer)
  if eta == 0:
    return minimizer
  else:
    return minimizer + sample_l2lap(eta, d).reshape(d, -1)

def generate_linear_data(n:int, d:int, sigma:float):
  '''
    Input:
      n: number of datapoints
      d: dimension of feature
      sigma: std for gaussian noise for the synthetic linear data
    returns
      X the design matrix shape is (n x d) each element in the matrix is in [0, 1]
      y the associated labels shape is (n,) each y is in [-1, 1]
  '''
  X = np.random.rand(n, d) # entries uniform in [0,1), thus each row ||x_i|| bounded by \sqrt{d}
  theta = np.random.normal(0, 1, d) # d iid Gaussian each entry N(0, 1)
  theta = theta / np.linalg.norm(theta) # theta is uniformly distributed on the surface of unit sphere, shape (d,)
  y = (X @ theta) / d**0.5 + np.random.normal(0, sigma, size=n) # so essentially the learner is trying to recover \theta / d**0.5
  return X, y #return actual theta = theta/sqrt{d}, return that too

def dataset_mask_jorgensen(epsilons:np.ndarray, thresh:float) -> np.ndarray:
    '''
    randomized sampling of dataset using definition 9 from
    Conservative or Liberal? Personalized Differential 
    Privacy  https://wrap.warwick.ac.uk/67370/7/WRAP_Cormode_pdp.pdf

    epsilons: privacy requirement of each agent, shape (n,)
    thres: global threshold in defn 9

    Returns:
        binary array representing with 1 meaning pick that datapoint, 0 meaning don't pick
    TODO threshold_average
    '''
    p = np.clip((np.exp(epsilons) - 1) / (np.exp(thresh) - 1), None, 1.0) # shape (n,); bernoulli probabilities for each datapoint
    select = (np.random.random(len(p)) < p).astype(np.uint32) # binary array shape (n,); 1 means select that datapoint
    return select
    # return X[select.astype(bool)]

def epsilons_54_37_9(N_train : int) -> np.ndarray:
  '''
      3 privacy levels
      values chosen from this paper : "Utility-aware Exponential Mechanism for Personalized Differential Privacy"
  '''
  epsilons = np.zeros(N_train)# training datas agents privacy levels, want 3 privacy levels
  epsilons[:int(0.54*N_train)] = 0.1 # 54% care abt privacy (FUNDAMENTALISTS)
  epsilons[int(0.54*N_train) : int(0.91 * N_train)] = 0.5 # 37% care little abt privacy (PRAGMATISTS)
  epsilons[int(0.91 * N_train) : ] = 1.0 # 9% dont care (UNCONCERNED)
  return epsilons

def epsilons_34_43_23(N_train: int) -> np.ndarray:
  '''
      3 privacy levels
      values chosen from this paper : "Heterogeneous differential privacy"
  '''
  epsilons = np.zeros(N_train)# training datas agents privacy levels, want 3 privacy levels
  # values chosen from this paper : "Heterogeneous differential privacy"
  epsilons[:int(0.34*N_train)] = 0.1 # 34% care abt privacy (FUNDAMENTALISTS)
  epsilons[int(0.34*N_train) : int(0.77 * N_train)] = 0.5 # 43% care little abt privacy (PRAGMATISTS)
  epsilons[int(0.77 * N_train) : ] = 1.0 # 23% dont care (UNCONCERNED)
  return epsilons

def set_epsilons(N_train : int, f_c : float, f_m : float, eps_c : float, eps_m : float, eps_l=1.0) -> np.ndarray:
  '''
  Distribution of privacy levels

  f_c : fraction of high privacy level individuals (high)
  f_m : fraction of medium privacy level individuals (medium)
  f_l : fraction of low privacy level individuals (1-(f_c+f_m)) (low)
  eps_c : high privacy level (lower epsilon)
  eps_m : medium privacy level
  eps_l : low privacy level (higher epsilon)

  Returns :
    An array with values of the privacy levels for every individual in the dataset
  '''
  N_low, N_medium = int(f_c*N_train), int(f_m*N_train)
  N_high = N_train - N_low - N_medium
  c = np.random.uniform(eps_c, eps_m, size = N_low)
  m = np.random.uniform(eps_m, eps_l, size = N_medium)
  h = np.array([eps_l] * N_high)
  return np.concatenate((c, m, h), axis=None)