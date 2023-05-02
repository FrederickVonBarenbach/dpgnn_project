import numpy as np
from scipy.stats import hypergeom
from scipy.special import logsumexp

def get_N(K, r):
  return int((np.power(K, r + 1) - 1) / (K - 1))

def get_k_range(N, K, n):
  return range(max(0, n + K - N), min(K, n) + 1)

def get_gamma(dataset_size, batch_size, l2_norm_clip, noise_multiplier, r_hop, degree_bound, alpha, delta):
  N = int(dataset_size)
  K = get_N(degree_bound, r_hop)
  n = int(batch_size)
  rho = hypergeom(N, K, n)
  const = alpha * (alpha-1) * (2 * l2_norm_clip * l2_norm_clip) / (noise_multiplier * noise_multiplier)
  # calc expected value
  a = [] # values
  b = [] # weight
  for k in get_k_range(N, K, n):
    dist = rho.pmf(k)
    a = np.append(a, np.float128(const * k * k))
    b = np.append(b, dist)
  E_rho = logsumexp(a, b=b)
  return np.log(E_rho) / (alpha - 1)

def get_epsilon(gamma, iteration, alpha, delta):
  # convert RDP to DP
  return (iteration * gamma) - np.log(delta) / (alpha - 1)