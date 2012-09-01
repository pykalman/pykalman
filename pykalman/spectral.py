import numpy as np
from scipy import linalg

def _estimate_covariance(Z, offset=0):
  '''Estimate E[Z[t+offset] Z[t].T]'''
  T, n_dim_obs = Z.shape
  T -= offset

  if not T > 1:
      raise ValueError('Not enough time steps to estimate covariance')

  result = np.zeros([n_dim_obs, n_dim_obs])
  for t in range(T):
      result += np.outer(Z[t+offset], Z[t]) / (T-1)
  return result


def _spectral(observations, n_dim_state, given={}):
  '''Estimate parameters using spectral learning

  Parameters
  ----------
  observations: [T, n_dim_obs] array
      observations[t] = observation at time t

  Returns
  -------
  transition_matrix: [n_dim_state, n_dim_state] array
      estimated transition matrix
  observation_matrix: [n_dim_obs, n_dim_state] array
      estimated observation matrix
  '''
  T, n_dim_obs = observations.shape

  # estimate one-step and two-step covariance matrices
  Sigma_1 = _estimate_covariance(observations, offset=1)
  Sigma_2 = _estimate_covariance(observations, offset=2)

  # get first n_dim_state singular vectors of Sigma_1
  U = linalg.svd(Sigma_1)[0]
  U = U[:, 0:n_dim_state]

  # estimate matrices
  if 'transition_matrices' in given:
      transition_matrix = given['transition_matrices']
  else:
      transition_matrix = (
          U.T
          .dot(Sigma_2)
          .dot(linalg.pinv(U.T.dot(Sigma_1)))
      )

  if 'observation_matrices' in given:
      observation_matrix = given['observation_matrices']
  else:
      observation_matrix = U.dot(linalg.pinv(transition_matrix))

  return (
      transition_matrix, observation_matrix,
#      given['transition_offsets'], given['observation_offsets'],
#      given['transition_covariance'], given['observation_covariance'],
#      given['initial_state_mean'], given['initial_state_covariance']
  )
