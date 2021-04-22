import numpy as np

# def estimate_covariance(X):
#     """
#     X is of shape (n, p), with n the number of experts and p = S + T is the size of the predictions.
#     In our case each column (or observation of an expert) is normal with mean 0 and covariane cov_p.
#     """
#     X_mean = np.mean(X, axis = 1)
#     n, p = X.shape
#     cov_est = np.zeros((n, n))
#     for i in range(p):
#         contrib_i = np.einsum('i,j->ij', X[:,i] - X_mean, X[:,i] - X_mean)
#         cov_est += contrib_i
#     cov_est = (1/(p-1))*cov_est
#     return cov_est

def estimate_covariance(X):
    return np.cov(X)