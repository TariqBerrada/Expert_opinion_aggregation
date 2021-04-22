import numpy as np
import scipy.stats as st

class Expert():
    def __init__(self, error_std, n_seed, n_replicates):
        self.error_std = error_std # The observation error distribution is known to the expert.
        self.n_seed = n_seed
        self.n_replicates = n_replicates
    
    def obs_error(self, covariance):
        # observation_error = np.random.normal(loc = 0.0, scale = self.error_std, size = self.n_seed + self.n_replicates)
        variable = np.zeros((self.n_seed + self.n_replicates))
        distribution = st.multivariate_normal(mean = variable, cov = covariance)
        observation = distribution.rvs(size = 1)
        predictions = np.array([[obs, self.error_std] for obs in observation])
        return predictions    

class Variable():
    def __init__(self, value, experts_std):
        self.value = value
        self.experts_std = np.array(experts_std)
    
    def get_exp_predictions(self, covariance):
        distribution = st.multivariate_normal(mean = [self.value]*len(self.experts_std), cov = covariance)
        observation = distribution.rvs(size = 1)
        predictions = np.concatenate((observation[None], self.experts_std[None]), axis = 0).T # [n_experts, 2]
        return predictions