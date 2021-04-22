import numpy as np

from pandas import qcut
from scipy.stats import chi2, norm

import matplotlib.pyplot as plt

def get_calibration(distribution, realizations, b = 10, plot = False):
    """
    b : number of bins.
    distribution of the variable and realizations of the variable. (mean, std), float list of length T.
    """
    mean, std = distribution
    # distribution will be divided into b bins delimited by the deciles of the disribution.
    pdf = norm.pdf(np.linspace(-5, 5, 100))
    rvf = norm.rvs(size = 1000, loc = mean, scale = std)
    deciles = set(qcut(rvf, b).to_numpy())
    deciles = [x.right for x in deciles]
    if plot == True:
        plt.plot(np.linspace(-5, 5, 100), pdf)
        x_axis = deciles
        y_axis = [norm.pdf(x) for x in x_axis]
        plt.scatter(x_axis, y_axis)
        plt.show()
    deciles = sorted([-5] + deciles[:-1] + [5])
    bins = [[deciles[i], deciles[i+1]] for i in range(len(deciles)-1)]

    p_content = [0.1 for i in range(10)]
    counts, _= np.histogram(realizations, deciles)
    counts = counts/np.sum(counts)

    # Now that we have the bins, we should get the 
    right_tail = 2*len(realizations)*np.sum([counts[i]*np.log((counts[i]/p_content[i] if counts[i] != 0 else 1)) for i in range(len(p_content))])

    C = 1 - chi2.pdf(right_tail, df = b - 1)
    
    return C

def get_l_u(distributions):
    """
    Contains the realizations of the variable t for each expert. (n_experts, 2)
    """
    l = [np.quantile(norm.rvs(size = 1000, loc = pred[0], scale = pred[1]), 0.05) for pred in distributions]
    u = [np.quantile(norm.rvs(size = 1000, loc = pred[0], scale = pred[1]), 0.05) for pred in distributions]
    return l, u

def get_informativeness(distribution, realizations, b, us, ls, theta = 0):
    mean, std = distribution

    fractiles = [0 for _ in range(b+1)]
    
    rvf = norm.rvs(size = 1000, loc = mean, scale = std)
    ks = np.linspace(0, 1, b+1)
    for i in range(1, b):
        fractiles[i] = np.quantile(rvf, ks[i])

    l0 = np.min((*ls, theta))
    u0 = np.max((*us, theta))
    
    l = l0 - 0.1*(u0 - l0)
    u = u0 + 0.1*(u0 - l0)
    
    fractiles[0] = l
    fractiles[-1] = u
    
    p_content = [0.1 for i in range(b)]

    I = np.sum([p_content[i]*np.log(p_content[i]/((fractiles[i] - fractiles[i-1])/(u-l)))])
    return I

def get_combined_score(combined_distribution, distributions, realizations, b = 10):
    combined_scores = []
    C_list = []
    I_list = []
    for var_id in range(distributions.shape[1]):
        C = get_calibration(combined_distribution[var_id], np.mean(realizations, axis = 0), b = b)
        C_list.append(C)
        ls, us = get_l_u(distributions[:, var_id, :])
        I = get_informativeness(combined_distribution[var_id], np.mean(realizations, axis = 0), b, us, ls)
        I_list.append(I)
    combined = np.multiply(C_list, I_list)
    combined_scores.append(combined)
    return np.array(combined_scores)[0, -1]

def get_weights(distributions, realizations, b = 10):
    """[summary]

    Args:
        distributions ([type]): [n_experts, S+1, 2]
        realizations ([type]): [n_experts, 30]
        b (int, optional): [description]. Defaults to 10.
    """
    # assert distributions.shape[1] == realizations.shape[1], 'number of variables does not match!'
    combined_scores = []
    for exp_id in range(distributions.shape[0]):
        C_list = []
        I_list = []
        for var_id in range(distributions.shape[1]):
            C = get_calibration(distributions[exp_id, var_id], realizations[exp_id], b = b)
            C_list.append(C)

            ls, us = get_l_u(distributions[:, var_id, :])
            I = get_informativeness(distributions[exp_id, var_id], realizations[exp_id], b, us, ls)
            I_list.append(I)
        combined = np.multiply(C_list, I_list)
        combined_scores.append(combined)
    return np.array(combined_scores)


# ls, us = get_l_u(expert_predictions[:, 0, :])

# I = get_informativeness((0, 1), [np.random.normal() for _ in range(30)], 10, us, ls)

# print(I)