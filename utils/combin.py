import numpy as np

# The weight of each expert is his combined score.

def sum(distributions):
    mean = np.sum([distribution[0] for distribution in distributions])
    std = np.sqrt(np.sum([distribution[1]**2 for distribution in distributions]))
    return (mean, std)

def normalize_weights(weights, alpha = 0.01):
    """[summary]

    Args:
        weights ([type]): [n_experts, S+1]
    """
    new_weights=  weights.copy()
    new_weights[np.where(weights < alpha)] = 0.
    norm = np.sum(new_weights, axis = 0)
    new_weights = new_weights/norm
    return new_weights

"""def weighted_avg(distributions, weights):
    assert len(distributions) == len(weights), 'Must have the same amount of experts as weights in order to combine distributions!'
    print('shapes__________', distributions.shape, weights.shape)
    n_weights = normalize_weights(weights)
    S_1 = 1/np.sum(n_weights, axis = 0)
    print('s1', S_1.shape)
    # mean = S_1*np.sum([weights[i, :]*distributions[i, :, 0] for i in range(len(distributions))])
    # mean = S_1*np.dot(weights, distributions[:, :, 0], axis = 0)
    mean = S_1*np.einsum('ij,ij->j', n_weights, distributions[:, :, 0])
    print('mean', mean.shape)
    var = S_1**2*np.einsum('ij,ij->j', n_weights**2, distributions[:, :, 0]**2+distributions[:, :, 1]**2) - mean**2
    # var = S_1*np.sum([weights[i, :]**2*(distributions[i, :, 0]**2+distributions[i, :, 1]**2)  for i in range(len(distributions))])- mean**2 
    print(var.shape)
    # print('w', weights)
    return (mean, var)"""

def weighted_avg(distributions, weights):
    # print('___', distributions.shape, weights.shape)
    # weighted_means = np.einsum('ij,i->ij', distributions[:, :, 0], weights)
    weighted_means = np.multiply(distributions[:, :, 0], weights)
    # print(weighted_means.shape, 'wwwwwwwww')
    mean = np.sum(weighted_means, axis = 0)
    # print('shape of mena', mean.shape)
    # weighted_std = np.einsum('ij,i->ij', distributions[:, :, 1], weights)
    weighted_std = np.multiply(distributions[:, :, 1], weights)
    std = np.sqrt(np.mean(weighted_std**2, axis = 0))
    # print('go', mean.shape, std.shape)
    return mean, std
