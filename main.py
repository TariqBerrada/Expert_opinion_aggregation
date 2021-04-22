import numpy as np

import sys
sys.path.append('.')

import tqdm

from expert import Expert, Variable
from utils.stats import estimate_covariance
from utils.score import get_weights, get_combined_score
from utils.combin import weighted_avg

import matplotlib.pyplot as plt

n_experts = 3
n_seed = 30 # S in the paper, number of seed variables.
n_replicates = 30 # T in the paper, number of replicates of the target.

for epoch in range(100):

    error_stds = [np.random.random() for i in range(n_experts)]
    experts = [Expert(error_std = error_stds[i], n_seed = n_seed, n_replicates = n_replicates) for i in range(n_experts)]
    variables = [Variable(0, error_stds) for _ in range(n_seed + n_replicates)]
    cov_init = np.eye(n_experts)
    cov_update = cov_init

    # y1 = []
    # s1 = []
    # y2 = []
    # s2 = []
    # y3 = []
    # s3 = []
    # y4 = []
    # s4 = []
    # s5 = []
    # y5 = []

    score_mean, score_w = [], []
    f_list = []

    for iteration in tqdm.tqdm(range(100)):
        # Generate expert predictions for the variables.
        expert_predictions = np.array([variable.get_exp_predictions(cov_update) for variable in variables]).transpose(1, 0, 2)
        # Update covariance matrix.
        X = np.array(expert_predictions[:, :n_seed, 0])
        cov_seed = estimate_covariance(X)
        cov_update = cov_seed
        # Calculate the weights for each expert.
        Tt = np.mean(expert_predictions[:, n_seed:, :], axis = 1)[None].transpose(1, 0, 2)
        observations = np.concatenate((expert_predictions[:, :n_seed, :], Tt), axis = 1)
        weights = get_weights(observations, expert_predictions[:, n_seed:, 0])
        # Test the weighted combination.
        combined_pred = weighted_avg(observations, weights)
        # Test the same weight combination.
        same_combined_pred = weighted_avg(observations, 1/n_experts*np.ones_like(weights))
        # Evaluate combined score for each method.
        score = get_combined_score(np.stack(same_combined_pred).T, observations, expert_predictions[:, :n_seed, :])
        avg_score = get_combined_score(np.stack(same_combined_pred).T, observations, expert_predictions[:, :n_seed, :])
        score_mean.append(avg_score)
        score_w.append(score)
        # print('w_score : ', score)
        # print('avg_score : ', avg_score)
        
        # y1.append(combined_pred[0][0])
        # s1.append(combined_pred[1][0])
        # y2.append(combined_pred[0][1])
        # s2.append(combined_pred[1][1])
        # y3.append(combined_pred[0][2])
        # s3.append(combined_pred[1][2])
        # y4.append(combined_pred[0][-2])
        # s4.append(combined_pred[1][-2])
        # y5.append(combined_pred[0][-1])
        # s5.append(combined_pred[1][-1])

    # calculate the frequeny with which each decision maker has the largest combined_score.
    f_w = len(np.where(np.subtract(score_w, score_mean)>0)[0])/len(score_w)
    print('\%weighted_avg > mean_avg : {}\%'.format(f_w))
    f_list.append(f_w)

    # y1 = np.array(y1)
    # y2 = np.array(y2)
    # y3 = np.array(y3)
    # y4 = np.array(y4)
    # s1 = np.array(s1)
    # s2 = np.array(s2)
    # s3 = np.array(s3)
    # s4 = np.array(s4)
    # s5 = np.array(s5)
    # y5 = np.array(y5)

    print('average : {} | weighted_avg : {}'.format(np.mean(score_mean), np.mean(score_w)))
    # x = list(range(len(y1)))
    # plt.plot(x, y1)
    # plt.fill_between(x, np.array(y1)-s1, np.array(y1)+s1, alpha = .5)
    # plt.plot(x, y2)
    # plt.fill_between(x, np.array(y2)-s2, np.array(y2)+s2, alpha = .5)
    # plt.plot(x, y3)
    # plt.fill_between(x, np.array(y3)-s3, np.array(y3)+s3, alpha = .5)
    # plt.plot(x, y4)
    # plt.fill_between(x, np.array(y4)-s4, np.array(y4)+s4, alpha = .5)
    # plt.plot(x, y5)
    # plt.fill_between(x, np.array(y5)-s5, np.array(y5)+s5, alpha = .5)
    # plt.legend(['1', '2', '3', '4', '5'])

    # plt.figure()
    # plt.plot(score_mean, label = 'average')
    # plt.plot(score_w, label = 'weighted average')

    # plt.show()

print(' - mean : ', np.mean(f_list))
print(' - var : ', np.var(f_list))
print('f > .5 : ', len(np.where(f_list > .5)[0]))
