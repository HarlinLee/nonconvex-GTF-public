from sklearn.neighbors import kneighbors_graph
import csv
from datetime import datetime
import sys
import numpy as np
sys.path.append('../GTF3')
from Utilities import *

# Semi-supervised classification on graph.
# Table 3 in paper.

max_evals = 1
pspace = (hp.loguniform('gamma', -4, 4),
        hp.loguniform('rho_mult', -3, 3)
)

PENALTIES = ['L1', 'SCAD-L1', 'MCP-L1']
ORDER_K = [0, 1]
eps = 0.01
num_trial = 10
percent_seeds = 0.2

delim = ','

data_dir = '../datasets/UCI_data/preprocessed/'
results_dir = '../datasets/UCI_data/results/'

""
fn = 'iris'
""
print (fn)

# read data files
features = np.loadtxt(data_dir + fn + '.features', dtype='float', delimiter=delim)
classes = np.loadtxt(data_dir + fn + '.classes', dtype='float', delimiter=delim)
print (features.shape)

# normalize features to have 0 mean and variance 1 (standard-)
for col in range(features.shape[1]):
    features[:, col] = (features[:, col]-features[:, col].mean())/features[:, col].std()

n = features.shape[0]
K = int(max(classes) + 1)

Y_true = np.zeros((n, K))
print (K)

for j in range(K):
    Y_true[:, j] = 1.0 * (classes == j)

# W, D : construct 5-NN graph based on the Euclidean distance between provided features
Dist = kneighbors_graph(features, 5, mode='distance')

#Dist[Dist > 0] = 1.0  # try the no weight version
W = Dist.copy()
W[Dist > 0] = np.exp(-Dist[Dist > 0])

G = nx.Graph(W.power(0.5))  # make sure G is an undirected graph
D = nx.incidence_matrix(G, nodelist=G.nodes(), oriented=True, weight='weight').T

# R : uniform prior belief
R = 1.0 / K * np.ones((n, K))

# Y, mask : randomly select 20% seeds per class to serve as the observed class labels.
# Save the seeds per experiment so that we can make a fair comparison between each methods (or perform paired t-tests)
seeds = np.zeros((int(percent_seeds*n), K, num_trial))
largest_num_seeds = 0

for trial in range(num_trial):
    for j in range(K):
        inds = np.argwhere(Y_true[:, j] > 0).ravel()
        num_seeds = max(int(percent_seeds*len(inds)), 1)  # have 20%, or at least 1
        inds_obs = np.random.choice(inds, min(num_seeds, len(inds)), replace=False)  # but no more than # of instances
        inds_obs = np.concatenate((inds_obs, [min(inds_obs)]*(int(percent_seeds*n)-len(inds_obs))))
        seeds[:, j, trial] = inds_obs
        if num_seeds > largest_num_seeds:
            largest_num_seeds = num_seeds

seeds = seeds[:largest_num_seeds, :, :]

outputfn = results_dir + fn + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'
print (outputfn)

with open(outputfn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(
        ['k', 'penalty_f', 'gamma', 'rho_mult', 'penalty_param', 'avg miscls rate', 'num_trial', 'misclass rates'])

    for k in ORDER_K:
        print ('k =', k)
        B_init = None
        Dk = penalty_matrix(G, k)
        
        for penalty_f in PENALTIES:
            print (penalty_f)

            if penalty_f in ['L1']:
                B_init = None  # default initialization Y_obs + R_unobs
            else:
                B_init = l1_init  # initialization with L1 output

            opt_param, B_hat, miscls_avg, penalty_param, miscls = autotune_ssl(seeds, Y_true, Dk, R, penalty_f, max_evals,
                                                                               pspace, B_init=B_init)
            if penalty_f == 'L1':
                l1_init = B_hat.copy()

            miscls_avg = np.around(miscls_avg, decimals=4)
            writer.writerow(
                [k, penalty_f,  np.around(opt_param['gamma'], decimals=4), np.around(opt_param['rho_mult'], decimals=4),
                 penalty_param, np.around(miscls_avg, decimals=4), num_trial]
                + list(np.around(miscls, decimals=4))
            )

            print (' ')
            print ('misclassification rate averaged over ' + str(num_trial) + ' trials:', np.around(miscls_avg, decimals=4))
            print (np.around(opt_param['gamma'], decimals=4), np.around(opt_param['rho_mult'], decimals=4), penalty_param)
print ('===================================================')
