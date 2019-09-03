# +
import numpy as np
import pandas as pd
import networkx as nx
import pygsp as pg
from scipy.interpolate import interp2d
import csv
from datetime import datetime
from hyperopt import hp

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../GTF3')
from Utilities import *


# -

# Run GTF on piecewise constant signals on a 2D-grid graph and Minnesota road network. Get SNRss.
# Figure 3 middle panel.

def runDenoisingSimulation(name, INPUT_SNRs, PENALTIES, k, Gnx, y_true, max_evals, pspace, d):
    Dk = penalty_matrix(Gnx, k)
    DTD = Dk.T.dot(Dk).toarray()
    [S, V] = np.linalg.eigh(DTD)

    n = nx.number_of_nodes(Gnx)
    y_true_norm_sq = np.linalg.norm(y_true, 2) ** 2
    # Observation = signal + random noise
    SIGMA_SQs = y_true_norm_sq / np.array([10 ** (snr / 10.0) for snr in INPUT_SNRs]) / n

    print ('INPUT_SNRs:', INPUT_SNRs)
    print ('SIGMA_SQs:', SIGMA_SQs)

    outputfn = '../datasets/' + name + '/'+name+'-simulation' + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'

    with open(outputfn, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['n', 'k', 'd'])
        writer.writerow([n, k, d])
        writer.writerow(
            ['input_snr (dB)', 'sigma_sq', 'penalty_f', 'gamma_mult', 'rho_mult', 'penalty_param',
             'avg output_snr (dB)', 'output_snr (dB)'])

        for noise_level in range(len(INPUT_SNRs)):
            sigma_sq = SIGMA_SQs[noise_level]
            print (' ')
            print ('SNR:', INPUT_SNRs[noise_level], 'sigma sq:', sigma_sq)
            print ("========================")

            # Observed vector-valued graph signal Y
            Y = np.tile(y_true.reshape(-1, 1), (1, d)) + np.random.normal(scale=np.sqrt(sigma_sq),
                                                                                   size=(n, d))

            for penalty in range(len(PENALTIES)):
                penalty_f = PENALTIES[penalty]
                print (penalty_f)

                # Vector-GTF
                if penalty_f in ['L1']:
                    B_init = None
                else:
                    B_init = l1_init_vec  # warm start with L1 GTF estimate

                opt_param, B_hat, snr_avg, penalty_param, output_snr = autotune_denoising(Y, y_true, Dk,
                                                                                          penalty_f, sigma_sq,
                                                                                          max_evals, pspace, eig=(S,V),
                                                                                          B_init=B_init, vec=True)
                if penalty_f == 'L1':
                    l1_init_vec = B_hat.copy()

                snr_avg = np.around(snr_avg, decimals=4)
                writer.writerow(
                    [INPUT_SNRs[noise_level], np.around(sigma_sq, decimals=4), penalty_f,
                     np.around(opt_param['gamma_mult'], decimals=4), np.around(opt_param['rho_mult'], decimals=4),
                     penalty_param, np.around(snr_avg, decimals=4)]
                    + list(np.around(output_snr, decimals=4))
                )

                print (' ')
                print ('vector-GTF')
                print ('Output SNR averaged over ' + str(d) + ' trials:', np.around(snr_avg, decimals=4))
                print (np.around(opt_param['gamma_mult'], decimals=4), np.around(opt_param['rho_mult'],
                                                                                decimals=4), penalty_param)
                print (np.around(output_snr, decimals=4))

                # Scalar-GTF                
                if penalty_f in ['L1']:
                    B_init = None
                else:
                    B_init = l1_init  # warm start with L1 output

                opt_param, B_hat, snr_avg, penalty_param, output_snr = autotune_denoising(Y, y_true, Dk,
                                                                                          penalty_f, sigma_sq,
                                                                                          max_evals, pspace, eig=(S,V),
                                                                                          B_init=B_init, vec=False)
                if penalty_f == 'L1':
                    l1_init = B_hat.copy()

                snr_avg = np.around(snr_avg, decimals=4)
                writer.writerow(
                    [INPUT_SNRs[noise_level], np.around(sigma_sq, decimals=4), penalty_f,
                     np.around(opt_param['gamma_mult'], decimals=4), np.around(opt_param['rho_mult'], decimals=4),
                     penalty_param, np.around(snr_avg, decimals=4)]
                    + list(np.around(output_snr, decimals=4))
                )

                print (' ')
                print ('scalar-GTF')
                print ('Output SNR averaged over ' + str(d) + ' trials:', np.around(snr_avg, decimals=4))
                print (np.around(opt_param['gamma_mult'], decimals=4), np.around(opt_param['rho_mult'],
                                                                                decimals=4), penalty_param)
                print (np.around(output_snr, decimals=4))
                print (' ')
                
    return outputfn


def saveSimulationResults(outputfn, name):
    data = pd.read_csv(outputfn, skiprows=3,
                       names=['input_snr (dB)', 'sigma_sq', 'penalty_f', 'gamma_mult', 'rho_mult', 'penalty_param',
                              'avg output_snr (dB)'] + [str(e) for e in range(d)])
    input_snr = np.array(sorted(list(set(data['input_snr (dB)'].values))))
    L1 = data.loc[data['penalty_f'] == 'L1', 'avg output_snr (dB)'].values
    SCAD = data.loc[data['penalty_f'] == 'SCAD-L1', 'avg output_snr (dB)'].values
    MCP = data.loc[data['penalty_f'] == 'MCP-L1', 'avg output_snr (dB)'].values
    
    L1_vec = L1[range(0,len(L1),2)]
    SCAD_vec = SCAD[range(0,len(SCAD),2)]
    MCP_vec = MCP[range(0,len(MCP),2)]
    
    L1 = L1[range(1,len(L1),2)]
    SCAD = SCAD[range(1,len(SCAD),2)]
    MCP = MCP[range(1,len(MCP),2)]
    
    np.savez(name +'-results'+'.npz', L1=L1, SCAD=SCAD, MCP=MCP, L1_vec=L1_vec, SCAD_vec=SCAD_vec, MCP_vec=MCP_vec,
             input_snr=input_snr)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(input_snr, L1, '*-', label='L1')
    plt.plot(input_snr, SCAD, 'o-.', label='SCAD')
    plt.plot(input_snr, MCP, '+--', label='MCP')
    
    plt.plot(input_snr, L1_vec, '*-', label='L1_vec')
    plt.plot(input_snr, SCAD_vec, 'o-.', label='SCAD_vec')
    plt.plot(input_snr, MCP_vec, '+--', label='MCP_vec')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('Output SNR (dB)')
    plt.legend()
    #fig.savefig(name+'-snr.pdf', bbox_inches='tight')
    plt.show()
    return 1


""
PENALTIES = ['L1', 'SCAD-L1', 'MCP-L1']
INPUT_SNRs = range(-10, 31, 5)
k = 0
""

""
name = '2d-grid'
n = 20
Y_HIGH = 10
Y_LOW = -5

Gnx, signal_2d, y_true, xs, ys = create2DSignal(k, n, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)

pspace = (
    hp.loguniform('gamma_mult', -4, 4),
    hp.loguniform('rho_mult', -3, 3)
)
max_evals = 40
d = 10
""

# ###########################################################
# name = 'minnesota'
# G = pg.graphs.Minnesota(connect=True)
# Gnx = nx.from_scipy_sparse_matrix(G.A.astype(float), edge_attribute='weight')
# y_true = np.load('../datasets/minnesota/beta_0.npy')

# pspace = (
#     hp.loguniform('gamma_mult', -3, 5),
#     hp.loguniform('rho_mult', -3, 3)
# )
# max_evals = 100
# d = 20
""
outputfn = runDenoisingSimulation(name, INPUT_SNRs, PENALTIES, k, Gnx, y_true, max_evals, pspace, d)
saveSimulationResults(outputfn, name)
