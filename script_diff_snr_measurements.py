import numpy as np
import pandas as pd
import networkx as nx
import pygsp as pg
from scipy.interpolate import interp2d
import csv
from datetime import datetime
from hyperopt import hp
from GTF.Utilities import penalty_matrix
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm
from scipy import sparse
from functools import *
from GTF.admm import admm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def admm_denoising_snr(args, Y, Gnx, Y_true, k, penalty_f, gamma, B_init, vec):
    rho_mult = args
    n, d = Y.shape

    Dk = penalty_matrix(Gnx, k)
    Y_true_norm_sq = np.linalg.norm(Y_true, 2, axis=0) ** 2

    if penalty_f == 'L2' or penalty_f == 'L1':
        penalty_param = 0
        rho = rho_mult * gamma
    elif "SCAD" in penalty_f:
        penalty_param = 3.7
        rho = max(rho_mult * gamma, 1 / penalty_param)
    else:
        penalty_param = 1.4
        rho = max(rho_mult * gamma, 1 / penalty_param)

    if vec:
        B, obj, err_path = admm(Y, gamma, rho, Dk, penalty_f, penalty_param,
                                             tol_abs=10 ** (-3), tol_rel=10 ** (-2), max_iter=500, B_init=B_init)
    else:
        B = np.zeros((n, d))
        for trial in range(d):
            if B_init is None:
                b_init = None
            else:
                b_init = B_init[:, trial].reshape(-1,1)
            y = Y[:, trial].reshape(-1,1)
            beta, obj, err_path = admm(y, gamma, rho, Dk, penalty_f, penalty_param,
                                             tol_abs=10 ** (-3), tol_rel=10 ** (-2), max_iter=500, B_init=b_init)

            B[:, trial] = beta.copy().ravel()
    
    ses = np.linalg.norm(B - Y_true, 2, axis=0) ** 2/n
    output_snr = 10*np.log10(np.divide(Y_true_norm_sq,ses))
    mean_snr = np.mean(output_snr)
    mse = np.mean(ses)
    
    pbar.update()
    return {'B': B, 'loss': -mean_snr, 'penalty_param': penalty_param, 'output_snr': output_snr, 'status': STATUS_OK, 'ses': ses}


def autotune_denoising(Gnx, Y, Y_true, k, penalty_f, gamma, max_evals, pspace, B_init=None, vec=False):
    _, num_trial = Y.shape
    admm_proxy = partial(admm_denoising_snr, Y=Y, Gnx=Gnx, gamma=gamma, Y_true=Y_true, k=k, penalty_f=penalty_f, B_init=B_init, vec=vec)
    trials = Trials()
    global pbar
    pbar = tqdm(total=max_evals, desc="HyperOpt")
    results = fmin(
        admm_proxy,
        space=pspace,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals)
    pbar.close()
    out = admm_proxy(results['rho_mult'])

    return results, out['B'], -out['loss'], out['penalty_param'], out['output_snr']


def create2DSignal(k=0, n=10, Y_HIGH=10, Y_LOW=-5):
    if k == 0:
        signal_2d = np.zeros((n, n))
        signal_2d[:n/4+1, :n/4+1] = Y_HIGH
        signal_2d[3*n/4-1:, 3*n/4-1:] = Y_LOW

    elif k == 1:
        f = interp2d([0, int(n / 2), n], [0, int(3.0 / 5 * n), n], [[10, 9, 8], [-4, -5, -10], [5, 4, 3]],
                     kind='linear')
        signal_2d = f(np.arange(n), np.arange(n))

    else:
        print "we dont have that yet!"
        return
    Gnx = create2DGraph(n, plot_flag=0)
    xs = []
    ys = []
    y_true = []

    for node in Gnx.nodes():
        x, y = node
        xs.append(x)
        ys.append(y)

        y_true.append(signal_2d[y, x])

    y_true = np.array(y_true)

    return Gnx, signal_2d, y_true, xs, ys


def create2DGraph(n=10, plot_flag=0):
    # Create lattice graph. Thanks networkx.
    G = nx.grid_2d_graph(n, n, periodic=False)
    #D = nx.incidence_matrix(G, nodelist=G.nodes(), edgelist=G.edges(), oriented=True, weight='weight').T
    if plot_flag:
        nx.draw_kamada_kawai(G)
        plt.title('Lattice Graph Visualization')
        plt.show()
    return G

def runDenoisingSimulation(name, INPUT_SNR, PENALTIES, k, Gnx, y_true, max_evals, pspace, SNAPSHOTS):
    num_nodes = nx.number_of_nodes(Gnx)
    y_true_norm_sq = np.linalg.norm(y_true, 2) ** 2
    # Observation = signal + random noise
    SIGMA_SQ = y_true_norm_sq / 10 ** (INPUT_SNR / 10.0) / num_nodes

    print 'INPUT_SNR:', INPUT_SNR
    print 'SIGMA_SQ:', SIGMA_SQ
    print SNAPSHOTS

    SCALES = np.sqrt(10**np.random.uniform(-1, 3, max(SNAPSHOTS)-1))
    SCALES = np.array([1] +list(SCALES))
    #SCALES = np.array([1,1.480768008,1.94284566,3.868883974,27.15872209,48.86767295,0.993093933,0.199217915])
    print SCALES
    scales = np.tile(SCALES.reshape(1, -1), (num_nodes, 1))
    Y_true = np.multiply(np.tile(y_true.reshape(-1, 1), (1, max(SNAPSHOTS))), scales)
    Y_noise = np.random.normal(scale=np.sqrt(SIGMA_SQ),size=(num_nodes, max(SNAPSHOTS)))
    Y_all = Y_true + Y_noise
    
    Y_true_norm_sq = np.linalg.norm(Y_true, 2, axis=0) ** 2
    Y_noise_norm_sq = np.linalg.norm(Y_noise, 2, axis=0) ** 2
    INPUT_SNRs = 10*np.log10(np.divide(Y_true_norm_sq,Y_noise_norm_sq))
    print INPUT_SNRs
    
    outputfn = 'datasets/' + name + '/'+name+'-diff-measurements' + datetime.now().strftime('-%y-%m-%d-%H-%M') + '.csv'

    with open(outputfn, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['num_nodes', 'k', 'sigma_sq', 'scales'])
        writer.writerow([num_nodes, k, SIGMA_SQ]+ list(SCALES))
        writer.writerow(['input_snrs (dB)'])
        writer.writerow(INPUT_SNRs)
        writer.writerow(
            ['snapshots', 'penalty_f', 'gamma_mult', 'rho_mult', 'penalty_param',
             'average SNR', 'SNR per snapshot'])
       

        for L in SNAPSHOTS:
            print ' '
            print 'L:', L
            print "========================"
            Y = Y_all[:, :L]

            for penalty in range(len(PENALTIES)):
                penalty_f = PENALTIES[penalty]
                print ' '
                print penalty_f

                if penalty_f in ['L1', 'SCAD', 'MCP']:
                    B_init = None
                else:
                    B_init = l1_init_mmv  # initialization with L1 output
                    
                if penalty_f == 'L2' or penalty_f == 'L1':
                    gamma_mult = 0.5
                    gamma = SIGMA_SQ * gamma_mult
                elif "SCAD" in penalty_f:
                    gamma_mult = 0.5
                    gamma = SIGMA_SQ * gamma_mult
                else:
                    gamma_mult = 0.5
                    gamma = SIGMA_SQ * gamma_mult
                    
                opt_param, B_hat, avg_snr, penalty_param, output_snr = autotune_denoising(Gnx, Y, Y_true[:, :L], k,
                                                                                          penalty_f, gamma*np.sqrt(L), 
                                                                                          max_evals, pspace,
                                                                                          B_init=B_init, vec=True)
                if penalty_f == 'L1':
                    l1_init_mmv = B_hat.copy()

                snr = np.around(avg_snr, decimals=4)
                writer.writerow(
                    [L, penalty_f,
                     np.around(gamma_mult, decimals=4), np.around(opt_param['rho_mult'], decimals=4),
                     penalty_param, snr]
                    + list(np.around(output_snr, decimals=4))
                )

                print 'vector-GTF'
                print 'SNR averaged over ' + str(L) + ' trials:', snr
                print np.around(gamma_mult, decimals=4), np.around(opt_param['rho_mult'],decimals=4), penalty_param
                print np.around(output_snr, decimals=4)
                
                if penalty_f in ['L1', 'SCAD', 'MCP']:
                    B_init = None
                else:
                    B_init = l1_init  # initialization with L1 output

                opt_param, B_hat, avg_snr, penalty_param, output_snr = autotune_denoising(Gnx, Y, Y_true[:, :L], k,
                                                                                          penalty_f, gamma, 
                                                                                          max_evals, pspace,B_init=B_init,
                                                                                          vec=False)

                if penalty_f == 'L1':
                    l1_init = B_hat.copy()

                snr = np.around(avg_snr, decimals=4)
                writer.writerow(
                    [L, penalty_f,
                     np.around(gamma_mult, decimals=4), np.around(opt_param['rho_mult'], decimals=4),
                     penalty_param, snr]
                    + list(np.around(output_snr, decimals=4))
                )

                print 'scalar-GTF'
                print 'Output SNR averaged over ' + str(L) + ' trials:', snr
                print np.around(gamma_mult, decimals=4), np.around(opt_param['rho_mult'], decimals=4), penalty_param
                print np.around(output_snr, decimals=4)                                                                               
                print ' '
                


    return outputfn


############################################################
PENALTIES = ['L1', 'SCAD-L1', 'MCP-L1'] 
INPUT_SNR = 0
SNAPSHOTS = [8]                              
k = 0
############################################################

# ############################################################
name = '2d-grid'
n = 20
Y_HIGH = 10
Y_LOW = -5

Gnx, signal_2d, y_true, xs, ys = create2DSignal(k, n, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)

pspace = (
    hp.loguniform('rho_mult', -3, 3)
)
max_evals = 100
# ############################################################

############################################################
# name = 'minnesota'
# G = pg.graphs.Minnesota(connect=True)
# Gnx = nx.from_scipy_sparse_matrix(G.A.astype(float), edge_attribute='weight')
# y_true = np.load('datasets/minnesota/beta_0.npy')

# pspace = (
#     hp.loguniform('rho_mult', -3, 3)
# )
# max_evals = 30
############################################################
outputfn = runDenoisingSimulation(name, INPUT_SNR, PENALTIES, k, Gnx, y_true, max_evals, pspace, SNAPSHOTS)

