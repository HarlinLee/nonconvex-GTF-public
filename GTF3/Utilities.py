from functools import *
import networkx as nx
import numpy as np
import random
import scipy as sp
from admm import admm
from admm_SSL import admm_SSL
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm
from scipy import sparse


# generate piecewise-constant signal based on shortest path distances
def gen_pwc_sig(Gnx,num_seeds,path_lens):
    seeds = random.sample(range(0,Gnx.number_of_nodes()),num_seeds)
    seeds_sig_dict = dict(zip(seeds,range(num_seeds)))
    #path_lens = dict(nx.shortest_path_length(Gnx))
    get_nearest = lambda source: seeds[np.argmin(np.array([path_lens[source][target]
                                       for target in seeds]))]
    closest_seeds = np.array(list(map(get_nearest, range(Gnx.number_of_nodes()))))
    sig = [seeds_sig_dict[x] for x in closest_seeds]
    return np.array(sig)


# solve generalized Poisson eqn \Delta x = b ,sparsity of b = df
def poisson_sig(Gnx, k, df):
    if k % 2 == 0:
        r = Gnx.number_of_edges()
    elif k % 2 == 1:
        r = Gnx.number_of_nodes()
    eta = sp.sparse.random(r,1,density = float(df)/r)
    return np.squeeze(np.asarray(np.linalg.pinv(penalty_matrix(Gnx,k).todense()) * eta))


# get (weighted) incidence matrix
def incidence_matrix(Gnx):
    Delta = nx.incidence_matrix(Gnx, oriented=True, weight='weight').transpose()
    Delta_sign = sp.sparse.csr_matrix(np.sign(Delta.todense()))
    Delta_sqrt = np.sqrt(np.abs(Delta))
    return Delta_sign.multiply(Delta_sqrt)


# get generalized penalty matrix
def penalty_matrix(Gnx, k):
    if k < 0:
        raise ValueError("k must be non-negative")
    elif k == 0:
        return incidence_matrix(Gnx)
    elif k % 2 == 0:
        return incidence_matrix(Gnx) * penalty_matrix(Gnx,k-1)
    elif k % 2 == 1:
        return incidence_matrix(Gnx).transpose() * penalty_matrix(Gnx,k-1)


""
def create2DGraph(n=10, plot_flag=0):
    # Create lattice graph. Thanks networkx.
    G = nx.grid_2d_graph(n, n, periodic=False)
    if plot_flag:
        nx.draw_kamada_kawai(G)
        plt.title('Lattice Graph Visualization')
        plt.show()
    return G

def create2DSignal(k=0, n=10, Y_HIGH=10, Y_LOW=-5):
    if k == 0:
        signal_2d = np.zeros((n, n))
        signal_2d[:n//4+1, :n//4+1] = Y_HIGH
        signal_2d[3*n//4-1:, 3*n//4-1:] = Y_LOW

    elif k == 1:
        f = interp2d([0, int(n / 2), n], [0, int(3.0 / 5 * n), n], [[10, 9, 8], [-4, -5, -10], [5, 4, 3]],
                     kind='linear')
        signal_2d = f(np.arange(n), np.arange(n))

    else:
        print ("we dont have that yet!")
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

########################################################################
# Support recovery simulation
# #######################################################################


def admm_mse2(gamma, rho, penalty_param, Gnx, beta_0, y, k, penalty_f, beta_init=None):
    #rho = gamma
    Dk = penalty_matrix(Gnx, k)
    Psi = sp.sparse.csr_matrix(np.identity(Gnx.number_of_nodes()))
    beta, obj, err_path = admm(y=y, Psi=Psi, gamma=gamma, rho=rho, Dk=Dk,
                                         penalty_f=penalty_f, penalty_param=penalty_param,
                                         tol_abs=10**(-3), tol_rel=10**(-2), max_iter=500, beta_init=beta_init)
    mse = (np.linalg.norm(beta-beta_0, 2)**2)/len(beta)
    t0 = Dk*(beta_0)
    t0[abs(t0) < 10e-4] = 0
    t0supp = t0 > 0

    t = Dk*(beta)
    t[abs(t) < 10e-4] = 0
    tsupp = t > 0

    total = len(beta)
    TP = sum(t0supp & tsupp)  # true positive
    FN = sum(t0supp & (1 - tsupp))  # false negative
    FP = sum((1 - t0supp) & tsupp)  # false positive
    TN = total - TP - FN - FP

    TPR = float(TP)/(TP+FN)
    FPR = float(FP)/(FP+TN)

    l0beta = np.linalg.norm(t,0)

    return {'beta': beta, 'l0beta': l0beta, 'mse': mse, 
            'suppTPR': TPR, 'suppFPR': FPR}

########################################################################
# Denoising simulations
# #######################################################################

def admm_denoising_snr(args,  Y, sigma_sq, y_true, Dk, penalty_f, eig, B_init, vec):
    gamma_mult, rho_mult = args
    n, d = Y.shape
    S, V = eig
    gamma = sigma_sq * gamma_mult
    y_true_norm_sq = np.linalg.norm(y_true, 2) ** 2

    if penalty_f == 'L1':
        penalty_param = 0
        rho = rho_mult * gamma
    elif "SCAD" in penalty_f:
        penalty_param = 3.7
        rho = max(rho_mult * gamma, 1 / penalty_param)
    else:
        penalty_param = 1.4
        rho = max(rho_mult * gamma, 1 / penalty_param)
    
    invX = V.dot(np.diag(1/(1+rho*S))).dot(V.T)
    
    if vec:
        B, obj, err_path = admm(Y, gamma, rho, Dk, penalty_f, penalty_param,
                                             tol_abs=10 ** (-3), tol_rel=10 ** (-2), max_iter=500, B_init=B_init, invX=invX)
    else:
        B = np.zeros((n, d))
            
        for trial in range(d):
            if B_init is None:
                b_init = None
            else:
                b_init = B_init[:, trial].reshape(-1,1)
            y = Y[:, trial].reshape(-1,1)
            beta, obj, err_path = admm(y, gamma, rho, Dk, penalty_f, penalty_param,
                                             tol_abs=10 ** (-3), tol_rel=10 ** (-2), max_iter=500, B_init=b_init, invX = invX)

            B[:, trial] = beta.copy().ravel()
    
    ses = np.linalg.norm(B - np.tile(y_true.reshape(-1, 1), (1, d)), 2, axis=0) ** 2
    output_snr = (10*np.log10(y_true_norm_sq/ses))
    se = np.sum(ses)
    mse = np.mean(ses)
    mean_snr = 10*np.log10(y_true_norm_sq/mse)
    

    return {'B': B, 'loss': -mean_snr, 'output_snr': output_snr, 'penalty_param': penalty_param, 'status': STATUS_OK, 'mse': mse, 'ses': ses}

def autotune_denoising(Y, y_true, Dk, penalty_f, sigma_sq, max_evals, pspace, eig=None, B_init=None, vec=False):
    _, num_trial = Y.shape
    if eig is None:
        [S, V] = np.linalg.eigh(Dk.T.dot(Dk))
        eig = (S, V)

    admm_proxy = partial(admm_denoising_snr, Y=Y, sigma_sq=sigma_sq, eig=eig, y_true=y_true, Dk=Dk, penalty_f=penalty_f, B_init=B_init, vec=vec)

    trials = Trials()
    results = fmin(
        admm_proxy,
        space=pspace,
        algo=tpe.suggest,
        #algo=rand.suggest,
        trials=trials,
        max_evals=max_evals)
    out = admm_proxy((results['gamma_mult'], results['rho_mult']))

    return results, out['B'], -out['loss'], out['penalty_param'], out['output_snr']#, out['mse'], out['ses']

########################################################################
# Semi-supervised classification
# #######################################################################

def admm_ssl_miscls(args, seeds, Y_true, Dk, R, penalty_f, B_init):

    gamma, rho_mult = args
    
    if penalty_f == 'L1':
        penalty_param = 0
        rho = rho_mult * gamma
    elif 'SCAD' in penalty_f:
        penalty_param = 3.7
        rho = max(rho_mult * gamma, 1.0 / penalty_param)
    else:
        penalty_param = 1.4
        rho = max(rho_mult * gamma, 1.0 / penalty_param)

    n, K = Y_true.shape
    num_trial = seeds.shape[2]
    
    miscls = []
    Bs = np.zeros((n, K, num_trial))

    for trial in range(num_trial):
        mask = np.zeros((n, n))
        Y_obs = np.zeros((n, K))
        for j in range(K):
            inds_obs = seeds[:, j, trial].astype(np.int)
            mask[inds_obs, inds_obs] = 1.0
            Y_obs[inds_obs, j] = 1.0
                    
        if B_init is None:
            b_init = None
        else:
            b_init = B_init[:, :, trial]
        B, err_path = admm_SSL(mask=sparse.csr_matrix(mask), Y=Y_obs, gamma=gamma, rho=rho, Dk=Dk, penalty_f=penalty_f,
                               penalty_param=penalty_param, eps=0.01, R=R, tol_abs=10**(-3), tol_rel=10**(-2), max_iter=1000,
                               B_init=b_init)

        Y = np.argwhere(Y_true > 0)[:, 1]
        Y_hat = np.argmax(B, axis=1)
        miscl = sum(Y_hat != Y) * 1.0 / n
        miscls.append(miscl)
        Bs[:, :, trial] = B.copy()

    pbar.update()

    return {'B': Bs, 'loss': np.mean(miscls), 'miscls': miscls, 'penalty_param': penalty_param, 'status': STATUS_OK}


def autotune_ssl(seeds, Y_true, Dk, R, penalty_f, max_evals, pspace, B_init=None):
    admm_proxy = partial(admm_ssl_miscls, seeds=seeds, Y_true=Y_true, Dk=Dk, R=R, penalty_f=penalty_f, B_init=B_init)
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
    out = admm_proxy((results['gamma'], results['rho_mult']))

    return results, out['B'], out['loss'], out['penalty_param'], out['miscls']
