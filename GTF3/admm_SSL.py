import numpy as np

from GroupProximalOperator import L1ProximalOperator, SCADProximalOperator, MCPProximalOperator
from Penalty import L1Penalty, SCADPenalty, MCPPenalty
from scipy import sparse


def admm_SSL(mask, Y, gamma, rho, Dk, penalty_f, penalty_param, eps, R, tol_abs=10**(-4), tol_rel=10**(-3), max_iter=1000,
             B_init=None):
    n, K = Y.shape  # number of classes

    B = np.zeros((n, K))

    if B_init is None:
        B_init = R.copy()
        B_init[Y.nonzero()[0], :] = Y[Y.nonzero()[0], :]

    err_paths = {}
    for j in range(K):
        beta, obj, err_path = admm_with_prior(mask, Y[:, j].reshape(-1,1), gamma/2.0, rho, Dk, penalty_f, penalty_param, eps,
                                              R[:, j].reshape(-1,1), tol_abs, tol_rel, max_iter, B_init[:, j].reshape(-1,1))
        B[:, j] = beta.reshape(n)
        err_paths[j] = err_path

    return B, err_paths


def admm_with_prior(mask, y, gamma, rho, D, penalty_f, penalty_param, eps, r, tol_abs, tol_rel, max_iter, beta_init):
    """
    Semi-supervised classification on graphs.
    solves min_{Beta} 1/2||mask(y-beta)||_2^2 + gamma* \sum_ij (W_ij*f(beta_i-beta_j)) + eps/2*||r - beta||^2
    augmented Lagrangian problem:
        L_rho(beta, eta, u) = 1/2||mask(y-beta)||_2^2 + gamma* \sum_ij (W_ij*f(beta_i-beta_j)) + eps/2*||r - beta||^2
                            + rho/2* \sum_ij (beta_i-beta_j- eta_ij + u_ij)^2 - rho/2* \sum_ij u_ij^2

    y : observed signal on graph.
    mask : 1 if observed, 0 otherwise. diagonal matrix.
    W : weights of the graph
    gamma : parameter for penalty term
    rho : parameter for Lagrangian variable
    D : Dk
    penalty_f : L2, L1, Scad, MCP
    penalty_param : extra parameter needed for calculating SCAD and MCP proximal operators
    r : some prior belief, not dependent on y.
    eps : parameter for prior belief.
    """
    if penalty_f == "L1":
        prox = L1ProximalOperator()
        pen_func = L1Penalty().calculate
    elif "SCAD" in penalty_f:
        prox = SCADProximalOperator(penalty_param)
        pen_func = SCADPenalty(penalty_param).calculate
    elif "MCP" in penalty_f:
        prox = MCPProximalOperator(penalty_param)
        pen_func = MCPPenalty(penalty_param).calculate
    else:
        print ("This penalty is not supported yet.")
        raise Exception

    iter_num = 0
    conv = 0

    n = y.shape[0]  # number of signals/nodes
    if len(y.shape) < 2:
        y = y.reshape((n, 1))
    m = D.shape[0]  # number of edges
    if len(r.shape) < 2:
        r = r.reshape((n, 1))

    # Initialize beta, eta_tilde and u_tilde
    beta = beta_init.copy()
    if len(beta.shape) < 2:
        beta = beta.reshape((n, 1))
    Db = D.dot(beta)
    eta_tilde = Db.copy()
    if isinstance(eta_tilde, sparse.csr.csr_matrix):
        eta_tilde = eta_tilde.toarray()
    u_tilde = Db - eta_tilde

    # Calculate the initial objective function value
    obj = 0.5 * np.linalg.norm(y - beta, 2) ** 2
    #
    # vfunc = np.vectorize(lambda x: pen_func(x, gamma))
    # f_eta_tilde = vfunc(Db)
    # obj += gamma * np.matmul(w.T, f_eta_tilde)

    # This will contain obj, r_norm, eps_pri, s_norm, eps_dual
    err_path = [[],[],[],[],[]]
    err_path[0].append(obj)


    # Calculate this out of the loop for speed up
    I = sparse.identity(n)
    DTD = D.T.dot(D)
    MTM = mask.T.dot(mask)
    MTM_epsI_DTD_inv = np.linalg.inv(MTM.toarray() + eps * I.toarray() + rho * DTD.toarray())

    while not conv:
        ########################################
        # Update beta (filtered y)
        # beta = (M'M + eps*I + rho*D'D)^(-1)* (rho*D'*(eta_tilde - u_tilde) + M'M*y + eps*r)
       ########################################

        beta = np.matmul(MTM_epsI_DTD_inv, MTM.dot(y) + eps*r + rho*D.T.dot(eta_tilde - u_tilde))
        Db = D.dot(beta)

        ########################################
        # Update eta
        # eta_ij = prox(beta_i - beta_j + u_ij), param = gamma*W_ij/rho
        ########################################

        eta_tilde_prev = eta_tilde.copy()
        eta_tilde = prox.threshold(Db + u_tilde, gamma / rho)

        ########################################
        # Update u (scaled lagrangian variable)
        # u_ij = u_ij + beta_i - beta_j - eta_ij
        ########################################

        u_tilde += Db - eta_tilde

        ## Check the stopping criteria
        eps_pri = np.sqrt(m) * tol_abs + tol_rel * max(np.linalg.norm(Db, 2), np.linalg.norm(eta_tilde, 2))
        eps_dual = np.sqrt(n) * tol_abs + tol_rel * np.linalg.norm(rho * D.T.dot(u_tilde), 2)
        r_norm = np.linalg.norm(Db - eta_tilde, 2)
        s_norm = np.linalg.norm(rho * D.T.dot(eta_tilde - eta_tilde_prev), 2)

        if r_norm < eps_pri and s_norm < eps_dual:
            conv = 1

        err_path[1].append(r_norm)
        err_path[2].append(eps_pri)
        err_path[3].append(s_norm)
        err_path[4].append(eps_dual)

        ## Calculate the objective function 1/2||y-beta||_2^2 + gamma* \sum_ij (W_ij*f(beta_i-beta_j)) + eps/2*||r - beta||^2
        # obj = 0.5 * np.linalg.norm(y - beta,2) ** 2
        # vfunc = np.vectorize(lambda x: pen_func(x, gamma))
        # f_eta_tilde = vfunc(Db)
        # obj += gamma * np.matmul(w.T, f_eta_tilde)
        # obj += 0.5*eps*np.linalg.norm(r - beta,2) ** 2

        err_path[0].append(obj)

        iter_num += 1
        if iter_num > max_iter:
            #print 'Did not converge after', max_iter, 'iterations.'
            break

    return beta, obj, err_path







def showPlots(y_true, y, beta, rho, gamma, penalty_f, err_paths):
    # Just to declutter the test functions
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

 #   y_idx_sort = np.argsort(y_true)
    plt.plot(y_true, 'b--', label='true y')
    plt.plot(y[:,0], y[:,1],'g*', label='noisy y')
    plt.plot(beta, 'r', label='beta')
    plt.legend()
    plt.title(penalty_f + ' trend filtered y. rho: ' + str(rho) + ' gamma: ' + str(gamma))
    plt.show()

    for i in range(len(err_paths)):
        err_path = err_paths[i]

        # plt.plot(err_path[0])
        # plt.xlabel('iterations')
        # plt.ylabel(r'$\frac{1}{2}||y-\beta||_2^2 + \gamma\sum (W_{ij}f(\beta_i-\beta_j)+ \frac{\epsilon}{2}||r - beta||_2^2$')
        # plt.title(penalty_f+'. rho: '+str(rho)+' gamma: '+str(gamma))
        # plt.show()

        plt.subplot(211)
        plt.plot(err_path[1], 'k', label='r norm')
        plt.plot(err_path[2], 'k--', label='eps_pri')
        plt.title(penalty_f+'. rho: '+str(rho)+' gamma: '+str(gamma))
        plt.ylabel(r'$||r||_2$')
        plt.legend()

        plt.subplot(212)
        plt.plot(err_path[3], 'k', label='s norm')
        plt.plot(err_path[4], 'k--', label='eps_dual')
        plt.ylabel(r'$||s||_2$')
        plt.xlabel('iterations')
        plt.legend()
        plt.show()
