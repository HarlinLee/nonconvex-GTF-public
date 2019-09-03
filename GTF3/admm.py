import numpy as np
import networkx as nx

from GroupProximalOperator import L1ProximalOperator, SCADProximalOperator, MCPProximalOperator
from Penalty import L1Penalty, SCADPenalty, MCPPenalty
from scipy import sparse
from scipy.linalg import inv

def admm(Y, gamma, rho, Dk,  penalty_f, penalty_param, tol_abs=10**(-5), tol_rel=10**(-4), max_iter=1000, B_init = None, invX=None):
    """
    solves min_{beta} 1/2||Y-B||_F^2 + h(D^(k+1)*B; gamma, penalty_param)
    augmented Lagrangian problem:
        L_rho(beta, eta, u) = 1/2||Y-B||_F^2 + h(Z)
                            + rho/2* ||D^(k+1)*B- Z + U||_F^2 - rho/2* ||U||_F^2
    Y : observed signal on graph
    gamma : parameter for penalty term (lambda in paper)
    rho : Lagrangian multiplier
    Dk : kth order graph difference operator
    penalty_f : L1, SCAD, MCP
    penalty_param : extra parameter needed for calculating SCAD and MCP proximal operators

    Z  = D^(k+1)*B
    """
    if penalty_f == "L1":
        prox = L1ProximalOperator()
        pen = L1Penalty()
        pen_func = pen.calculate
    elif "SCAD" in penalty_f:
        prox = SCADProximalOperator(penalty_param)
        pen = SCADPenalty(penalty_param)
        pen_func = pen.calculate
    elif "MCP" in penalty_f:
        prox = MCPProximalOperator(penalty_param)
        pen = MCPPenalty(penalty_param)
        pen_func = pen.calculate
    else:
        print ("This penalty is not supported yet.")
        raise Exception

    iter_num = 0
    conv = 0

    n = Y.shape[0]  # number of nodes
    d = Y.shape[1] # number of features per node
    m = Dk.shape[0]  # number of edges

    # # Initialize beta, eta_tilde and u_tilde
    if B_init is None:
        B_init = Y
    B = B_init.copy()

    DB = Dk.dot(B)
    Z = DB.copy()
    U = DB - Z

    # Calculate the initial objective function value
    obj = 0.5 * np.linalg.norm(Y - B, 'fro') ** 2
    db_norms = np.linalg.norm(DB, axis=1)
    vfunc = np.vectorize(lambda x: pen_func(x, gamma))
    f_Z = vfunc(db_norms)
    obj += gamma * sum(f_Z)

    # This will contain obj, r_norm, eps_pri, s_norm, eps_dual
    err_path = [[],[],[],[],[]]
    err_path[0].append(obj)

    if invX is None:
        # Calculate this out of the loop for speed up
        DTD = Dk.T.dot(Dk)
        invX = inv(np.eye(n) + rho * DTD.toarray())

    while not conv:
        ########################################
        ## Update B (filtered y)
        ## beta = (I+ rho*Dk'Dk)^(-1)* (rho*Dk'*(z - u) + y)
        ########################################
        for j in range(d):
            B[:, j] = np.matmul(invX, rho*Dk.T.dot(Z[:, j] - U[:, j]) + Y[:, j])

        DB = Dk.dot(B)

        ########################################
        ## Update Z
        ## z = prox([Dk*B]_l + u_l), param = gamma/rho
        ########################################

        Z_prev = Z.copy()
        Z = prox.threshold(DB + U, gamma / rho)

        ########################################
        ## Update u (scaled lagrangian variable)
        ## u_l = u_l + [Dk*beta]_l- eta_l
        ########################################

        U += DB - Z

        ## Check the stopping criteria
        eps_pri = np.sqrt(m) * tol_abs + tol_rel * max(np.linalg.norm(DB, 'fro'), np.linalg.norm(Z, 'fro'))
        eps_dual = np.sqrt(n) * tol_abs + tol_rel * np.linalg.norm(rho * Dk.T.dot(U), 'fro')
        r = np.linalg.norm(DB - Z, 'fro')
        s = np.linalg.norm(rho * Dk.T.dot(Z - Z_prev), 'fro')

        if r < eps_pri and s < eps_dual:
            conv = 1

        err_path[1].append(r)
        err_path[2].append(eps_pri)
        err_path[3].append(s)
        err_path[4].append(eps_dual)

        # commented out to save time.
        ## Calculate the objective function 1/2||y-Psi*beta||_2^2 + gamma* \sum f(D^(k+1)*beta))
#         obj = 0.5 * np.linalg.norm(Y - B, 'fro') ** 2
#         db_norms = np.linalg.norm(DB, axis=1)
#         vfunc = np.vectorize(lambda x: pen_func(x, gamma))
#         f_Z = vfunc(db_norms)
#         obj += gamma * sum(f_Z)

        
        err_path[0].append(obj)

        iter_num += 1
        if iter_num > max_iter:
            break

    return B, obj, err_path







def showPlots(rho, gamma, penalty_f, err_path):
    # Just to declutter the test functions
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(err_path[0])
    plt.xlabel('iterations')
    plt.ylabel(r'$\frac{1}{2}||y-\beta||_2^2 + \gamma\sum (W_{ij}f(\beta_i-\beta_j)$')
    plt.title(penalty_f+'. rho: '+str(rho)+' gamma: '+str(gamma))
    #plt.show()

    plt.figure()
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
    #plt.show()
