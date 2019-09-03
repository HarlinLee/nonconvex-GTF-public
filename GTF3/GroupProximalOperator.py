import numpy as np

'''
Now we assume V is a matrix. Apply group proximal operator row-wise. 
https://arxiv.org/pdf/1204.6491.pdf
'''
class GroupProximalOperator(object):
    def __init__(self, param=None):
        self.param = param

    def threshold(self, V, gamma):
        """
        returns the answer to prox_{gamma, f}(v) = argmin_x (f(x) + 1/2gamma ||x-v||_2^2)
        """
        pass

    def sign(self, x):
        result = x.copy()
        result[x > 0] = 1.0
        result[x < 0] = -1.0
        return result

    def soft_threshold(self, V, gamma):
        v_norms = np.linalg.norm(V, axis=1) # row-wise norm
        shrink = np.zeros(v_norms.shape)
        shrink[v_norms>0] = np.maximum(1 - gamma/v_norms[v_norms>0], np.zeros(v_norms[v_norms>0].shape))
        return np.dot(np.diag(shrink), V)


class L1ProximalOperator(GroupProximalOperator):
    def threshold(self, V, gamma):
        """
        returns the answer to prox_{gamma}(v) = argmin_x (||x||_1 + 1/2gamma ||x-v||_2^2)
        This is just soft thresholding with parameter gamma.
        """
        return self.soft_threshold(V, gamma)


class SCADProximalOperator(GroupProximalOperator):
    def threshold(self, V, gamma):
        """
        returns the answer to prox_{gamma, a}(v) = argmin_x (scad(x, param) + 1/2gamma ||x-v||_2^2)
        """
        a = float(self.param)
        assert a > 2
        
        result = V.copy()
        v_norms = np.linalg.norm(V, axis=1) # row-wise norm
        
        result[v_norms <= 2*gamma, :] = self.soft_threshold(V[v_norms <= 2*gamma, :], gamma)
        result[(v_norms < a*gamma)*(v_norms > 2*gamma)] = \
            self.soft_threshold(V[(v_norms < a*gamma)*(v_norms > 2*gamma)], a*gamma/(a-1))/(1-1/(a-1))

        return result


class MCPProximalOperator(GroupProximalOperator):
    def threshold(self, V, gamma):
        """
        returns the answer to prox_{gamma, a}(v) = argmin_x (mcp(x, param) + 1/2gamma ||x-v||_2^2)
        """
        a = float(self.param)
        assert a > 1

        result = V.copy()
        v_norms = np.linalg.norm(V, axis=1) # row-wise norm
        
        result[v_norms <= a*gamma, :] = a/(a-1)*self.soft_threshold(V[v_norms <= a*gamma, :], gamma)
        
        return result


def testProximalOperator():
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    gamma = 1
    a = 3.7

    l1_prox = L1ProximalOperator()
    scad_prox = SCADProximalOperator(a)
    mcp_prox = MCPProximalOperator(a)

    v = 1.0*np.arange(-10, 10, 1).reshape((-1,1))

    l1_y = l1_prox.threshold(v, gamma)
    scad_y = scad_prox.threshold(v, gamma)
    mcp_y = mcp_prox.threshold(v, gamma)

    plt.plot(v, v, 'k--', label='y=x')
    plt.plot(v, l1_y, 'r', label='l1')
    plt.plot(v, scad_y, 'g', label='scad')

    plt.plot(v, mcp_y, 'b', label='mcp')
    plt.legend()
    plt.title('Proximal Operators')
    plt.show()

def testGroupProximalOperator():
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    gamma = 2
    a = 3.7

    l1_prox = L1ProximalOperator()
    scad_prox = SCADProximalOperator(a)
    mcp_prox = MCPProximalOperator(a)

    v = 1.0*np.arange(-3, 3, 1).reshape((1,-1))

    l1_y = l1_prox.threshold(v, gamma).T
    scad_y = scad_prox.threshold(v, gamma).T
    mcp_y = mcp_prox.threshold(v, gamma).T
    
    v = v.T
    plt.plot(v, v, 'k--', label='y=x')
    plt.plot(v, l1_y, 'r', label='l1')
    plt.plot(v, scad_y, 'g', label='scad')

    plt.plot(v, mcp_y, 'b', label='mcp')
    plt.legend()
    plt.title('Group Proximal Operators: one row')
    plt.show()

# testProximalOperator()
# testGroupProximalOperator()
