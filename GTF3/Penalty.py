from scipy.integrate import quad

class Penalty(object):
    def __init__(self, param=None):
        self.param = param
        self.maxValues = {}

    def calculate(self, x, gamma):
        """
        calculates the magnitude of the penalty term
        """
        pass


class L1Penalty(Penalty):
    def calculate(self, x, gamma):
        """
        returns gamma*|x|
        """
        return gamma*abs(x)


class SCADPenalty(Penalty):
    def getMaxValue(self, gamma):
        a = self.param

        if (a, gamma) not in self.maxValues:
            f = lambda t: gamma * min(1, max(a - t / gamma, 0) / (a - 1))
            self.maxValues[(a, gamma)] = quad(f, 0, abs(a*gamma))[0]

        return self.maxValues[(a, gamma)]

    def calculate(self, x, gamma):
        """
        returns SCAD penalty = gamma * int_0^x[min(1, (a - t/gamma)_+/(a-1))] dt
        """
        a = self.param
        assert a > 2

        if x >= a*gamma:
            return self.getMaxValue(gamma)

        f = lambda t: gamma*min(1, max(a - t/gamma, 0)/(a-1))
        return quad(f, 0, abs(x))[0]


class MCPPenalty(Penalty):
    def getMaxValue(self, gamma):
        a = self.param
        assert a > 1

        if (a, gamma) not in self.maxValues:
            f = lambda t: gamma * max(0, 1 - t / (a * gamma))
            self.maxValues[(a, gamma)] = quad(f, 0, abs(a*gamma))[0]

        return self.maxValues[(a, gamma)]

    def calculate(self, x, gamma):
        """
        returns gamma * int_0^x[(1- t/(a*gamma))_+] dt
        """
        a = self.param
        assert a > 1

        if x >= a*gamma:
            return self.getMaxValue(gamma)

        f = lambda t: gamma*max(0, 1 - t/(a*gamma))

        return quad(f, 0, abs(x))[0]

# def testPenalty():
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     gamma = 2
#     a = 3.7
#
#     l1_prox = L1Penalty()
#     scad_prox = SCADPenalty(a)
#     mcp_prox = MCPPenalty(a)
#
#     x = range(-10, 10, 1)
#     l1_y = []
#     scad_y = []
#     mcp_y = []
#
#     for v in x:
#         l1_y.append(l1_prox.calculate(v, gamma))
#         scad_y.append(scad_prox.calculate(v, gamma))
#         mcp_y.append(mcp_prox.calculate(v, gamma))
#
#     plt.plot(x, l1_y, 'r', label='l1')
#     plt.plot(x, scad_y, 'g', label='scad')
#     plt.plot(x, mcp_y, 'b', label='mcp')
#     plt.legend()
#     plt.title('Penalty Functions')
#     plt.show()
#
# testPenalty()
