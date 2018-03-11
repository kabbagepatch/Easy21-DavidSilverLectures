from MonteCarlo import MonteCarlo
from TDLambda import TDLambda
from LinFuncApprox import LinFuncApprox
import matplotlib.pyplot as plt
from plotQValues import plotQValues


'''Monte Carlo'''
mc = MonteCarlo()
q_mc, n_mc = mc.run_episodes(10000)
# plotQValues(q_mc)

'''TD Lambda'''
# td = TDLambda(0.5)
# td.run_episodes(100000)
# q_td = td.q_values()
# plotQValues(q_td)
#
# mse = []
# lmbda = []
# for i in range(0, 10):
#     lmbda.append(i / 10.)
#     td = TDLambda(i / 10.)
#     q_td, n_td = td.run_episodes(10000)
#
#     error = (q_td - q_mc) ** 2
#     mse.append(sum(sum(sum(error * 1./ (2 * 21 * 10)))))
#
# plt.plot(lmbda, mse)
#
# plt.show()

'''Linear Function Approximation'''
# lin = LinFuncApprox()
# lin.run_episodes(10000)
# q_lin = lin.q_values()
# plotQValues(q_lin)
#
mse = []
lmbda = []
for i in range(0, 10):
    lmbda.append(i / 10.)
    lin = LinFuncApprox(i / 10.)
    lin.run_episodes(10000)

    error = lin.error(q_mc)
    mse.append(error * 1./ (2 * 21 * 10))

plt.plot(lmbda, mse)

plt.show()