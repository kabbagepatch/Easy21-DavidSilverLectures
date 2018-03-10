from MonteCarlo import MonteCarlo
from TDLambda import TDLambda
import numpy as np
import matplotlib.pyplot as plt
from plotQValues import plotQValues

mc = MonteCarlo()
q_mc, n_mc = mc.run_episodes(100000)

mse = []
lmbda = []
for i in range(0, 10):
    lmbda.append(i / 10.)
    td = TDLambda(i / 10.)
    q_td, n_td = td.run_episodes(10000)

    error = (q_td - q_mc) ** 2
    mse.append(sum(sum(sum(error))))

plt.plot(lmbda, mse)

plt.show()

plotQValues(q_mc)
