from expert import LinearExpert1d
import numpy as np
import matplotlib.pyplot as plt

T = 1000

def doppler(x, epsilon):
    return np.sin(2 * np.pi * (1 + epsilon) / (x + epsilon))

X = np.linspace(0, T, T)/T

Y = doppler(X, 0.38) + np.random.normal(0, 0.05, T)

# E = LinearExpert1d(1)
# for t in range(5):
#     E.predict(X[t], t)
#     E.update(Y[t], t)

# print(E.history_x)
# print(E.history_y)

# plt.hold = False
# #t = np.linspace(0,sim_steps,sim_steps)/sim_steps
# plt.figure(num=1, figsize=(12,6), dpi=80, facecolor='w', edgecolor='k')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.plot(Y,linewidth=4)
# plt.scatter(np.arange(5), np.array(E.predictions), 20, "red")
# plt.tick_params(labelsize=20)
# plt.title('Doppler function',fontsize=20)
# #plt.savefig('doppler.pdf',bbox_inches='tight')
# plt.show()

from flh import FLH
alg = FLH(3)
alg.step(X[0], Y[0])
alg.step(X[1], Y[1])
alg.step(X[2], Y[2])