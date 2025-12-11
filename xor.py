import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
Х = np.random.randn(512, 2)
y = np.logical_xor(Х[:,0] > 0, Х[:,1] > 0)
y = np .where (y, 1, -1)
plt.figure(1)
plt.scatter(Х[y == 1, 0], Х[y == 1, 1], c='b', marker='x', label='1')
plt.scatter(Х[y == -1, 0], Х[y == -1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0, 3.0); plt.xlim(-3.0, 3.0)
plt.legend()
plt.title("Исходные данные")
plt.show()