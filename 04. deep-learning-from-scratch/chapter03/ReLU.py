import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)

x = np.arange(-6, 6, 1)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-1, 6)
plt.show()