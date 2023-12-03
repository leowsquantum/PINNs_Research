import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 2, 100)
y = x**2

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.yscale('log')
plt.show()
