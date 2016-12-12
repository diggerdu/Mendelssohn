import matplotlib.pyplot as plt
import numpy as np

a = np.load("train_669_0014.npy")
fig, ax = plt.subplots()
ax.set_yscale('linear')
ax.pcolormesh(a[0,...,0])
plt.show()
