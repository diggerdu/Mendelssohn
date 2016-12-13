import matplotlib.pyplot as plt
import numpy as np

a = np.load("mo1_16k.npy")
ax1 = plt.subplot(111)

ax1.set_yscale('linear')
ax1.pcolormesh(a[0:1233,...,0])
plt.show()
