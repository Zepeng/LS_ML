import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
dataset = []
x = np.linspace(0, np.pi/10, 201)
for i in range(10000):
    sigar = np.sin(x) + np.random.random_sample((201,))
    bkgar = x/np.pi*2 + np.random.random_sample((201,))
    dataset.append((sigar, 1))
    dataset.append((bkgar, 0))

np.save('fake_data.npy', dataset)
