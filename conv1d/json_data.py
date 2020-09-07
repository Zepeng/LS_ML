import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
dataset = []
x = np.linspace(0, np.pi/2, 201)
for i in range(100000):
    sigar = np.sin(x) + np.random.random_sample((201,))
    bkgar = x/np.pi*2 + np.random.random_sample((201,))
    sigdict = {}
    sigdict['vector1d'] = sigar.tolist()
    sigdict['tag'] = 1
    dataset.append(sigdict)
    bkgdict = {}
    bkgdict['vector1d'] = bkgar.tolist()
    bkgdict['tag'] = 0
    dataset.append(bkgdict)

with open('data_fake.json', 'w') as data_file:
    json.dump(dataset, data_file)
