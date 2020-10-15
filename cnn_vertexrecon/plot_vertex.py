import numpy as np
import torch
import matplotlib.pyplot as plt

ar = np.load('test_score_1.npy', allow_pickle=True)
epoch = ar[1]
dists = []
for i in range(len(epoch)):
    mc = epoch[i][0].cpu().numpy()
    recon = epoch[i][1].cpu().numpy()
    dist = np.sum(np.power(recon - mc, 2))
    dist = np.power(dist, 0.5)
    dists.append(dist)

np.save('vertex_bias.npy', np.array(dists))
#plt.hist(np.array(dists), bins=np.linspace(0, 1000, 20), histtype='step')
#plt.xlabel('Vertex bias (mm)')
#plt.savefig('vertex_recon.pdf')
