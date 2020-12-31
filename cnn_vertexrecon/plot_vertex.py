import numpy as np
import torch
import matplotlib.pyplot as plt

ar = np.load('test_score_11.npy', allow_pickle=True)
epoch = ar[7]
dists = []
z_diff = []
energy = []
for i in range(len(epoch)):
    mc = epoch[i][0] #.cpu().numpy()
    recon = epoch[i][1] #.cpu().numpy()
    z_diff.append(mc[2] - recon[2])
    dist = np.sum(np.power(recon - mc, 2))
    dist = np.power(dist, 0.5)
    dists.append(dist)
    energy.append(epoch[i][2])

np.save('vertex_bias.npy', np.array(dists))
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.array(dists), bins=np.linspace(0, 1000, 1001), histtype='step', color='blue')
res = 0
for i in range(len(n)):
    if sum(n[:i]) > sum(n)*0.6827:
        res = i
        break
print(res)
ax.axvline(x=res, color='red')
ax.set_xlim(0,1000)
ax.set_xlabel('Vertex bias (mm)')
fig.savefig('vertex_recon.pdf')

from scipy.stats import norm
(mu, sigma) = norm.fit(z_diff)
print(mu, sigma)
fig2, ax2 = plt.subplots()
n, bins, patches = ax2.hist(np.array(z_diff), bins = np.linspace(-500, 500, 1001), histtype='step', color='blue')
ax2.set_xlabel('Diff in z position (mm)')
fig2.savefig('z_diff.pdf')

H, xedges, yedges = np.histogram2d(energy, dists, bins = (np.linspace(1, 11, 11), np.linspace(0, 1000, 201)))
fig3, ax3 = plt.subplots()
for i in range(10):
    ax3.plot(np.linspace(1, 1000, 200), H[i], label='energy %d MeV' % (i+1) )
ax3.legend()
ax3.set_xlabel('Vertex bias (mm)')
fig3.savefig('vertex_bias_e.pdf')
