import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/jobs_DSNB/data/10.npz"
file_mesh = "/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/ugscnn/mesh_files/icosphere_5.pkl"
batch = np.load(filename)
pmtinfos = batch['pmtinfo']  # [idx % self.nevt_file]
vertices = batch['vertex']  # [idx % self.nevt_file]
edeps = batch['eqen']
event2dimage_intep = pmtinfos[1]
# print(pmtinfos[0])

p = pickle.load(open(file_mesh, "rb"))
plot_partial: bool = False
V = p['V']  # V is the matrix which presents xyz of each mesh points
F = p['F']
V = np.array(V)
print("V shape : ", V.shape)
x, y, z = V[:, 0], V[:, 1], V[:, 2]

fig_hittime = plt.figure("hittime_intep")
ax = fig_hittime.add_subplot(111, projection='3d')
# indices = (event2dimage_intep[1] != 0.)
# img_hittime = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[1][indices], cmap=plt.hot(), s=1)
img_hittime = ax.scatter(x, y, z, c=event2dimage_intep[1], cmap=plt.hot(), s=1)
fig_hittime.colorbar(img_hittime)

fig_eqen = plt.figure("eqen_intep")
ax = fig_eqen.add_subplot(111, projection='3d')
# indices = (event2dimage_intep[0] != 0)
# img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[0][indices], cmap=plt.hot(), s=1)
img_eqen = ax.scatter(x, y, z, c=event2dimage_intep[0], cmap=plt.hot(), s=1)
fig_eqen.colorbar(img_eqen)

plt.show()