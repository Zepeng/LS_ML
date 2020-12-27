import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

# filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/data_usgcnn_total/0.npz"
# filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/data_300_usgcnn/2.npz"
filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/0.npz"
file_mesh = "/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/ugscnn/mesh_files/icosphere_5.pkl"
batch = np.load(filename)
pmtinfos = batch['pmtinfo']  # [idx % self.nevt_file]
vertices = batch['vertex']  # [idx % self.nevt_file]
edeps = batch['eqen']
types = batch['eventtype']
print("len(pmtinfos:", len(pmtinfos))
E_low = 10
E_high = 15
for i,type in enumerate(types):
    R = math.sqrt(np.sum(vertices[i]**2))
    if type==0 and edeps[i]<E_high and edeps[i] >E_low and R<10**4:
        event2dimage_intep_bkg = pmtinfos[i]
        print(f"bkg --- Equen:{edeps[i]}, vertex:{vertices[i]} ")
        break
for i,type in enumerate(types):
    R = math.sqrt(np.sum(vertices[i]**2))
    if type==1 and edeps[i]<E_high and edeps[i]>E_low and R<10**4:
        event2dimage_intep_sig = pmtinfos[i]
        print(f"sig --- Equen:{edeps[i]}, vertex:{vertices[i]} ")
        break
print(types)

p = pickle.load(open(file_mesh, "rb"))
plot_partial: bool = False
V = p['V']  # V is the matrix which presents xyz of each mesh points
F = p['F']
V = np.array(V)
print("V shape : ", V.shape)
x, y, z = V[:, 0], V[:, 1], V[:, 2]

fig_hittime = plt.figure("hittime_intep_bkg")
ax = fig_hittime.add_subplot(111, projection='3d')
# indices = (event2dimage_intep_bkg[1] != 0.)

# img_hittime = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[1][indices], cmap=plt.hot(), s=1)
img_hittime = ax.scatter(x, y, z, c=event2dimage_intep_bkg[1], cmap=plt.hot(), s=1)
ax.set_xlabel("Detector X")
ax.set_ylabel("Detector Y")
ax.set_zlabel("Detector Z")
fig_hittime.colorbar(img_hittime)

fig_eqen = plt.figure("eqen_intep_bkg")
ax = fig_eqen.add_subplot(111, projection='3d')
# indices = (event2dimage_intep[0] != 0)
# img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[0][indices], cmap=plt.hot(), s=1)
img_eqen = ax.scatter(x, y, z, c=event2dimage_intep_bkg[0], cmap=plt.hot(), s=1)
ax.set_xlabel("Detector X")
ax.set_ylabel("Detector Y")
ax.set_zlabel("Detector Z")
fig_eqen.colorbar(img_eqen)

fig_eqen = plt.figure("eqen_intep_sig")
ax = fig_eqen.add_subplot(111, projection='3d')
# indices = (event2dimage_intep[0] != 0)
# img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[0][indices], cmap=plt.hot(), s=1)
img_eqen = ax.scatter(x, y, z, c=event2dimage_intep_sig[0], cmap=plt.hot(), s=1)
ax.set_xlabel("Detector X")
ax.set_ylabel("Detector Y")
ax.set_zlabel("Detector Z")
fig_eqen.colorbar(img_eqen)

fig_eqen = plt.figure("hittime_intep_sig")
ax = fig_eqen.add_subplot(111, projection='3d')
# indices = (event2dimage_intep[0] != 0)
# img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[0][indices], cmap=plt.hot(), s=1)
img_eqen = ax.scatter(x, y, z, c=event2dimage_intep_sig[1], cmap=plt.hot(), s=1)
ax.set_xlabel("Detector X")
ax.set_ylabel("Detector Y")
ax.set_zlabel("Detector Z")
fig_eqen.colorbar(img_eqen)

plt.show()