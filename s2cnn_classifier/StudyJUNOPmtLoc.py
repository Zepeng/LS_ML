import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


def DrawPmtLoc(file_pmt: str, ax: Axes3D):
    pmtmap = np.loadtxt(file_pmt)
    plot_partial: bool = False
    # print(pmtmap)
    x = pmtmap[:, 1]
    y = pmtmap[:, 2]
    z = pmtmap[:, 3]
    print(f"n_pmt: {len(x)}")
    if plot_partial:
        indices = np.where(z > 0)
        x = x[indices]
        y = y[indices]
        z = z[indices]
    # print(f"x: {x}, y: {y}, z: {z}")
    # fig = plt.figure()
    # ax = Axes3D(fig)
    ax.scatter(x, y, z, s=1)
    R_scale = (x[0] ** 2 + y[0] ** 2 + z[0] ** 2) ** 0.5
    return R_scale


def DrawMesh(file_mesh: str, ax: Axes3D, R_scale: tuple):
    p = pickle.load(open(file_mesh, "rb"))
    plot_partial: bool = False
    V = p['V']  # V is the matrix which presents xyz of each mesh points
    F = p['F']
    V = np.array(V)
    print("V shape : ", V.shape)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    if plot_partial:
        indices = np.where(z > 0)
        x = x[indices]
        y = y[indices]
        z = z[indices]
    # print(f"x: {x}, y: {y}, z: {z}")
    R = (x[0] ** 2 + y[0] ** 2 + z[0] ** 2) ** 0.5
    factor = R_scale / R
    print(f"xyz scale : {factor}")
    x = x * factor
    y = y * factor
    z = z * factor
    # fig_mesh = plt.figure()
    # ax_mesh = Axes3D(fig_mesh)
    ax.scatter(x, y, z, s=1)


if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig)
    v_scale_factor = DrawPmtLoc(
        "/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv",
        ax)
    DrawMesh("/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/ugscnn/mesh_files/icosphere_5.pkl", ax, v_scale_factor)# icosphere_6 we got 40962 points and icosphere_5 we got 10242 points
    plt.show()
