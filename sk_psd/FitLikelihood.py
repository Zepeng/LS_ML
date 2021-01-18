# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
from modules import nEXOFitLikelihood
from matplotlib.colors import LogNorm
class FLH:
    def __init__(self):
        self.v_vertex = np.array([])
        self.v_equen = np.array([])
        self.v_predict = np.array([])
        self.v_R3 = np.array([])
    def LoadPrediction(self, infile:str):
        f = np.load(infile, allow_pickle=True)
        self.v_vertex = f["vertex"]
        self.v_equen = f["equen"]
        self.v_predict = f["predict_proba"]
        self.v_R3 = ( np.sqrt(np.sum(self.v_vertex ** 2, axis=1)) / 1000 ) **3
        print("check loading status")
        print(f"length -> vertex: {len(self.v_vertex)}, equen: {len(self.v_equen)}, prediction:{len(self.v_predict)}")
        print(f"content -> vertex: {self.v_vertex[:5]},\n equen: {self.v_equen[:5]},\n prediction:{self.v_predict[:5]}")
    def Get3Dhist(self):
        predict_1 = self.v_predict[:,1]
        equen = self.v_equen
        fig1, ax1 = plt.subplots()
        h2d = hl.hist((predict_1, equen), bins=50, range=((-0.01,1.01), (9, 32)))
        hl.plot2d(ax1, h2d, log=True, cbar=True, clabel="counts per bin")
        plt.xlabel("Prediction Output")
        plt.ylabel("$E_quen$")
        plt.show()


if __name__ == '__main__':
    name_file_predict = "./model_maxtime_time_job_data_dividemax/predict_0.npz"
    flh = FLH()
    flh.LoadPrediction(name_file_predict)
    flh.Get3Dhist()