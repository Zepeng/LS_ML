# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
# from modules import nEXOFitLikelihood
from matplotlib.colors import LogNorm
class FLH:
    def __init__(self):
        self.v_vertex = np.array([])
        self.v_equen = np.array([])
        self.v_predict = np.array([])
        self.v_R3 = np.array([])
        # self.model_tool = nEXOFitLikelihood.nEXOFitLikelihood()
    def LoadPrediction(self, infile:str):
        f = np.load(infile, allow_pickle=True)
        self.v_vertex = f["vertex"]
        self.v_equen = f["equen"]
        self.v_predict = f["predict_proba"]
        self.v_labels = f["labels"]
        self.v_R3 = ( np.sqrt(np.sum(self.v_vertex ** 2, axis=1)) / 1000 ) **3
        self.h2d = None
        print("check loading status")
        print(f"length -> vertex: {len(self.v_vertex)}, equen: {len(self.v_equen)}, prediction:{len(self.v_predict)}")
        print(f"content -> vertex: {self.v_vertex[:5]},\n equen: {self.v_equen[:5]},\n prediction:{self.v_predict[:5]}")
    def Get2Dhist(self):
        n_bins = 50
        n_to_fit_sig = 50
        n_to_fit_bkg = 1000
        range_hist = ((-0.01,1.01), (9, 32))
        predict_1 = self.v_predict[:,1]
        indices_bkg = (self.v_labels==0)
        indices_sig = (self.v_labels==1)
        equen = self.v_equen

        def SetTitle(title:str, ax:plt.Axes=None):
            if ax == None:
                plt.title(title)
                plt.xlabel("Prediction Output")
                plt.ylabel("$E_{quen}$")
            else:
                ax.set_title(title)
                ax.set_xlabel("Prediction Output")
                ax.set_ylabel("$E_{quen}$")
        fig1, ax1 = plt.subplots()
        h2d = hl.hist((predict_1, equen), bins=n_bins, range=range_hist)
        hl.plot2d(ax1, h2d, log=True, cbar=True, clabel="counts per bin")
        SetTitle("Signal + Background")

        h2d_sig = hl.hist((predict_1[indices_sig][:-n_to_fit_sig], equen[indices_sig][:-n_to_fit_sig]), bins=n_bins, range=range_hist)
        h2d_bkg = hl.hist((predict_1[indices_bkg][:-n_to_fit_bkg], equen[indices_bkg][:-n_to_fit_bkg]), bins=n_bins, range=range_hist)
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16,6))
        hl.plot2d(ax2, h2d_sig, log=True, cbar=True, clabel="counts per bin")
        SetTitle("Signal", ax2)
        hl.plot2d(ax3, h2d_bkg, log=True, cbar=True, clabel="counts per bin")
        SetTitle("Background", ax3)

        h2d_sig_to_fit = hl.hist((predict_1[indices_sig][-n_to_fit_sig:], equen[indices_sig][-n_to_fit_sig:]), bins=n_bins, range=range_hist)
        h2d_bkg_to_fit = hl.hist((predict_1[indices_bkg][-n_to_fit_bkg:], equen[indices_bkg][-n_to_fit_bkg:]), bins=n_bins, range=range_hist)
        fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16,6))
        hl.plot2d(ax4, h2d_sig_to_fit, log=True, cbar=True, clabel="counts per bin")
        SetTitle("Signal", ax4)
        hl.plot2d(ax5, h2d_bkg_to_fit, log=True, cbar=True, clabel="counts per bin")
        SetTitle("Background", ax5)


        self.h2d = h2d
        self.h2d_bkg = h2d_bkg
        self.h2d_sig = h2d_sig
        self.h2d_to_fit = h2d_sig_to_fit + h2d_bkg_to_fit
        fig4, ax6 = plt.subplots()
        hl.plot2d(ax6,self.h2d_to_fit, log=True, cbar=True, clabel="counts per bin")

    # def FitHist(self):
    #     self.model_tool.AddDataset(self.h2d)


if __name__ == '__main__':
    name_file_predict = "./model_maxtime_time_job_data_dividemax/predict_0.npz"
    flh = FLH()
    flh.LoadPrediction(name_file_predict)
    flh.Get2Dhist()
    print(flh.h2d)
    # flh.FitHist()
    # plt.show()

