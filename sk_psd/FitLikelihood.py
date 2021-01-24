# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
from iminuit import Minuit
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
        self.v_labels = f["labels"]
        self.v_R3 = ( np.sqrt(np.sum(self.v_vertex ** 2, axis=1)) / 1000 ) **3
        self.h2d = None
        print("check loading status")
        print(f"length -> vertex: {len(self.v_vertex)}, equen: {len(self.v_equen)}, prediction:{len(self.v_predict)}")
        print(f"content -> vertex: {self.v_vertex[:5]},\n equen: {self.v_equen[:5]},\n prediction:{self.v_predict[:5]}")
    def Get2Dhist(self):
        n_bins = 50
        n_to_fit_sig = 200
        n_to_fit_bkg = 3000
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

        h2d_sig = hl.hist((predict_1[indices_sig][:-n_to_fit_sig], equen[indices_sig][:-n_to_fit_sig]), bins=n_bins, range=range_hist).normalize()
        h2d_bkg = hl.hist((predict_1[indices_bkg][:-n_to_fit_bkg], equen[indices_bkg][:-n_to_fit_bkg]), bins=n_bins, range=range_hist).normalize()
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

    def LikelihoodFunc(self, v_n:np.ndarray):
        """

        Args:
            v_n: v_n[0] is for the Nevt of signal e.g DSNB, v_n[1] is for the Nevt of bkg e.g nu_atm

        Returns:
            nll which are supposed to be minimized by minuit.

        """
        def LogFactorial(v_d_j:np.ndarray):
            v_to_sum = np.zeros(v_d_j.shape)
            for j in range(len(v_d_j)):
                for k in range(len(v_d_j[j])):
                    if v_d_j[j][k]>0:
                        v_to_sum[j][k] = np.sum(np.array([np.log(i) for i in range(1,int(v_d_j[j][k])+1)]))
            return v_to_sum
        try_a = np.array([[0,1,2], [3, 4, 5]])

        n_j = v_n[0]*self.h2d_sig.values + v_n[1]*self.h2d_bkg.values
        #set pdf = 0 as 1 in order not to encounter nan in log(pdf)
        log_n_j = np.zeros(n_j.shape)
        indices = (n_j!=0)
        log_n_j[indices] = np.log(n_j[indices])

        nll = - np.sum(self.h2d_to_fit.values * log_n_j-n_j-LogFactorial(self.h2d_to_fit.values))
        # print(f"nll:{nll}")
        return nll

    def FitHist(self):
        v_n_initial = np.array([3, 500])
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=[(0, None), (0, None)])
        m.migrad()
        print(m.values)



if __name__ == '__main__':
    name_file_predict = "./model_maxtime_time_job_data_dividemax/predict_0.npz"
    flh = FLH()
    flh.LoadPrediction(name_file_predict)
    flh.Get2Dhist()
    # flh.LikelihoodFunc(np.array([1,100]))
    flh.FitHist()
    # plt.show()

