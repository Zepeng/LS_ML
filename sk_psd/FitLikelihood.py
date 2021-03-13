# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
from iminuit import Minuit
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
fit_down_limit = 0
def SetTitle(title:str, ax:plt.Axes=None):
    if ax == None:
        plt.title(title)
        plt.xlabel("Prediction Output")
        plt.ylabel("$E_{quen}$")
    else:
        ax.set_title(title)
        ax.set_xlabel("Prediction Output")
        ax.set_ylabel("$E_{quen}$")
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



    def Get2DPDFHist(self, n_remain_to_fit_sig, n_remain_to_fit_bkg):
        plot_2d_pdf = False
        # n_bins = [np.array([0,0.1, 0.2,0.3, 0.5, 0.7, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.1]), np.arange(9, 32)]
        # n_bins = [np.array([ 0.5, 0.7, 0.9, 0.95, 0.96 , 0.97, 0.98, 0.985, 0.99,0.995, 1.001]), np.arange(9, 32)]
        # n_bins = [np.array([ 0.5, 0.7, 0.9, 0.95, 0.96 , 0.97, 0.98, 0.985, 0.99,0.995, 1.001]), np.linspace(9, 32, 10)]
        n_bins = [np.array([ 0.5, 0.95, 0.975,  1.001]), np.linspace(9, 32, 5)]
        # n_bins = [np.array([0,0.5, 0.7, 0.9, 0.95, 1.01]), np.arange(9,32)]
        # n_bins = 50
        self.n_bins = n_bins
        range_hist = ((-0.01,1.01), (9, 32))
        predict_1 = self.v_predict[:,1]
        self.predict_1 = predict_1
        indices_bkg = (self.v_labels==0)
        indices_sig = (self.v_labels==1)
        equen = self.v_equen

        # plot 2D hist PDF
        fig1, ax1 = plt.subplots()
        h2d = hl.hist((predict_1, equen), bins=n_bins, range=range_hist)
        hl.plot2d(ax1, h2d, log=True, cbar=True, clabel="counts per bin")
        SetTitle("Signal + Background")

        # h2d_sig = hl.hist((predict_1[indices_sig][:-n_to_fit_sig], equen[indices_sig][:-n_to_fit_sig]), bins=n_bins, range=range_hist).normalize(integrate=False)
        # h2d_bkg = hl.hist((predict_1[indices_bkg][:-n_to_fit_bkg], equen[indices_bkg][:-n_to_fit_bkg]), bins=n_bins, range=range_hist).normalize(integrate=False)
        if n_remain_to_fit_sig != 0:
            h2d_sig = hl.hist((predict_1[indices_sig][:-n_remain_to_fit_sig], equen[indices_sig][:-n_remain_to_fit_sig]), bins=n_bins).normalize(integrate=False)
        else:
            h2d_sig = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=n_bins).normalize(integrate=False)
        h2d_bkg = hl.hist((predict_1[indices_bkg][:-n_remain_to_fit_bkg], equen[indices_bkg][:-n_remain_to_fit_bkg]), bins=n_bins).normalize(integrate=False)
        if plot_2d_pdf:
            fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16,6))
            hl.plot2d(ax2, h2d_sig, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Signal", ax2)
            hl.plot2d(ax3, h2d_bkg, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Background", ax3)
            plt.show()
        self.h2d = h2d
        self.h2d_bkg = h2d_bkg
        self.h2d_sig = h2d_sig

    def GetHistToFit(self, n_to_fit_sig:int, n_to_fit_bkg:int, n_remain_to_fit_bkg:int, n_remain_to_fit_sig:int):
        plot_2d_to_fit = False
        indices_bkg = (self.v_labels==0)
        indices_sig = (self.v_labels==1)
        if n_remain_to_fit_sig != 0:
            index_begin_sig = len(self.v_equen[indices_sig][:-n_remain_to_fit_sig])
        else:
            index_begin_sig = len(self.v_equen[indices_sig])

        index_begin_bkg = len(self.v_equen[indices_bkg][:-n_remain_to_fit_bkg])
        index_to_fit_sig = np.random.randint(index_begin_sig, len(self.v_equen[indices_sig]), size=n_to_fit_sig)
        index_to_fit_bkg = np.random.randint(index_begin_bkg, len(self.v_equen[indices_bkg]), size=n_to_fit_bkg)
        # h2d_sig_to_fit = hl.hist((predict_1[indices_sig][-n_to_fit_sig:], equen[indices_sig][-n_to_fit_sig:]), bins=n_bins, range=range_hist)
        # h2d_bkg_to_fit = hl.hist((predict_1[indices_bkg][-n_to_fit_bkg:], equen[indices_bkg][-n_to_fit_bkg:]), bins=n_bins, range=range_hist)
        if n_remain_to_fit_sig!=0:
            h2d_sig_to_fit = hl.hist((self.predict_1[indices_sig][index_to_fit_sig], self.v_equen[indices_sig][index_to_fit_sig]), bins=self.n_bins)
        else:
            h2d_sig_to_fit = []
        h2d_bkg_to_fit = hl.hist((self.predict_1[indices_bkg][index_to_fit_bkg], self.v_equen[indices_bkg][index_to_fit_bkg]), bins=self.n_bins)
        if plot_2d_to_fit:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16,6))
            if n_to_fit_sig != 0:
                hl.plot2d(ax4, h2d_sig_to_fit, log=True, cbar=True, clabel="counts per bin")
                SetTitle("Signal", ax4)
            else:
                print("h2d_sig_to_fit:\t", h2d_sig_to_fit)
            hl.plot2d(ax5, h2d_bkg_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Background", ax5)

        self.h2d_sig_to_fit = h2d_sig_to_fit
        self.h2d_bkg_to_fit = h2d_bkg_to_fit
        if n_remain_to_fit_sig != 0:
            self.h2d_to_fit = h2d_sig_to_fit + h2d_bkg_to_fit
        else:
            self.h2d_to_fit =h2d_bkg_to_fit
        if plot_2d_to_fit:
            fig4, ax6 = plt.subplots()
            hl.plot2d(ax6,self.h2d_to_fit, log=True, cbar=True, clabel="counts per bin")
            plt.show()

    def LikelihoodFunc(self, v_n:np.ndarray):
        """

        Args:
            v_n: v_n[0] is for the Nevt of signal e.g DSNB,
             v_n[1] is for the Nevt of bkg e.g nu_atm

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

        n_j = v_n[0]*self.h2d_sig.values + v_n[1]*self.h2d_bkg.values
        #set pdf = 0 as 1 in order not to encounter nan in log(pdf)
        log_n_j = np.zeros(n_j.shape)
        indices = (n_j!=0)
        log_n_j[indices] = np.log(n_j[indices])

        N_exp = v_n[0] + v_n[1]

        nll = - np.sum(self.h2d_to_fit.values * log_n_j-n_j-LogFactorial(self.h2d_to_fit.values))
        # nll = - np.sum(self.h2d_to_fit.values * log_n_j - n_j )
        # +N_exp - np.sum(self.h2d_to_fit.values)*np.log(N_exp)
        # print(f"nll:{nll}")
        return nll

    def FitHistZeroFix(self, v_n_initial=np.array([500, 2000])):
        check_result = False
        check_zero_result = False
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=[(fit_down_limit, None), (0, None)], errordef=0.5)
        m.migrad()
        self.fitter = m
        print(m.values)
        if  check_zero_result and m.np_values()[0] <=10.5 and m.np_values()[0]>=9.5:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16,6))
            hl.plot2d(ax4, flh.h2d_sig_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Signal", ax4)
            hl.plot2d(ax5, flh.h2d_bkg_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Background", ax5)
            plt.show()


        # print(m.np_values())
        if check_result:
            x = np.arange(self.n_bins)
            y = np.arange(self.n_bins)
            X, Y = np.meshgrid(x, y)
            n_sig, n_bkg = m.np_values()
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            result_fit = n_sig * self.h2d_sig + n_bkg * self.h2d_bkg
            indices = (result_fit.values!=0)
            # Z_result = np.log(result_fit.values[indices])
            Z_result = result_fit.values[indices]
            ax.scatter(X[indices], Y[indices], Z_result, c=Z_result, s=5, cmap=plt.hot(), marker=1)
            ax.scatter(X, Y, self.h2d_to_fit.values,c=self.h2d_to_fit.values,  cmap='viridis', s=5, marker=2)

            # hl.plot2d(ax, result_fit, cbar=True, log=True)
        return (m.np_values(), m.fval)
    def FitHistFixSigN(self, v_n_initial=np.array([500, 2000])):
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=[(fit_down_limit, None), (0, None)],\
                                   fix=[True, False], errordef=0.5 )
        m.migrad()
        print(m.values)
        return (m.np_values(),m.fval)

    # def FitHistFixOne(self):


if __name__ == '__main__':
    name_file_predict = "./model_maxtime_time_jobs_DSNB_sk_data/predict_0.npz"
    no_fix_fit_only = False
    flh = FLH()
    flh.LoadPrediction(name_file_predict)
    n_to_fit_bkg = 800
    n_to_fit_sig = 0
    scale_to_fit_dataset = 20

    flh.Get2DPDFHist(n_remain_to_fit_bkg=n_to_fit_bkg*scale_to_fit_dataset, n_remain_to_fit_sig=n_to_fit_sig*scale_to_fit_dataset)
    # flh.LikelihoodFunc(np.array([1,100]))
    if no_fix_fit_only:
        v_fit_result = {"sig": [], "bkg": []}
        v_fit_val_nofix = []
        for i in range(500):
            num_iterations = 0
            flh.GetHistToFit(n_to_fit_bkg=n_to_fit_bkg, n_to_fit_sig=n_to_fit_sig,
                             n_remain_to_fit_sig=n_to_fit_sig * scale_to_fit_dataset,
                             n_remain_to_fit_bkg=n_to_fit_bkg * scale_to_fit_dataset)
            v_n, f_val = flh.FitHistZeroFix([np.random.randint(0, 100), np.random.randint(0, 5000)])
            while not (flh.fitter.get_fmin()['is_valid'] and flh.fitter.get_fmin()['has_accurate_covar']):
                if num_iterations > 9:
                    break
                print(f"Refitting!! {num_iterations} times")
                v_n,f_val = flh.FitHistZeroFix([np.random.randint(0,100), np.random.randint(0,5000)])
                num_iterations += 1
            v_fit_result["sig"].append(v_n[0])
            v_fit_result["bkg"].append(v_n[1])
            v_fit_val_nofix.append(f_val)

        plt.figure()
        if n_to_fit_sig != 0:
            plt.hist(v_fit_result["sig"], histtype="step", label="bkg", bins=50)
        else:
            plt.hist(v_fit_result["sig"], histtype="step", label="bkg", bins=100, range=(-1, 3))
        plt.xlabel("N of Signal")
        plt.figure()
        plt.hist(v_fit_result["bkg"], histtype="step", label="sig",bins=100)
        plt.xlabel("N of Background")
        # print("v_fit_result:\t", v_fit_result)
    else:
        n_fix_sig_num_try = 30
        n_max_sig = 30
        v_n_sig_to_fix = np.linspace(0, n_max_sig, n_fix_sig_num_try)
        v_fit_result = {"sig": [], "bkg": []}
        v2D_chi2 = [] # dimension 0 is for the times of trying , dimension 1 is for the number of fix signal
        num_iterations = 0
        for i in range(20):
            flh.GetHistToFit(n_to_fit_bkg=n_to_fit_bkg, n_to_fit_sig=n_to_fit_sig,
                             n_remain_to_fit_sig=n_to_fit_sig * scale_to_fit_dataset,
                             n_remain_to_fit_bkg=n_to_fit_bkg * scale_to_fit_dataset)
            v_n, f_val_nofix = flh.FitHistZeroFix([np.random.randint(0, 100), np.random.randint(0, 5000)])
            while not (flh.fitter.get_fmin()['is_valid'] and flh.fitter.get_fmin()['has_accurate_covar']):
                if num_iterations > 9:
                    break
                print(f"Refitting!! {num_iterations} times")
                v_n, f_val_nofix = flh.FitHistZeroFix([np.random.randint(0,100), np.random.randint(0,5000)])
                num_iterations += 1
            print("f_val_nofix:\t", f_val_nofix)
            v_chi2 = []
            for j in v_n_sig_to_fix:
                v_n_fix, f_val_fix =flh.FitHistFixSigN(v_n_initial=[j, np.random.randint(0, 5000)])
                print("f_val_fix:\t",f_val_fix )
                v_chi2.append(f_val_fix-f_val_nofix)
            v2D_chi2.append(v_chi2)
            v_fit_result["sig"].append(v_n[0])
            v_fit_result["bkg"].append(v_n[1])

        plt.figure()
        if n_to_fit_sig != 0:
            plt.hist(v_fit_result["sig"], histtype="step", label="bkg", bins=50)
        else:
            plt.hist(v_fit_result["sig"], histtype="step", label="bkg", bins=100, range=(-1, 3))

        plt.xlabel("N of Signal")
        plt.figure()
        plt.hist(v_fit_result["bkg"], histtype="step", label="sig", bins=100)
        plt.xlabel("N of Background")
        plt.figure()
        for i in range(len(v2D_chi2)):
            plt.plot(v_n_sig_to_fix,v2D_chi2[i])
        plt.plot([0, n_max_sig], [2.706, 2.706],"--", label="90% confidence")
        plt.xlabel("Number of Signal Counts")
        plt.ylabel("$L_{min}^{fix}-L_{min}^{nofix}$")
        plt.legend()
    plt.show()

### dictionary map : /junofs_500G/sk_psd_DSNB
###or map : /gpu_500G/LS_ML_CNN

