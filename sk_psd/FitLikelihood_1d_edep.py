# -*- coding:utf-8 -*-
# @Time: 2021/3/10 15:21
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood_1d_edep.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
from iminuit import Minuit
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")


# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def SetTitle(title: str, ax: plt.Axes = None):
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

    def LoadPrediction(self, infile: str):
        f = np.load(infile, allow_pickle=True)
        self.v_vertex = f["vertex"]
        self.v_equen = f["equen"]
        self.v_predict = f["predict_proba"]
        self.v_labels = f["labels"]
        self.v_R3 = (np.sqrt(np.sum(self.v_vertex ** 2, axis=1)) / 1000) ** 3
        self.h1d = None
        print("check loading status")
        print(f"length -> vertex: {len(self.v_vertex)}, equen: {len(self.v_equen)}, prediction:{len(self.v_predict)}")
        print(f"content -> vertex: {self.v_vertex[:5]},\n equen: {self.v_equen[:5]},\n prediction:{self.v_predict[:5]}")

    def Get1DEdepPDFHist(self, n_remain_to_fit_sig, n_remain_to_fit_bkg):
        plot_1d_edep = False
        self.criteria = 0.9492999999999946
        # n_bins_edep = np.linspace(9, 32, 30)
        n_bins_edep = np.arange(9, 32)
        self.n_bins = n_bins_edep
        predict_1 = self.v_predict[:, 1]
        self.predict_1 = predict_1
        indices_bkg = (self.v_labels == 0)
        indices_sig = (self.v_labels == 1)
        predict_1_bkg = predict_1[indices_bkg]
        predict_1_sig = predict_1[indices_sig]
        self.h1d_sig = hl.hist(self.v_equen[indices_sig][:-n_remain_to_fit_sig][predict_1_sig[:-n_remain_to_fit_sig]>self.criteria], bins=n_bins_edep).normalize(
            integrate=False)
        self.h1d_bkg = hl.hist(self.v_equen[indices_bkg][:-n_remain_to_fit_bkg][predict_1_bkg[:-n_remain_to_fit_bkg]>self.criteria], bins=n_bins_edep).normalize(
            integrate=False)
        # self.h1d_sig = hl.hist(self.v_equen[indices_sig][:-n_remain_to_fit_sig], bins=n_bins_edep).normalize(
        #     integrate=False)
        # self.h1d_bkg = hl.hist(self.v_equen[indices_bkg][:-n_remain_to_fit_bkg], bins=n_bins_edep).normalize(
        #     integrate=False)
        if plot_1d_edep:
            fig3, ax2_edep = plt.subplots(1, 1, figsize=(8, 6))
            hl.plot1d(ax2_edep, self.h1d_sig, label="DSNB evt:"+str(len(self.v_equen[indices_sig][:-n_remain_to_fit_sig][predict_1_sig[:-n_remain_to_fit_sig]>self.criteria])))
            hl.plot1d(ax2_edep, self.h1d_bkg, label="atmNC evt:"+str(len(self.v_equen[indices_bkg][:-n_remain_to_fit_bkg][predict_1_bkg[:-n_remain_to_fit_bkg]>self.criteria])))
            plt.xlabel("$E_{dep}$ [ MeV ]")
            plt.ylabel("Density")
            plt.title("PDF")
            plt.legend()
            plt.show()


    def GetHistToFit(self, n_to_fit_sig: int, n_to_fit_bkg: int, n_remain_to_fit_bkg: int, n_remain_to_fit_sig: int):
        plot_1d_to_fit = False
        indices_bkg = (self.v_labels == 0)
        indices_sig = (self.v_labels == 1)
        index_begin_sig = len(self.v_equen[indices_sig][:-n_remain_to_fit_sig])
        index_begin_bkg = len(self.v_equen[indices_bkg][:-n_remain_to_fit_bkg])
        index_to_fit_sig = np.random.randint(index_begin_sig, len(self.v_equen[indices_sig]), size=n_to_fit_sig)
        index_to_fit_bkg = np.random.randint(index_begin_bkg, len(self.v_equen[indices_bkg]), size=n_to_fit_bkg)
        # h1d_sig_to_fit = hl.hist((predict_1[indices_sig][-n_to_fit_sig:], equen[indices_sig][-n_to_fit_sig:]), bins=n_bins, range=range_hist)
        # h1d_bkg_to_fit = hl.hist((predict_1[indices_bkg][-n_to_fit_bkg:], equen[indices_bkg][-n_to_fit_bkg:]), bins=n_bins, range=range_hist)
        predict_1_sig_to_fit = self.predict_1[indices_sig][index_to_fit_sig]
        predict_1_bkg_to_fit = self.predict_1[indices_bkg][index_to_fit_bkg]
        h1d_sig_to_fit = hl.hist( self.v_equen[indices_sig][index_to_fit_sig][predict_1_sig_to_fit>self.criteria], bins=self.n_bins)
        h1d_bkg_to_fit = hl.hist( self.v_equen[indices_bkg][index_to_fit_bkg][predict_1_bkg_to_fit>self.criteria], bins=self.n_bins)
        if plot_1d_to_fit:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))
            hl.plot1d(ax4, h1d_sig_to_fit)
            ax4.set_xlabel("$E_{dep}$")
            ax4.set_title("Signal")
            hl.plot1d(ax5, h1d_bkg_to_fit)
            ax5.set_xlabel("$E_{dep}$")
            ax5.set_title("Background")
            # SetTitle("Background", ax5)

        self.h1d_sig_to_fit = h1d_sig_to_fit
        self.h1d_bkg_to_fit = h1d_bkg_to_fit
        self.h1d_to_fit = h1d_sig_to_fit + h1d_bkg_to_fit
        if plot_1d_to_fit:
            fig4, ax6 = plt.subplots()
            hl.plot1d(ax6, self.h1d_to_fit, log=True, cbar=True, clabel="counts per bin")
            ax6.set_xlabel("$E_{dep}$")
            plt.show()

    def LikelihoodFunc(self, v_n: np.ndarray):
        """

        Args:
            v_n: v_n[0] is for the Nevt of signal e.g DSNB,
             v_n[1] is for the Nevt of bkg e.g nu_atm

        Returns:
            nll which are supposed to be minimized by minuit.

        """

        def LogFactorial(v_d_j: np.ndarray):
            v_to_sum = np.zeros(v_d_j.shape)
            for j in range(len(v_d_j)):
                if v_d_j[j] > 0:
                    v_to_sum[j] = np.sum(np.array([np.log(i) for i in range(1, int(v_d_j[j]) + 1)]))
            return v_to_sum

        n_j = v_n[0] * self.h1d_sig.values + v_n[1] * self.h1d_bkg.values
        # set pdf = 0 as 1 in order not to encounter nan in log(pdf)
        log_n_j = np.zeros(n_j.shape)
        indices = (n_j != 0)
        log_n_j[indices] = np.log(n_j[indices])

        N_exp = v_n[0] + v_n[1]

        nll = - np.sum(self.h1d_to_fit.values * log_n_j - n_j - LogFactorial(self.h1d_to_fit.values))
        # +N_exp - np.sum(self.h1d_to_fit.values)*np.log(N_exp)
        # print(f"nll:{nll}")
        return nll

    def FitHistZeroFix(self, v_n_initial=np.array([500, 2000])):
        check_result = False
        check_zero_result = False
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=[(0, None), (0, None)], errordef=0.5)
        m.migrad()
        self.fitter = m
        print(m.values)
        if check_zero_result and m.np_values()[0] <= 10.5 and m.np_values()[0] >= 9.5:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))
            hl.plot2d(ax4, flh.h1d_sig_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Signal", ax4)
            hl.plot2d(ax5, flh.h1d_bkg_to_fit, log=True, cbar=True, clabel="counts per bin")
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
            result_fit = n_sig * self.h1d_sig + n_bkg * self.h1d_bkg
            indices = (result_fit.values != 0)
            # Z_result = np.log(result_fit.values[indices])
            Z_result = result_fit.values[indices]
            ax.scatter(X[indices], Y[indices], Z_result, c=Z_result, s=5, cmap=plt.hot(), marker=1)
            ax.scatter(X, Y, self.h1d_to_fit.values, c=self.h1d_to_fit.values, cmap='viridis', s=5, marker=2)

            # hl.plot2d(ax, result_fit, cbar=True, log=True)
        return (m.np_values(), m.fval)

    def FitHistFixSigN(self, v_n_initial=np.array([500, 2000])):
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=[(0, None), (0, None)], \
                                   fix=[True, False], errordef=0.5)
        m.migrad()
        print(m.values)
        return (m.np_values(), m.fval)

    # def FitHistFixOne(self):


if __name__ == '__main__':
    name_file_predict = "./model_maxtime_time_jobs_DSNB_sk_data/predict_0.npz"
    no_fix_fit_only = True
    flh = FLH()
    flh.LoadPrediction(name_file_predict)
    n_to_fit_bkg = 800
    n_to_fit_sig = 10
    scale_to_fit_dataset = 20

    flh.Get1DEdepPDFHist(n_remain_to_fit_bkg=n_to_fit_bkg * scale_to_fit_dataset,
                     n_remain_to_fit_sig=n_to_fit_sig * scale_to_fit_dataset)
    # flh.GetHistToFit(n_to_fit_bkg=n_to_fit_bkg, n_to_fit_sig=n_to_fit_sig,
    #                          n_remain_to_fit_sig=n_to_fit_sig * scale_to_fit_dataset,
    #                          n_remain_to_fit_bkg=n_to_fit_bkg * scale_to_fit_dataset)
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
                v_n, f_val = flh.FitHistZeroFix([np.random.randint(0, 100), np.random.randint(0, 5000)])
                num_iterations += 1
            v_fit_result["sig"].append(v_n[0])
            v_fit_result["bkg"].append(v_n[1])
            v_fit_val_nofix.append(f_val)

        plt.figure()
        plt.hist(v_fit_result["sig"], histtype="step", label="bkg", bins=100)
        plt.xlabel("N of Signal")
        plt.figure()
        plt.hist(v_fit_result["bkg"], histtype="step", label="sig", bins=100)
        plt.xlabel("N of Background")
    else:
        n_fix_sig_num_try = 30
        n_max_sig = 30
        v_n_sig_to_fix = np.linspace(0, n_max_sig, n_fix_sig_num_try)
        v_fit_result = {"sig": [], "bkg": []}
        v2D_chi2 = []  # dimension 0 is for the times of trying , dimension 1 is for the number of fix signal
        num_iterations = 0
        for i in range(10):
            flh.GetHistToFit(n_to_fit_bkg=n_to_fit_bkg, n_to_fit_sig=n_to_fit_sig,
                             n_remain_to_fit_sig=n_to_fit_sig * scale_to_fit_dataset,
                             n_remain_to_fit_bkg=n_to_fit_bkg * scale_to_fit_dataset)
            v_n, f_val_nofix = flh.FitHistZeroFix([np.random.randint(0, 100), np.random.randint(0, 5000)])
            while not (flh.fitter.get_fmin()['is_valid'] and flh.fitter.get_fmin()['has_accurate_covar']):
                if num_iterations > 9:
                    break
                v_n, f_val_nofix = flh.FitHistZeroFix([np.random.randint(0, 100), np.random.randint(0, 5000)])
                num_iterations += 1
            print("f_val_nofix:\t", f_val_nofix)
            v_chi2 = []
            for j in range(n_fix_sig_num_try):
                v_n_fix, f_val_fix = flh.FitHistFixSigN(v_n_initial=[j, np.random.randint(0, 5000)])
                print("f_val_fix:\t", f_val_fix)
                v_chi2.append(f_val_fix - f_val_nofix)
            v2D_chi2.append(v_chi2)
            v_fit_result["sig"].append(v_n[0])
            v_fit_result["bkg"].append(v_n[1])

        plt.figure()
        plt.hist(v_fit_result["sig"], histtype="step", label="bkg", bins=100)
        plt.xlabel("N of Signal")
        plt.figure()
        plt.hist(v_fit_result["bkg"], histtype="step", label="sig", bins=100)
        plt.xlabel("N of Background")
        plt.figure()
        for i in range(len(v2D_chi2)):
            plt.plot(v_n_sig_to_fix[1:], v2D_chi2[i][1:])
        plt.plot([0, n_max_sig], [2.706, 2.706], "--", label="90% confidence")
        plt.xlabel("Number of Signal Counts")
        plt.ylabel("$L_{min}^{fix}-L_{min}^{nofix}$")
        plt.legend()
    plt.show()

### dictionary map : /junofs_500G/sk_psd_DSNB
###or map : /gpu_500G/LS_ML_CNN

