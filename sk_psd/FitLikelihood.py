# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
from iminuit import Minuit
from collections import Counter
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
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
        self.check_result = False
        self.check_result_threshold = 50
        self.v_vertex = np.array([])
        self.v_equen = np.array([])
        self.v_predict = np.array([])
        self.v_R3 = np.array([])
        self.dir_other_bkg = {}
        self.dir_other_bkg_sys_uncertainty = {}
        self.dir_h2d_other_bkg = {}
        self.dir_h2d_other_bkg_full = {}
        self.dir_h2d_other_bkg_to_fit = {}
        self.n_h2d_need_to_fit = 0
        self.seed = 0
        self.v_nll = []
        self.plot_v_nll = False
    def SetNOtherBkg(self, dir_n_other_bkg:dict):
        self.dir_n_other_bkg = dir_n_other_bkg
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
    def GetBkgRatio(self):
        # bkg_criteria = self.n_bins[0][0]
        bkg_criteria = 0.5
        self.ratio_bkg_NC =  Counter(self.v_predict[self.v_labels==0][:,1]>bkg_criteria)[True]/len(self.v_predict[self.v_labels==0])
        self.ratio_sig =  Counter(self.v_predict[self.v_labels==1][:,1]>bkg_criteria)[True]/len(self.v_predict)
        print("ratio:\t", self.ratio_bkg_NC)
        print(Counter(self.v_predict[:,1]>bkg_criteria))
        exit()
    def LoadOtherBkg(self, name_file:str, key:str, key_in_dict:str, sys_uncertrainty:float):
        f = np.load(name_file, allow_pickle=True)
        samples = f[key].item()
        self.dir_other_bkg[key_in_dict] = samples
        self.dir_other_bkg_sys_uncertainty[key_in_dict] = sys_uncertrainty
        # if key_in_dict == "FastN":
        #     self.dir_other_bkg[key_in_dict]["equen"] = np.random.uniform(10, 31, size=len(self.dir_other_bkg[key_in_dict]["equen"]))


    def Get2DPDFHist(self, fit_2d):
        plot_2d_pdf = False
        plot_pdf_projection = False
        # n_bins = [np.array([ 0.5, 0.7, 0.9, 0.95, 0.96 , 0.97, 0.98, 0.985, 0.99,0.995, 1.001]), np.arange(9, 32)]
        # n_bins = [np.array([ 0.5, 0.7, 0.9, 0.95, 0.96 , 0.97, 0.98, 0.985, 0.99,0.995, 1.001]), np.linspace(9, 32, 10)]
        if fit_2d:
            n_bins = [np.array([ 0.5, 0.95, 0.975,  1.001]), np.linspace(9, 32, 8)]
            # n_bins = [np.array([0,0.1, 0.2,0.3, 0.5, 0.7, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.001]), np.arange(9, 32)]
        else:
            n_bins = [np.array([ 0.95, 1.001]), np.linspace(9, 32, 8)]
        if  n_bins[0][0]!=0.:
            self.n_bins_full = [np.concatenate((np.array([0]), n_bins[0])), n_bins[1]]
        else:
            self.n_bins_full = n_bins

        # n_bins = 50
        self.n_bins = n_bins
        # self.GetBkgRatio()
        # range_hist = ((-0.01,1.01), (9, 32))
        predict_1 = self.v_predict[:,1]
        self.predict_1 = predict_1
        indices_bkg_NC = (self.v_labels==0)
        indices_sig = (self.v_labels==1)
        equen = self.v_equen

        self.h2d_sig = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=n_bins).normalize(integrate=False)
        self.h2d_bkg_NC = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=n_bins).normalize(integrate=False)
        self.h2d_sig_full = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=self.n_bins_full).normalize(integrate=False)
        self.h2d_bkg_NC_full = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins_full).normalize(integrate=False)
        for key in self.dir_other_bkg.keys():
            self.dir_h2d_other_bkg[key] = hl.hist((self.dir_other_bkg[key]["prod"], self.dir_other_bkg[key]["equen"]), bins=n_bins).normalize(integrate=False)
            self.dir_h2d_other_bkg_full[key] = hl.hist((self.dir_other_bkg[key]["prod"], self.dir_other_bkg[key]["equen"]), bins=self.n_bins_full).normalize(integrate=False)
        self.n_h2d_need_to_fit = len(self.dir_h2d_other_bkg.keys())+2
        if plot_2d_pdf:
            # plot 2D hist PDF
            fig1, ax1 = plt.subplots()
            h2d = hl.hist((predict_1, equen), bins=n_bins)
            hl.plot2d(ax1, h2d, log=True, cbar=True, clabel="counts per bin")
            SetTitle("PDF(Signal + NC)")

            fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16,6))
            hl.plot2d(ax2, self.h2d_sig, log=True, cbar=True, clabel="counts per bin")
            SetTitle("PDF(Signal)", ax2)
            hl.plot2d(ax3, self.h2d_bkg_NC, log=True, cbar=True, clabel="counts per bin")
            SetTitle("PDF(NC)", ax3)

            n_other_bkg =len(self.dir_h2d_other_bkg_full.keys())
            for i, key in enumerate(self.dir_other_bkg.keys()):
                fig_other_bkg, ax_other_bkg = plt.subplots()
                hl.plot2d(ax_other_bkg, self.dir_h2d_other_bkg_full[key],log=True, cbar=True, clabel="counts per bin")
                SetTitle(f"PDF({key})", ax_other_bkg)
            plt.show()
        if plot_pdf_projection:
            fig_profile_equen, ax_profile_equen = plt.subplots()
            hl.plot1d(ax_profile_equen, self.h2d_sig_full.project([1]), label="DSNB")
            hl.plot1d(ax_profile_equen, self.h2d_bkg_NC.project([1]), label="atm-NC")
            for key in self.dir_h2d_other_bkg_full.keys():
                hl.plot1d(ax_profile_equen, self.dir_h2d_other_bkg_full[key].project([1]), label=key)
            ax_profile_equen.set_xlabel("$E_{quen}$ ")
            plt.title("Projection of $E_{quen}$")
            plt.legend()

            fig_profile_PSD, ax_profile_PSD_full = plt.subplots()
            hl.plot1d(ax_profile_PSD_full, self.h2d_sig_full.project([0]), label="DSNB")
            hl.plot1d(ax_profile_PSD_full, self.h2d_bkg_NC_full.project([0]), label="atm-NC")
            for key in self.dir_h2d_other_bkg_full.keys():
                hl.plot1d(ax_profile_PSD_full, self.dir_h2d_other_bkg_full[key].project([0]), label=key)
            ax_profile_PSD_full.set_xlabel("PSD Output")
            plt.title("Projection of PSD Output(Full PDF)")
            plt.legend()

            fig_profile_PSD, ax_profile_PSD = plt.subplots()
            hl.plot1d(ax_profile_PSD, self.h2d_sig.project([0]), label="DSNB")
            hl.plot1d(ax_profile_PSD, self.h2d_bkg_NC.project([0]), label="atm-NC")
            for key in self.dir_h2d_other_bkg_full.keys():
                hl.plot1d(ax_profile_PSD, self.dir_h2d_other_bkg[key].project([0]), label=key)
            ax_profile_PSD.set_xlabel("PSD Output")
            plt.title("Projection of PSD Output")
            plt.legend()
            plt.show()

    def PlotHistToFit(self):
        fig1, ax1 = plt.subplots()
        hl.plot2d(ax1, self.h2d_to_fit, cbar=True, clabel="counts per bin")
        SetTitle("PDF(Signal + NC)")

        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16,6))
        hl.plot2d(ax2, self.h2d_sig_to_fit, cbar=True, clabel="counts per bin")
        SetTitle("PDF(Signal)", ax2)
        hl.plot2d(ax3, self.h2d_bkg_NC_to_fit, cbar=True, clabel="counts per bin")
        SetTitle("PDF(NC)", ax3)

        n_other_bkg =len(self.dir_h2d_other_bkg_to_fit.keys())
        for i, key in enumerate(self.dir_other_bkg.keys()):
            fig_other_bkg, ax_other_bkg = plt.subplots()
            hl.plot2d(ax_other_bkg, self.dir_h2d_other_bkg_to_fit[key], cbar=True, clabel="counts per bin")
            SetTitle(f"PDF({key})", ax_other_bkg)
        plt.show()

    def GetHistToFit(self, n_to_fit_sig:int, n_to_fit_bkg_NC:int, seed:int, print_fit_result:bool):
        self.print_fit_result = print_fit_result
        # plot_2d_to_fit = True
        # plot_2d_to_fit_full_bins = True
        self.seed = seed
        plot_2d_to_fit = False
        plot_2d_to_fit_full_bins = False
        from matplotlib.colors import LogNorm

        from collections import Counter
        #n_sig_samples = np.random.poisson(n_to_fit_sig)
        #n_bkg_samples = np.random.poisson(n_to_fit_bkg_NC)
        n_sig_samples = n_to_fit_sig
        n_bkg_samples = n_to_fit_bkg_NC
        if print_fit_result:
            print("input DSNB:\t", n_sig_samples)
            print("input atm-NC:\t", n_bkg_samples)
        sig_sample = self.h2d_sig_full.sample( n_sig_samples,seed=seed)
        bkg_NC_sample = self.h2d_bkg_NC_full.sample(n_bkg_samples, seed=seed)
        dir_other_bkg_samples = {}
        for key in self.dir_h2d_other_bkg_full.keys():
            # print(key+":\t", self.dir_n_other_bkg[key])
            #n_samples = np.random.poisson(self.dir_n_other_bkg[key])
            n_samples = int(self.dir_n_other_bkg[key])
            if print_fit_result:
               print("input "+key+":\t", n_samples)
            dir_other_bkg_samples[key] = self.dir_h2d_other_bkg_full[key].sample( n_samples,seed=seed)
            self.dir_h2d_other_bkg_to_fit[key] = hl.hist((dir_other_bkg_samples[key][0], dir_other_bkg_samples[key][1]), bins=self.n_bins)
        # sig_sample = self.h2d_sig_full.sample(int(n_to_fit_sig), seed=seed)
        # bkg_NC_sample = self.h2d_bkg_NC_full.sample(int(n_to_fit_bkg_NC), seed=seed)
        h2d_sig_to_fit = hl.hist((sig_sample[0], sig_sample[1]), bins=self.n_bins)
        h2d_bkg_NC_to_fit = hl.hist((bkg_NC_sample[0], bkg_NC_sample[1]), bins=self.n_bins)
        self.input_sig_n = np.sum(h2d_sig_to_fit.values)
        self.input_bkg_NC_n = np.sum(h2d_bkg_NC_to_fit.values)
        if plot_2d_to_fit:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16,6))
            if n_to_fit_sig != 0:
                hl.plot2d(ax4, h2d_sig_to_fit, log=True, cbar=True, clabel="counts per bin")
                SetTitle("Signal", ax4)
            else:
                print("h2d_sig_to_fit:\t", h2d_sig_to_fit)
            hl.plot2d(ax5, h2d_bkg_NC_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("NC", ax5)
        if plot_2d_to_fit_full_bins:
            plt.figure()
            # plt.hist2d(bkg_NC_sample[0], bkg_NC_sample[1], bins=self.n_bins_full, norm=LogNorm())
            plt.hist2d(bkg_NC_sample[0], bkg_NC_sample[1], bins=self.n_bins_full)
            plt.title("NC with full bins")
            plt.colorbar()
            for key in self.dir_h2d_other_bkg_full.keys():
                plt.figure()
                plt.hist2d(dir_other_bkg_samples[key][0], dir_other_bkg_samples[key][1], bins=self.n_bins_full)
                # norm=LogNorm())
                plt.title(f"{key} with full bins")
                plt.colorbar()
            plt.show()

        self.h2d_sig_to_fit = h2d_sig_to_fit
        self.h2d_bkg_NC_to_fit = h2d_bkg_NC_to_fit
        self.h2d_to_fit = h2d_sig_to_fit + h2d_bkg_NC_to_fit
        for key in self.dir_h2d_other_bkg_to_fit.keys():
            self.h2d_to_fit = self.h2d_to_fit + self.dir_h2d_other_bkg_to_fit[key]
        # print(np.sum(h2d_sig_to_fit.values), np.sum(h2d_bkg_NC_to_fit.values), np.sum(self.h2d_to_fit.values))
        if plot_2d_to_fit:
            fig4, ax6 = plt.subplots()
            hl.plot2d(ax6,self.h2d_to_fit, log=True, cbar=True, clabel="counts per bin")
            plt.show()

    def LikelihoodFunc(self, v_n:np.ndarray):
        """

        Args:
            v_n: v_n[0] is for the Nevt of signal e.g DSNB,
             v_n[1] is for the Nevt of bkg e.g nu_atm
             v_n[2:self.n_h2d_to_fit] is for the other bkg like FastN
             v_n[self.n_h2d_to_fit:] is for epsilon of NC, ....other bkg(attention: not including DSNB)

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

        # Attention: v_n[self.n_h2d_need_to_fit:] is for the epsilon of NC, .. other bkg
        v_n_epsilon = v_n[self.n_h2d_need_to_fit:]
        n_j = v_n[0]*self.h2d_sig.values + (1+v_n_epsilon[0])*v_n[1]*self.h2d_bkg_NC.values
        N_exp = v_n[0] + (1+v_n_epsilon[0])*v_n[1]
        for i, key in enumerate(self.dir_h2d_other_bkg.keys()):
            n_j = n_j + (1+v_n_epsilon[i+1])*v_n[i+2]*self.dir_h2d_other_bkg[key].values
            N_exp = N_exp + (1+v_n_epsilon[i+1])*v_n[i+2]

        #set pdf = 0 as 1 in order not to encounter nan in log(pdf)
        log_n_j = np.zeros(n_j.shape)
        indices = (n_j>0)
        log_n_j[indices] = np.log(n_j[indices])

        nll = - 2. * np.sum(self.h2d_to_fit.values * log_n_j-n_j-LogFactorial(self.h2d_to_fit.values))
        #+ (N_exp - np.sum(self.h2d_to_fit.values)*np.log(N_exp))*2
        """
        Add the constraint of fast neutron
        """
        nll += (v_n[5] -0.4)**2/0.16
        # nll += (v_n[1] - 16)**2/160000
        # nll += (v_n[2] - 2.4)**2/10000
        """
        Add systematic uncertainty for all the bkg
        """
        # nll += (v_n_epsilon[0])**2/0.5**2
        nll += (v_n[self.n_h2d_need_to_fit])**2/0.5**2
        # print(v_n_epsilon**2/0.5**2)
        for i, key in enumerate(self.dir_other_bkg_sys_uncertainty.keys()):
            # nll += (v_n_epsilon[i+1]/self.dir_other_bkg_sys_uncertainty[key])**2
            nll += (v_n[self.n_h2d_need_to_fit+i+1]/self.dir_other_bkg_sys_uncertainty[key])**2

        # Penalize negative bins
        # indices_negative = (n_j<0)
        # nll += (-5.)*np.sum(n_j[indices_negative])

        # nll = - np.sum(self.h2d_to_fit.values * log_n_j - n_j )
        # +N_exp - np.sum(self.h2d_to_fit.values)*np.log(N_exp)
        if self.plot_v_nll:
            print("v_n:\t:", v_n)
            print(f"nll:\t{nll}\n")
            self.v_nll.append(nll)
        return nll

    def PlotFitProfile(self, m):
        """
        This function is for checking the fit result whether is correct and plot the related profile

        Args:
            m: the object of iminuit which can return the fit result

        Returns:

        """
        v_fit_once = m.np_values()
        result_fit = v_fit_once[0] * self.h2d_sig + (1+v_fit_once[self.n_h2d_need_to_fit])*v_fit_once[1] * self.h2d_bkg_NC
        for i,key in enumerate(self.dir_h2d_other_bkg):
            result_fit = result_fit+(1+v_fit_once[self.n_h2d_need_to_fit+i+1])*v_fit_once[i+2]*self.dir_h2d_other_bkg[key]
        indices = (result_fit.values!=0)
        fig_profile_PSD, ax_profile_PSD = plt.subplots()
        hl.plot1d(ax_profile_PSD, result_fit.project([0]), label="Fit Result")
        hl.plot1d(ax_profile_PSD, self.h2d_to_fit.project([0]), label="Input Events")
        hl.plot1d(ax_profile_PSD, v_fit_once[0]*self.h2d_sig.project([0]), label="DSNB Fit")
        hl.plot1d(ax_profile_PSD, v_fit_once[1]*(1+v_fit_once[self.n_h2d_need_to_fit])*self.h2d_bkg_NC.project([0]), label="atm-NC Fit")
        for i,key in enumerate(self.dir_h2d_other_bkg):
            hl.plot1d(ax_profile_PSD, v_fit_once[i+2]*(v_fit_once[self.n_h2d_need_to_fit+i+1]+1)*self.dir_h2d_other_bkg[key].project([0]), label=key+" Fit")
        ax_profile_PSD.set_xlabel("PSD Output")
        plt.title("Projection of PSD Output")
        plt.legend()
        fig_profile_Edep, ax_profile_Edep = plt.subplots()
        hl.plot1d(ax_profile_Edep, result_fit.project([1]),label="Fit Result")
        hl.plot1d(ax_profile_Edep, self.h2d_to_fit.project([1]), label="Input PDF")
        hl.plot1d(ax_profile_Edep, v_fit_once[0]*self.h2d_sig.project([1]), label="DSNB Fit")
        # hl.plot1d(ax_profile_Edep, 15*v_fit_once[1]*self.h2d_bkg_NC.project([1])/np.sum(self.h2d_bkg_NC.project([1]).values), label="atm-NC Fit")
        hl.plot1d(ax_profile_Edep, v_fit_once[1]*(1+v_fit_once[self.n_h2d_need_to_fit])*self.h2d_bkg_NC.project([1]), label="atm-NC Fit")
        for i,key in enumerate(self.dir_h2d_other_bkg):
            hl.plot1d(ax_profile_Edep, v_fit_once[i+2]*(v_fit_once[self.n_h2d_need_to_fit+i+1]+1)*self.dir_h2d_other_bkg[key].project([1]), label=key+" Fit")
        ax_profile_Edep.set_xlabel("$E_{quen}$")
        plt.title("Projection of $E_{quen}$")
        plt.legend()
        plt.show()

    def FitHistZeroFix(self, v_n_initial):
        if self.print_fit_result:
            print("v_n_initial:\t",v_n_initial)
        check_zero_result = False
        v_limit = [(fit_down_limit_sig, None), (0, None)]
        for i in range(len(self.dir_h2d_other_bkg.keys())):
            v_limit.append((fit_down_limit_other_bkg, fit_up_limit_other_bkg))
        for i in range(len(self.dir_h2d_other_bkg.keys())+1):
            v_limit.append((-eposilon_limit, eposilon_limit))
        # print(v_limit)
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=v_limit, errordef=0.5)
        m.migrad()
        self.fitter = m
        print("nll values:\t", m.fval)
        #print(m.values)
        if  check_zero_result and m.np_values()[0] <=10.5 and m.np_values()[0]>=9.5:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16,6))
            hl.plot2d(ax4, flh.h2d_sig_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("Signal", ax4)
            hl.plot2d(ax5, flh.h2d_bkg_NC_to_fit, log=True, cbar=True, clabel="counts per bin")
            SetTitle("NC", ax5)
            plt.show()

        # print(m.np_values())
        if self.check_result and np.sum(m.np_values()[:7])>50:
            self.PlotFitProfile(m)
        return (m.np_values(), m.fval)

    def FitHistFixSigN(self, v_n_initial):
        if self.print_fit_result:
            print("v_n_initial:\t",v_n_initial)
        v_limit = [(fit_down_limit_sig, None), (0, None)]
        v_fix = [ True, False]
        for i in range(len(self.dir_h2d_other_bkg.keys())):
            v_limit.append((fit_down_limit_other_bkg, fit_up_limit_other_bkg))
        for i in range(len(self.dir_h2d_other_bkg.keys())+1):
            v_limit.append((-eposilon_limit, eposilon_limit))
        for i in range(len(v_n_initial)-2):
            v_fix.append(False)
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, limit=v_limit,\
                                   fix=v_fix, errordef=0.5 )
        m.migrad()
        self.fitter_fix = m
        if self.check_result and np.sum(m.np_values()[:7])>50:
            self.PlotFitProfile(m)
        return (m.np_values(),m.fval)

    # def FitHistFixOne(self):
def PrintFitResult(v_n):
    print("Total input events:\t", np.sum(flh.h2d_to_fit.values))
    print("DSNB:\t", v_n[0])
    print("atmNC:\t", v_n[1])
    for i,key in enumerate(flh.dir_h2d_other_bkg.keys()):
        print(key+":\t", v_n[i+2])
        if key == "FastN" and check_FastN_input_hist:
            if v_n[i+2]>7:
                flh.PlotHistToFit()
    print("Total fit events:\t", np.sum(v_n[:-1]))
    print("fit vector:\t",v_n)
    print("-------------------------------------")
def WhetherProblemFit(v_n, index_begin_epsilon):
    threshold = 50
    threshold_epsilon = 0.99
    # if np.sum(v_n[:index_begin_epsilon])> threshold:
    if np.max(v_n[index_begin_epsilon:]) > threshold_epsilon or np.min(v_n[index_begin_epsilon:])<-threshold_epsilon:
        return True
    else:
        return False

if __name__ == '__main__':
    fit_2d = True
    print_fit_result = False
    # fit_2d = False
    set_initial_epsilon_zeros = True
    plot_nll_transition = False
    name_file_predict = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/predict_0.npz"
    # only_best_fit = False
    only_best_fit = False
    uncertainty_other_bkg = 0.5
    fit_down_limit_sig = 0
    fit_down_limit_other_bkg = 0
    fit_up_limit_other_bkg = 40
    up_limit_random_init = 20
    up_limit_random_init_DSNB = 10
    up_limit_random_init_NC = 10
    eposilon_limit = 1
    np.random.seed(100)

    dir_other_bkg = {}
    flh = FLH()
    flh.plot_v_nll = plot_nll_transition
    flh.LoadPrediction(name_file_predict)
    flh.LoadOtherBkg("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/atm-CC_samples_predict_0.npz",
                     key="dict_samples",
                     key_in_dict="CC",
                     sys_uncertrainty=uncertainty_other_bkg)
    flh.LoadOtherBkg("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/Reactor-anti-Nu_samples_predict_0.npz",
                     key="dict_samples",
                     key_in_dict="Reactor-anti-Nu",
                     sys_uncertrainty=uncertainty_other_bkg)
    flh.LoadOtherBkg("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/Li9He8_samples_predict_0.npz",
                     key="dict_samples",
                     key_in_dict="He8Li9",
                     sys_uncertrainty=uncertainty_other_bkg)
    flh.LoadOtherBkg("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/FastN_samples_predict_0.npz",
                     key="dict_samples",
                     key_in_dict="FastN",
                     sys_uncertrainty=uncertainty_other_bkg)

    n_to_fit_bkg_NC = 460
    n_to_fit_sig = 0
    dir_n_other_bkg = {"CC":2.4,  "FastN":9.7, "He8Li9":0.08, "Reactor-anti-Nu":3.4}
    flh.SetNOtherBkg(dir_n_other_bkg)

    flh.Get2DPDFHist(fit_2d=fit_2d)

    from scipy.interpolate import interp1d
    v_uplimit_n_sig = []
    n_max_sig = 25
    chi2_criteria = 2.706
    check_FastN_input_hist = False
    # v_n_sig_to_fix = np.linspace(0, n_max_sig, n_fix_sig_num_try)
    v_n_sig_to_fix = np.arange(0, n_max_sig+1)
    v_fit_result = {"sig": [], "bkg": []}
    for key in flh.dir_h2d_other_bkg.keys():
        v_fit_result[key] = []
    v2D_chi2 = [] # dimension 0 is for the times of trying , dimension 1 is for the number of fix signal

    n_trails = 10
    if only_best_fit:
        n_trails *= 100
    for i in range(n_trails):
    # for i in [1]:
        if print_fit_result:
            print("---------------------------")
            print("seed number:\t", i)
        num_iterations = 0
        if i %100 ==0:
            print(f"Processing {i} times fitting")
        v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
        if not set_initial_epsilon_zeros:
            v_n_other_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.dir_h2d_other_bkg.keys()))
            NC_bkg_initial_epsilon = np.random.uniform(-1, 1)
        else:
            NC_bkg_initial_epsilon = 0
            v_n_other_bkg_initial_epsilon = np.zeros(len(flh.dir_other_bkg.keys()))
        flh.GetHistToFit(n_to_fit_bkg_NC=n_to_fit_bkg_NC, n_to_fit_sig=n_to_fit_sig, seed=i, print_fit_result=print_fit_result)
        v_n, f_val_nofix = flh.FitHistZeroFix(np.concatenate(([np.random.randint(0, up_limit_random_init_DSNB), np.random.randint(0, up_limit_random_init_NC)],
                                                              v_n_other_bkg_initial,
                                                              [NC_bkg_initial_epsilon],
                                                              v_n_other_bkg_initial_epsilon)))

        while (not (flh.fitter.get_fmin()['is_valid'] and flh.fitter.get_fmin()['has_accurate_covar'])) \
                or WhetherProblemFit(v_n, flh.n_h2d_need_to_fit):
            if num_iterations > 9:
                break
            num_iterations += 1
            print(f"Refitting!! {num_iterations} times")
            v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
            if not set_initial_epsilon_zeros:
                v_n_other_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.dir_h2d_other_bkg.keys()))
                NC_bkg_initial_epsilon = np.random.uniform(-1, 1)
            else:
                v_n_other_bkg_initial_epsilon = np.zeros(len(flh.dir_h2d_other_bkg.keys()))
                NC_bkg_initial_epsilon = 0
            v_n, f_val_nofix = flh.FitHistZeroFix(
                np.concatenate(([np.random.randint(0, up_limit_random_init_DSNB), np.random.randint(0, up_limit_random_init_NC)],
                                v_n_other_bkg_initial,
                                [NC_bkg_initial_epsilon],
                                v_n_other_bkg_initial_epsilon)))
        v_fit_result["sig"].append(v_n[0])
        v_fit_result["bkg"].append(v_n[1])
        if print_fit_result:
            PrintFitResult(v_n)
        for i, key in enumerate(flh.dir_h2d_other_bkg.keys()):
            v_fit_result[key].append(v_n[i + 2])

        if not only_best_fit:
            v_chi2 = []
            for j in v_n_sig_to_fix:
            # for j in [23]:
                v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
                if not set_initial_epsilon_zeros:
                    v_n_other_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.dir_h2d_other_bkg.keys()))
                    NC_bkg_initial_epsilon = np.random.uniform(-1, 1)
                else:
                    v_n_other_bkg_initial_epsilon = np.zeros(len(flh.dir_h2d_other_bkg.keys()))
                    NC_bkg_initial_epsilon = 0
                if plot_nll_transition:
                    flh.v_nll = []
                v_n_fix, f_val_fix =flh.FitHistFixSigN(v_n_initial=np.concatenate(([j, np.random.randint(0, up_limit_random_init_NC)],
                                                                                   v_n_other_bkg_initial,
                                                                                   [NC_bkg_initial_epsilon],
                                                                                   v_n_other_bkg_initial_epsilon)))
                # v_n_fix, f_val_fix =flh.FitHistFixSigN( [23., 7.,18.,11.,14.,15., 0., 0. ,0. ,0. ,0.])
                if plot_nll_transition:
                    plt.plot(flh.v_nll)
                    plt.show()
                num_iterations = 0
                while (not (flh.fitter_fix.get_fmin()['is_valid'] and flh.fitter_fix.get_fmin()['has_accurate_covar']))\
                        or WhetherProblemFit(v_n_fix, flh.n_h2d_need_to_fit):
                    if num_iterations > 9:
                        break
                    print(f"Refitting!! {num_iterations} times in fixing fit")
                    num_iterations += 1
                    v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
                    if not set_initial_epsilon_zeros:
                        v_n_other_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.dir_h2d_other_bkg.keys()))
                        NC_bkg_initial_epsilon = np.random.uniform(-1, 1)
                    else:
                        v_n_other_bkg_initial_epsilon = np.zeros(len(flh.dir_h2d_other_bkg.keys()))
                        NC_bkg_initial_epsilon = 0
                    v_n_fix, f_val_fix =flh.FitHistFixSigN(v_n_initial=np.concatenate(([j, np.random.randint(0, up_limit_random_init_NC)],
                                                                                       v_n_other_bkg_initial,
                                                                                       [NC_bkg_initial_epsilon],
                                                                                       v_n_other_bkg_initial_epsilon)))
                if print_fit_result:
                    print(f"---------fix result(N_sig={j}) --------------")
                    PrintFitResult(v_n_fix)
                v_chi2.append(f_val_fix-f_val_nofix)

            index_min = np.argmin(v_chi2)
            v2D_chi2.append(v_chi2)
            try:
                f = interp1d( v_chi2[index_min:], v_n_sig_to_fix[index_min:], kind="linear", fill_value="extrapolate")
                uplimit_n_sig = f(chi2_criteria)
                v_uplimit_n_sig.append(uplimit_n_sig)
            except Exception:
                print("Getting chi2 uplimit failed!")
                continue

        # Check Getting uplimit
        # plt.figure()
        # plt.plot(v_n_sig_to_fix, v_chi2)
        # plt.plot([0, n_max_sig], [chi2_criteria, chi2_criteria], "--", label="90% confidence")
        # print("uplimit:\t",uplimit_n_sig)
        # print("chi2:\t", v_chi2)
        # print("index:\t", index_min)
        # plt.show()


    plt.figure()
    if n_to_fit_sig != 0:
        plt.hist(v_fit_result["sig"], histtype="step", bins=50)
    else:
        plt.hist(v_fit_result["sig"], histtype="step", bins=100)
    plt.xlabel("N of Signal")

    plt.figure()
    plt.hist(v_fit_result["bkg"], histtype="step", bins=100)
    plt.xlabel("N of NC")

    for i, key in enumerate(flh.dir_h2d_other_bkg.keys()):
        plt.figure()
        plt.hist(v_fit_result[key], histtype="step", bins=100)
        plt.xlabel("N of "+key)

    if not only_best_fit:
        plt.figure()
        for i in range(len(v2D_chi2)):
            plt.plot(v_n_sig_to_fix,v2D_chi2[i])
        plt.plot([0, n_max_sig], [chi2_criteria, chi2_criteria],"--", label="90% confidence")
        plt.ylim(0,10)
        plt.xlabel("Number of Signal Counts")
        plt.ylabel("$L_{min}^{fix}-L_{min}^{nofix}$")
        plt.legend()

        plt.figure()
        v_uplimit_n_sig = np.array(v_uplimit_n_sig).reshape(-1)
        # print(v_uplimit_n_sig)
        h_uplimit = plt.hist(v_uplimit_n_sig, histtype="step", bins=10)
        median_uplimit = np.median(v_uplimit_n_sig)
        print("median:\t", median_uplimit)
        plt.plot([median_uplimit, median_uplimit], [0, np.max(h_uplimit[0])], "--")
        plt.xlabel("Uplimit of Number of signal")
        if fit_2d:
            np.save("v_uplimit_2d.npy", v_uplimit_n_sig)
        else:
            np.save("v_uplimit_1d.npy", v_uplimit_n_sig)


    plt.show()

### dictionary map : /junofs_500G/sk_psd_DSNB
###or map : /gpu_500G/LS_ML_CNN

