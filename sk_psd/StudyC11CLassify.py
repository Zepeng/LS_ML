# -*- coding:utf-8 -*-
# @Time: 2021/2/25 15:44
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: StudyC11CLassify.py
import matplotlib.pylab as plt

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

#%%

# load file
import numpy as np
# f = np.load("./model_maxtime_combine_jobs_DSNB_sk_data/predict_0.npz", allow_pickle=True)
f = np.load("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/predict_0.npz", allow_pickle=True)
print(f"key: {f.files}")
predict_proba = f["predict_proba"][:,1]
equen = f["equen"]
vertex = f["vertex"]
labels = f["labels"]
pdgs = f["pdg_bkg"]
# print("predict_proba: ", predict_proba)
# print("labels: ", labels)
# print("pdg:  ", pdgs)

#%%

# seperate sig and bkg
dir_proba ={}
dir_vertex = {}
dir_equen = {}
dir_proba["sig"] = predict_proba[labels==1]
dir_proba["bkg"] = predict_proba[labels==0]
dir_vertex["sig"] = vertex[labels==1]
dir_vertex["bkg"] = vertex[labels==0]
dir_equen["sig"] = equen[labels==1]
dir_equen["bkg"] = equen[labels==0]

# print("Check bkg length: ")
# print("proba->  ", len(dir_proba["bkg"]))
# print("pdg-> ", len(pdgs))
# print("vertex-> ", len(dir_vertex["bkg"]))

#%%

# Study pdgs
from collections import Counter
def GetNucleiNum(pdg_evt):
    n_nuclei = 0
    counter = Counter(pdg_evt)
    for key in counter:
        if key > 1000000000:
            n_nuclei += counter[key]
    return n_nuclei

def PdgToN(Nuclei_pdg):
    N = int(Nuclei_pdg/10)%1000
    Z = int(Nuclei_pdg/10000)%1000
    return (N, Z)

class OneNucleiEvts:
    def __init__(self):
        self.probs = []
        self.v_NZ = []
    def Print(self):
        print(f"probs: {self.probs}")
        print(f"v_NZ: {self.v_NZ}")
evt_1Nuclei = OneNucleiEvts()

v_pdg_multi_nuclei = []
v_proba_multi_nuclei = []
for i, pdg_evt in enumerate(pdgs):
    n_nuclei = GetNucleiNum(pdg_evt)
    if n_nuclei == 1 :
        Nuclei_pdg = pdg_evt[pdg_evt>1000000000][0]
        (N, Z ) = PdgToN(Nuclei_pdg)
        evt_1Nuclei.probs.append(dir_proba["bkg"][i])
        evt_1Nuclei.v_NZ.append([N,Z])
    else:
        v_proba_multi_nuclei.append(dir_proba["bkg"][i] )
        v_pdg_multi_nuclei.append(pdg_evt)
# evt_1Nuclei.Print()


#%% md

## Find The Right Criteria

#%%

criteria_to_use = 0
for criteria in np.arange(0.9, 1, 0.005 ):
    print("Criteria :\t", criteria)
    index_bkg_rightPredict = (dir_proba["bkg"]<criteria)
    counter_bkglike = Counter(index_bkg_rightPredict)
    eff_bkg = counter_bkglike[True]/len(index_bkg_rightPredict)
    print("Efficiency of bkg:\t", eff_bkg)

    index_sig_rightPredict = (dir_proba["sig"]>criteria)
    counter_sig_rightPredict = Counter(index_sig_rightPredict)
    eff_sig = counter_sig_rightPredict[True]/len(index_sig_rightPredict)
    print("Efficiency of sig:\t", eff_sig)
    if eff_bkg > 0.99:
        criteria_to_use = criteria
        break
    print("###########################################")
print("Will use criteria --> ", criteria_to_use)

#%% md

## Check $^{11}C$ and $^{10}B$ Background (Signal like ratio)


#%%

def HistTimes(hist:np.ndarray, times:int):
    hist_return = list(hist)*times
    return np.array(hist_return)

#%%



#%%

## Draw Nuclei distribution

criteria = criteria_to_use
index_siglike = (np.array(evt_1Nuclei.probs)>=criteria)
index_bkglike = (np.array(evt_1Nuclei.probs)<criteria)

labels = ["$H$", "$He$", "$Li$", "$Be$", "$B$", "$C$", "$N$", "$O$"]
plt.figure(figsize=(9, 6))
x =np.arange(1, len(labels)+1)
h_siglike = plt.hist(HistTimes(np.array(evt_1Nuclei.v_NZ)[index_siglike][:,1],10), bins=x , histtype='step', label="Sig-like*10($P_{sig}>=$"+"{:.2f}".format(criteria)+")")
h_bkglike = plt.hist(np.array(evt_1Nuclei.v_NZ)[index_bkglike][:,1], bins=x, histtype='step', label="Bkg-like($P_{sig}"+"<${:.2f}".format(criteria)+")")
plt.title("Background Distribution")
plt.xticks(x+0.5, labels)
plt.xlim([0, len(labels)+1])
# plt.xlabel("Z of Proton")
plt.legend(loc="upper left")

plt.figure(figsize=(9, 6))
plt.hist(HistTimes(np.array(evt_1Nuclei.v_NZ)[index_siglike][:,0],10), bins=range(0, 15), histtype='step', label="Sig-like*10($P_{sig}>=$"+"{:.2f}".format(criteria)+")")
plt.hist(np.array(evt_1Nuclei.v_NZ)[index_bkglike][:,0], bins=range(0, 15), histtype='step', label="Bkg-like($P_{sig}"+"<${:.2f}".format(criteria)+")")
plt.title("Background Distribution")
plt.xlabel("N of Nuclei")
plt.legend(loc="upper left")

plt.figure(figsize=(8, 5))
h_siglike = np.array(h_siglike[0])/10
h_bkglike = h_bkglike[0]
h_ratio = np.nan_to_num(np.array(h_siglike)/((np.array(h_bkglike[0]))+np.array(h_siglike)))
plt.stem(h_ratio)
# plt.xlabel("Z of Proton")
plt.xticks(x-1, labels)
plt.ylabel("Ratio")
plt.title("Ratio(Sig-like/Total)")

# plt.show()

#%%

print(h_siglike)
print(h_bkglike)

#handle multi-nuclei evt
# print(v_pdg_multi_nuclei, v_proba_multi_nuclei)
print(f"Total multi-nuclei : {len(v_proba_multi_nuclei)}")
v_proba_multi_nuclei = np.array(v_proba_multi_nuclei)
print(f"sig like : {len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)>=criteria])}" )
print(f"bkg like : {len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)<criteria])}" )
n_siglike_multi_nuclei = len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)>=criteria])
n_bkglike_multi_nuclei = len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)<criteria])
h_bkglike = np.append(h_bkglike, [n_bkglike_multi_nuclei])
h_siglike = np.append(h_siglike, [n_siglike_multi_nuclei])
h_bkglike_ratio_to_total = np.nan_to_num(h_siglike/(np.sum(h_siglike+h_bkglike)))
h_siglike_ratio_to_total = np.nan_to_num(h_bkglike/(np.sum(h_siglike+h_bkglike)))

print(h_siglike, h_bkglike)

import pandas as pd
table = pd.DataFrame([np.array(h_bkglike , dtype=np.int),np.array(h_siglike, dtype=np.int)], index=["Background Like", "Signal Like"], columns=labels[:-1]+["Multi-nuclei"])
print(table)

print("C11 ratio: ", np.sum(h_bkglike_ratio_to_total)-h_bkglike_ratio_to_total[-3])
print("non-C11 ratio: ", h_bkglike_ratio_to_total[-3])
print(h_siglike_ratio_to_total)


