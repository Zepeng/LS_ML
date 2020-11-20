import ROOT
import numpy as np
import matplotlib.pyplot as plt

f = ROOT.TFile.Open(
    "root://junoeos01.ihep.ac.cn//eos/juno/valprod/valprod0/J20v2r0-Pre0/ACU+CLS/Laser/photon_11522/Laser_0_0_-15990.5/calib_coti/user-root/user-calib_coti-81201.root")

tree = f.Get("CALIBEVT")
v_totalPE=[]
# tree.SetBranchAddress("TotalPE", totalPe )
for i in range(tree.GetEntries()):
    tree.GetEntry(i)

    v_totalPE.append(tree.TotalPE)
v_totalPE = np.array(v_totalPE)
# print(v_totalPE)
h= plt.hist(v_totalPE,bins=100)
plt.show()



