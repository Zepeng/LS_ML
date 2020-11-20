import numpy as np
import matplotlib.pyplot as plt
import ROOT
import os
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

test_score = np.load("test_score_1.npy", allow_pickle=True)
# outputs, targets, spectators
# for i in range(len(test_score)):
#     print(i, test_score[i][:5])
# print(test_score[i][:5])
# print(test_score.shape)
# print( len(test_score) )
def GetDist( vertex_truth:np.ndarray, vertex_output:np.ndarray ):
    d = np.sum((vertex_output-vertex_truth)**2)
    d = d**0.5
    return d
def LoadAndPlotScore():
    name_saveGIF =  "DistanceBiasVsE.gif"
    v_dist = []
    v_E = []
    c_dVsE = ROOT.TCanvas("c_dVsE", "c_dVsE", 1000, 800)

    if os.path.exists( name_saveGIF ):
        os.remove( name_saveGIF )

    for epoch in range(len(test_score)):
        h2d = ROOT.TH2D("dist_E", "e+ epoch "+str(epoch), 200, 0, 11, 200, 0, 1000)
        for i_evt, v_vertex in enumerate(test_score[ epoch ]):
            vertex_truth = v_vertex[0]
            vertex_output = v_vertex[1]
            E =  v_vertex[2]
            d = GetDist( vertex_truth, vertex_output)
            # print( i_evt ,d )
            v_dist.append( d )
            v_E.append(E)
            h2d.Fill( E, d )

            # if i_evt >5:
            #     break
        c_dVsE.cd()
        h2d.SetStats(False)
        h2d.SetXTitle(" Energy [ MeV ]")
        h2d.SetYTitle(" Distance Bias [ mm ]")
        h2d.DrawCopy("colz")
        h2d.Delete()
        c_dVsE.Print(name_saveGIF+"+"+str(len(test_score)))
    plt.show()

if __name__ == "__main__":
    LoadAndPlotScore()