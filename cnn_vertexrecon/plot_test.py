import numpy as np
import matplotlib.pyplot as plt
import ROOT
import os
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

# print( os.path.getsize("test_score_1.npy") )
test_score = np.load("test_score_19.npy", allow_pickle=True)
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
def GetDistFromOrigin( vertex:np.ndarray ):
    d = np.sum(vertex**2)
    d = d**0.5
    return d
def LoadAndPlotScore():
    name_saveGIF =  "DistanceBiasVsE.gif"
    name_saveGIF_d0 = "D0BiasVs.gif"
    v_dist = []
    v_E = []
    c_dVsE = ROOT.TCanvas("c_dVsE", "c_dVsE", 1000, 800)
    c_d0VsE = ROOT.TCanvas("c_d0VsE", "c_d0VsE", 1000, 800)

    if os.path.exists( name_saveGIF ):
        os.remove( name_saveGIF )
    if os.path.exists( name_saveGIF_d0):
        os.remove(name_saveGIF_d0)

    for epoch in range(len(test_score)):
        h2d = ROOT.TH2D("dist_E", "e+ epoch "+str(epoch), 200, 0, 11, 200, 0, 1000)
        h2d_d0 = ROOT.TH2D("distFrom0_E", "e+ epoch "+str(epoch), 200, 0, 11, 200, -500, 500)
        for i_evt, v_vertex in enumerate(test_score[ epoch ]):
            vertex_truth = v_vertex[0]
            vertex_output = v_vertex[1]
            E = v_vertex[2]
            d = GetDist( vertex_truth, vertex_output)
            d0_truth = GetDistFromOrigin( vertex_truth )
            d0_predict = GetDistFromOrigin( vertex_output )
            # print( i_evt ,d )
            v_dist.append( d )
            v_E.append(E)
            h2d.Fill( E, d )
            h2d_d0.Fill( E, d0_truth-d0_predict )

            # if i_evt >5:
            #     break
        c_dVsE.cd()
        h2d.SetStats(False)
        h2d.SetXTitle(" Energy [ MeV ]")
        h2d.SetYTitle(" Distance Bias [ mm ]")
        h2d.DrawCopy("colz")
        h2d.Delete()
        c_dVsE.Print(name_saveGIF + "+60")

        c_d0VsE.cd()
        h2d_d0.SetStats(False)
        h2d_d0.SetXTitle(" Energy [ MeV ]")
        h2d_d0.SetYTitle(" d0 Bias [ mm ]")
        h2d_d0.DrawCopy("colz")
        h2d_d0.Delete()
        c_d0VsE.Print(name_saveGIF_d0 + "+60")
    plt.show()

if __name__ == "__main__":
    LoadAndPlotScore()