import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import ROOT
import os
# from matplotlib.animation import FuncAnimation,writers
import argparse
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
# print( os.path.getsize("test_score_1.npy") )
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
def LoadAndPlotScore(dir_test_score ):
    # name_saveGIF = dir_test_score+"DistanceBiasVsE.gif"
    # name_saveGIF_d0 = dir_test_score+"D0BiasVs.gif"
    # # c_dVsE = ROOT.TCanvas("c_dVsE", "c_dVsE", 1000, 800)
    # # c_d0VsE = ROOT.TCanvas("c_d0VsE", "c_d0VsE", 1000, 800)
    #
    # if os.path.exists( name_saveGIF ):
    #     os.remove( name_saveGIF )
    # if os.path.exists( name_saveGIF_d0):
    #     os.remove(name_saveGIF_d0)
    test_score = np.load(dir_test_score + "test_score_1.npy", allow_pickle=True)

    edges_x_d3 = np.arange(0,8000,200)
    edges_y_d = np.arange(0,600,10)
    edges_x_E = np.arange(0, 11, 0.05)
    edges_y_d0 = np.arange(-500, 500, 5)
    v_fig = []
    fig = plt.figure("d Bias VS d3 (1MeV-2MeV)", figsize=(12,8))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.6, hspace=0.2)
    # fig_d = plt.figure("E VS d Bias", figsize=(12,9))
    # fig_d0= plt.figure("E VS d0 Bias", figsize=(12,9))
    # for epoch in range(len(test_score)):
    def loop_epoch(epoch):
        v_d_1MeV = []
        v_d3 = []
        v_d = []
        v_E = []
        v_d0Bias = []
        # h2d = ROOT.TH2D("dist_E", "e+ epoch "+str(epoch), 200, 0, 11, 200, 0, 1000)
        # h2d_d0 = ROOT.TH2D("distFrom0_E", "e+ epoch "+str(epoch), 200, 0, 11, 200, -500, 500)
        for i_evt, v_vertex in enumerate(test_score[ epoch ]):
            vertex_truth = v_vertex[0]
            vertex_output = v_vertex[1]
            E = v_vertex[2]
            v_E.append(E)
            d = GetDist( vertex_truth, vertex_output)
            v_d.append(d)
            d0_truth = GetDistFromOrigin( vertex_truth )
            d0_predict = GetDistFromOrigin( vertex_output )
            v_d0Bias.append(d0_truth-d0_predict)
            # print( i_evt ,d )
            # h2d.Fill( E, d )
            # h2d_d0.Fill( E, d0_truth-d0_predict )
            v_d3.append(np.sum(vertex_truth*2)*1.5)
            v_d_1MeV.append(d)
        fig.clear()
        fig.add_subplot(1,3,1)
        im = plt.hist2d(  v_d3, v_d_1MeV, bins=(edges_x_d3, edges_y_d), cmap="Blues")
        cbar = plt.colorbar()
        plt.title("epoch : "+ str(epoch))
        cbar.ax.set_ylabel("Counts")
        plt.xlabel("d3")
        plt.ylabel("D Bias")

        fig.add_subplot(1,3,2)
        cbar1 = plt.colorbar()
        plt.title("epoch : "+ str(epoch))
        cbar1.ax.set_ylabel("Counts")
        plt.xlabel("Energy [ MeV ]")
        plt.ylabel("distance bewteen Truth and Recon")
        im2 = plt.hist2d( v_E, v_d, bins=(edges_x_E, edges_y_d), cmap="Blues")

        fig.add_subplot(1,3,3)
        cbar2 = plt.colorbar()
        plt.title("epoch : "+ str(epoch))
        cbar2.ax.set_ylabel("Counts")
        plt.xlabel("Energy [ MeV ]")
        plt.ylabel("| $d_0^Truth-d_0^Recon$ |")
        im3 = plt.hist2d( v_E, v_d0Bias, bins=(edges_x_E, edges_y_d0), cmap="Blues")
        return im, im2, im3

    # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani = animation.ArtistAnimation( fig, v_fig, interval=50, blit=True)
    anim = animation.FuncAnimation(fig, loop_epoch, frames=range(len(test_score)), interval=200)
    # anim_d = animation.FuncAnimation(fig_d, loop_epoch, frames=range(len(test_score)), interval=200)
    # anim_d0 = animation.FuncAnimation(fig_d0, loop_epoch, frames=range(len(test_score)), interval=200)
    writergif = animation.PillowWriter(fps=20)
    anim.save(dir_test_score+"dBiasVSd3.gif", writer=writergif)
    # anim_d0.save(dir_test_score+"dBiasVSE.gif", writer=writergif)
    # anim_d.save(dir_test_score+"distVSE.gif", writer=writergif)
    # plt.show()




            # if i_evt >5:
            #     break
        # c_dVsE.cd()
        # h2d.SetStats(False)
        # h2d.SetXTitle(" Energy [ MeV ]")
        # h2d.SetYTitle(" Distance Bias [ mm ]")
        # h2d.DrawCopy("colz")
        # h2d.Delete()
        # c_dVsE.Print( name_saveGIF + "+60")
        #
        # c_d0VsE.cd()
        # h2d_d0.SetStats(False)
        # h2d_d0.SetXTitle(" Energy [ MeV ]")
        # h2d_d0.SetYTitle(" d0 Bias [ mm ]")
        # h2d_d0.DrawCopy("colz")
        # h2d_d0.Delete()
        # c_d0VsE.Print( name_saveGIF_d0 + "+60")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_test")
    parser.add_argument('--dir', '-d', type=str, help="dictionary for test_score" )
    args = parser.parse_args()
    print( "input :  ",args.dir )
    LoadAndPlotScore( "./"+args.dir)