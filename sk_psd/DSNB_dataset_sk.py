import uproot as up
import ROOT
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tqdm
def GenFilesList(dir_data:str, n_start_sig, n_start_bkg, step ):
    if dir_data[-1] != "/":
        dir_data += "/"
    filelist_sig = []
    filelist_bkg = []
    for i_sig in range(n_start_sig, n_start_sig+step):
        filelist_sig.append(dir_data+"DSNB/data/dsnb_"+str(i_sig).zfill(6)+".root")
    for i_bkg in range(n_start_bkg, n_start_bkg+step*3):
        filelist_bkg.append(dir_data+"AtmNu/data/atm_"+str(i_bkg).zfill(6)+".root")
    print("Filelist :", filelist_bkg, filelist_sig)
    return (filelist_sig, filelist_bkg)

def LoadData(tchain:ROOT.TChain, name_type:str, h2d:ROOT.TH2D):
    bins_hist = range(-down_time, up_time, binwidth)
    max_index_hist = -10
    n_beforePeak = 20
    data_save = np.zeros((tchain.GetEntries(), 2, 420), dtype=np.float32)
    v_equen = []
    v_vertex = []
    n_tail = 400
    for i in tqdm.tqdm(range(tchain.GetEntries())):
    # for i in range(1,10):
        # plt.figure(name_type+str(i))
        tchain.GetEntry(i)
        pmtids = tchain.PMTID
        npes = tchain.Charge
        hittime = tchain.Time
        eqen = tchain.Eqen
        x = tchain.X
        y = tchain.Y
        z = tchain.Z
        v_equen.append(eqen)
        v_vertex.append([x, y, z])
        hist, bin_edges = np.histogram(hittime, bins=bins_hist )
        hist_weightE, bin_edges_weightE = np.histogram(hittime, bins=bins_hist, weights=npes)
        if max_index_hist == -10:
            max_index_hist = hist.argmax()
        hist = hist/hist.max()
        hist_weightE = hist_weightE/hist_weightE.max()
        data_save[i] = np.array([hist[max_index_hist-n_beforePeak: max_index_hist+n_tail], hist_weightE[max_index_hist-n_beforePeak: max_index_hist+n_tail]])
        # data_save.append(np.array([hist[max_index_hist-n_beforePeak: max_index_hist+n_tail], hist_weightE[max_index_hist-n_beforePeak: max_index_hist+n_tail]]))
        if plot_result:
            hist = hist_weightE[max_index_hist - n_beforePeak: max_index_hist + n_tail]
            bin_edges = bin_edges_weightE[max_index_hist - n_beforePeak: max_index_hist + n_tail]
            for j_time in range(len(hist)):
                h2d.Fill(bin_edges[j_time], hist[j_time])
        # plt.plot(hist, label=name_type+str(i))
    return (np.array(data_save), np.array(v_equen), np.array(v_vertex))
def GetProfile(h2d:ROOT.TH2D):
    h_profile = h2d.ProfileX()
    v_profile = []
    for i in range(h2d.GetNbinsX()):
       v_profile.append(h_profile.GetBinContent(i))
    return np.array(v_profile)
def PlotTimeProfile(h2d_sig:ROOT.TH2D, h2d_bkg:ROOT.TH2D):
    v_profile_sig = GetProfile(h2d_sig)
    v_profile_bkg = GetProfile(h2d_bkg)
    fig_profile_time = plt.figure("Profile_time")
    plt.plot(v_profile_bkg, label="background")
    plt.plot(v_profile_sig, label="signal")
    plt.xlabel("Time [ ns ]")
    return fig_profile_time

if __name__ == "__main__":
    up_time = 1000
    down_time = 200
    binwidth = 2
    plot_result = False
    n_bins = (up_time+down_time)/binwidth
    h2d_time_sig = ROOT.TH2D("h_time_sig", "h_time_sig", int(n_bins), -down_time, up_time, int(n_bins), 0, 1.1)
    h2d_time_bkg = ROOT.TH2D("h_time_bkg", "h_time_bkg", int(n_bins), -down_time, up_time, int(n_bins), 0, 1.1)

    parser = argparse.ArgumentParser(description='DSNB sklearn dataset builder.')
    parser.add_argument("--inputdir", "-d", type=str, default="/workfs/exo/zepengli94/JUNO_DSNB", help="Raw data directory")
    parser.add_argument("--nstart_sig", "-ss", type=int, default=1, help="start num of signal file")
    parser.add_argument("--nstart_bkg", "-sb", type=int, default=1, help="start num of bkg file")
    parser.add_argument("--step", "-s", type=int, default=10, help="how many signal files to put into one output")
    parser.add_argument("--outfile", "-o", type=str, default="try.npz", help="name of outfile")
    arg = parser.parse_args()
    (filelist_sig, filelist_bkg) = GenFilesList(arg.inputdir, arg.nstart_sig, arg.nstart_bkg, arg.step)

    # name_sig_files = "/workfs/exo/zepengli94/JUNO_DSNB/DSNB/data/dsnb_00000[1-9].root"
    # name_bkg_files_1 = "/workfs/exo/zepengli94/JUNO_DSNB/AtmNu/data/atm_00000[1-9].root"
    # name_bkg_files_2 = "/workfs/exo/zepengli94/JUNO_DSNB/AtmNu/data/atm_0000[10-19].root"
    sigchain = ROOT.TChain('psdtree')
    for i in range(len(filelist_sig)):
        sigchain.Add(filelist_sig[i])
    # sigchain.Add(name_sig_files)
    # sigchain.Add('%s/*root' % sig_dir)
    bkgchain = ROOT.TChain('psdtree')
    for i in range(len(filelist_bkg)):
        bkgchain.Add(filelist_bkg[i])
    # bkgchain.Add(name_bkg_files_1)
    # bkgchain.Add(name_bkg_files_2)

    print(f"sig_entries : {sigchain.GetEntries()},  bkg_entries : { bkgchain.GetEntries()}")
    # bkgchain.Add('%s/*root' % bkg_dir)
    (data_save_sig, v_equen_sig, v_vertex_sig) = LoadData(sigchain, "sig", h2d_time_sig)
    (data_save_bkg, v_equen_bkg, v_vertex_bkg) = LoadData(bkgchain, "bkg", h2d_time_bkg)
    # print(f"shape:   (data_save_sig:{data_save_sig.shape}), (v_equen_sig:{v_equen_sig}), (v_vertex_sig:{v_vertex_sig})")
    np.savez(arg.outfile, sig=data_save_sig, bkg=data_save_bkg, sig_vertex=v_vertex_sig, bkg_vertex=v_vertex_bkg, sig_equen=v_equen_sig, bkg_equen=v_equen_bkg )

    if plot_result:
        print(f"sig_data: {data_save_sig.shape},\n bkg_data: {data_save_bkg.shape}")
        h2d_time_bkg.SetStats(False)
        h2d_time_sig.SetStats(False)
        c_time_sig = ROOT.TCanvas("c_sig", "c_sig", 800, 600)
        c_time_sig.SetLogz()
        h2d_time_sig.SetXTitle("Time [ ns ]")
        h2d_time_sig.DrawCopy("colz")
        c_time_sig.SaveAs("TH2D_sig_time.png")
        c_time_bkg = ROOT.TCanvas("c_bkg", "c_bkg", 800, 600)
        h2d_time_bkg.SetXTitle("Time [ ns ]")
        c_time_bkg.SetLogz()
        h2d_time_bkg.DrawCopy("colz")
        c_time_bkg.SaveAs("TH2D_bkg_time.png")

        fig_profile_time = PlotTimeProfile(h2d_time_sig, h2d_time_bkg)
        fig_profile_time.savefig("profile_time.png")

        plt.legend()
        plt.show()

