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
    for i_bkg in range(n_start_bkg, n_start_bkg+(step+1)*3-1):
        # filelist_bkg.append(dir_data+"AtmNu/data/atm_"+str(i_bkg).zfill(6)+".root")
        filelist_bkg.append("/dybfs/users/chengj/data/PSD/AtmNC/Model-G/data/atm_" + str(i_bkg).zfill(6) + ".root")
    print("Filelist :", filelist_bkg, filelist_sig)
    return (filelist_sig, filelist_bkg)

def LoadData(tchain:ROOT.TChain, name_type:str, h2d:ROOT.TH2D):

    ################ rebinning strategy 1 #############################
    # binwidth_weightE = 5
    # binwidth = 5
    # bins_hist_weightE = np.concatenate((np.array([-200]), np.arange(-50+binwidth_weightE/2., 200+binwidth_weightE/2., binwidth_weightE), np.linspace(200+binwidth_weightE/2., 1000+binwidth_weightE/2., 6)))
    # # bins_hist_weightE = np.concatenate((np.array([-200]), np.arange(-50., 200., binwidth_weightE), np.linspace(200, 1000, 5)))
    # bins_hist = np.concatenate((np.array([-200]), np.arange(-50+binwidth/2., 200+binwidth/2., binwidth), np.linspace(200+binwidth/2., 1000+binwidth/2., 6)))
    ###################################################################

    ################ rebinning strategy 2 ###################################
    # binwidth_weightE = 1
    # binwidth = 10
    # tail_bin_edge = np.arange(20+binwidth/2., 100+binwidth/2.)
    # step_range = range(1000)
    # for i in range(len(tail_bin_edge)):
    #     if i==0:
    #         continue
    #     else:
    #         tail_bin_edge[i] = tail_bin_edge[i-1]+step_range[i]
    # tail_bin_edge = tail_bin_edge[tail_bin_edge<=1000]
    # bins_hist_weightE = np.concatenate(( np.arange(-20+binwidth_weightE/2., 20+binwidth_weightE/2., binwidth_weightE), tail_bin_edge))
    # # bins_hist_weightE = np.concatenate(( np.arange(-20., 20., binwidth_weightE), tail_bin_edge))
    # # bins_hist_weightE = np.concatenate((np.array([-200]), np.arange(-50., 200., binwidth_weightE), np.linspace(200, 1000, 5)))
    # bins_hist = np.concatenate(( np.arange(-20+binwidth_weightE/2., 20+binwidth_weightE/2., binwidth_weightE), tail_bin_edge))
    ############################################################################

    ################ rebinning strategy 3##################################
    bins_hist = [-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,66,72,80,90,102,116,132,150,170,192,216,242,270,300,332,366,402,440,480,522,566,612,660,710,762,816]
    bins_hist_weightE = [-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,66,72,80,90,102,116,132,150,170,192,216,242,270,300,332,366,402,440,480,522,566,612,660,710,762,816]
    # bins_width, bins_width_weightE = np.diff(bins_hist), np.diff(bins_hist_weightE)

    # bins_hist = np.arange(-20, 800)
    # bins_hist_weightE = np.arange(-20, 800)

    print(f"bins_hist: {bins_hist_weightE}")
    max_index_hist = -10
    n_beforePeak = 20
    # data_save = np.zeros((tchain.GetEntries(), 2, len(bins_hist)-1), dtype=np.float32)
    data_save = {"NoWeightE":[], "WeightE":[]}
    v_equen = []
    v_vertex = []
    v_pdg = []
    v_px = []
    v_py = []
    v_pz = []
    is_bkg = (name_type=="bkg")
    # for i in tqdm.tqdm(range(tchain.GetEntries())):
    if plot_single:
        v_iter = range(1,10)
    else:
        v_iter = range(tchain.GetEntries)
    for i in v_iter:
    # for i in range(1,10):
        if i % 200==0:
            print(f"processing entry {i}")
        # plt.figure(name_type+str(i))
        tchain.GetEntry(i)
        pmtids = tchain.PMTID
        npes = tchain.Charge
        hittime = tchain.Time
        # print(f"check entry {i} ,hittime: {len(hittime)}")
        eqen = tchain.Eqen
        x = tchain.X
        y = tchain.Y
        z = tchain.Z
        if is_bkg and save_pdg:
            pdg = np.array(tchain.initpdg)
            px  = np.array(tchain.initpx)
            py  = np.array(tchain.initpy)
            pz  = np.array(tchain.initpz)
            v_pdg.append(pdg)
            v_px.append(px)
            v_py.append(py)
            v_pz.append(pz)
        v_equen.append(eqen)
        v_vertex.append([x, y, z])
        hist, bin_edges = np.histogram(hittime, bins=bins_hist, density=True)
        hist_weightE, bin_edges_weightE = np.histogram(hittime, bins=bins_hist_weightE, weights=npes, density=True)
        # hist = hist/bins_width
        # hist_weightE = hist_weightE/bins_width_weightE
        # hist/= hist[18]
        # hist_weightE /= hist_weightE[18]

        ################### divided by max###################################
        # print(f"check entry {i}, hist of hittime : {hist_weightE}")
        # if max_index_hist == -10:
        #     max_index_hist = hist.argmax()
        # hist = hist/hist.max()
        # hist_weightE = hist_weightE/hist_weightE.max()
        ####################################################################

        # print(f"hist_weightE:  {hist_weightE}")
        # data_save[i] = np.array([hist, hist_weightE])
        # data_save.append(np.array([hist[max_index_hist-n_beforePeak: max_index_hist+n_tail], hist_weightE[max_index_hist-n_beforePeak: max_index_hist+n_tail]]))
        # data_save.append(np.array([hist, hist_weightE]))
        data_save["NoWeightE"].append(np.array([hist]))
        data_save["WeightE"].append(np.array([hist_weightE]))
        # print(f"data_save: {np.array(data_save).shape}")
        if plot_result:
            # hist = hist_weightE[max_index_hist - n_beforePeak: max_index_hist + n_tail]
            # bin_edges = bin_edges_weightE[max_index_hist - n_beforePeak: max_index_hist + n_tail]
            hist = hist_weightE
            bin_edges = bin_edges_weightE
            # hist = hist
            # bin_edges = bin_edges
            for j_time in range(len(hist)):
                h2d.Fill(bin_edges[j_time], hist[j_time])
        if plot_single:
            plt.figure(name_type+"_fig")
            plt.plot(bin_edges[:-1],hist, label=name_type+str(i))
    # plt.semilogy()
    if plot_single:
        plt.legend()
    if is_bkg and save_pdg:
        print(f"check shape ---> pdg:{len(v_pdg)}, px:{len(v_px)}, py:{len(v_py)}, pz: {len(v_pz)} ")
        return (data_save, np.array(v_equen), np.array(v_vertex), v_pdg, v_px, v_py, v_pz)
    else:
        return (data_save, np.array(v_equen), np.array(v_vertex))
def GetProfile(h2d:ROOT.TH2D):
    h_profile = h2d.ProfileX()
    v_profile = []
    for i in range(h2d.GetNbinsX()):
       v_profile.append(h_profile.GetBinContent(i))
    v_profile = np.array(v_profile)
    v_profile = v_profile[v_profile!=0.]
    return v_profile
def PlotTimeProfile(h2d_sig:ROOT.TH2D, h2d_bkg:ROOT.TH2D):
    v_profile_sig = GetProfile(h2d_sig)
    v_profile_bkg = GetProfile(h2d_bkg)
    fig_profile_time = plt.figure("Profile_time")
    plt.plot(v_profile_bkg, label="background")
    plt.plot(v_profile_sig, label="signal")
    plt.xlabel("Time [ ns ]")
    return fig_profile_time
def TestnpzOuput(name_infile:str):
    dataset = np.load(name_infile, allow_pickle=True)
    # dataset_sig = dataset["sig"].item()
    # dataset_bkg = dataset["bkg"].item()
    # data_sig_NoweightE = dataset_sig["NoWeightE"]
    # data_bkg_NoweightE = dataset_bkg["NoWeightE"]
    # data_sig_weightE = dataset_sig["WeightE"]
    # data_bkg_weightE = dataset_bkg["WeightE"]
    data_sig_NoweightE = dataset["sig_NoWeightE"]
    data_bkg_NoweightE = dataset["bkg_NoWeightE"]
    data_sig_weightE = dataset["sig_WeightE"]
    data_bkg_weightE = dataset["bkg_WeightE"]
    data_sig_vertex = dataset["sig_vertex"]
    data_bkg_vertex = dataset["bkg_vertex"]
    data_sig_equen = dataset["sig_equen"]
    data_bkg_equen = dataset["bkg_equen"]
    print("######################Check Saved npz data#######################")
    if save_pdg:
        data_bkg_pdg = dataset["bkg_pdg"]
        data_bkg_px = dataset["bkg_px"]
        data_bkg_py = dataset["bkg_py"]
        data_bkg_pz = dataset["bkg_pz"]
        print(f"pdg length: {data_bkg_pdg.shape}")
        print(f"p shape-> px:{data_bkg_px.shape}, py:{data_bkg_py.shape}, pz:{data_bkg_pz.shape}")
        print(f"pdg: {data_bkg_pdg[:10]}")
        print(f"px: {data_bkg_px[:10]}")
        print(f"py: {data_bkg_py[:10]}")
        print(f"pz: {data_bkg_pz[:10]}")
    print(f"data_sig_NoWeightE: {data_sig_NoweightE.shape}")
    print(f"data_bkg_NoWeightE: {data_bkg_NoweightE.shape}")
    print(f"data_sig_WeightE:  {data_sig_weightE.shape}")
    print(f"data_bkg_WeightE:  {data_bkg_weightE.shape}")

if __name__ == "__main__":
    up_time = 1000
    down_time = 200
    binwidth = 10
    plot_result = False
    plot_single = False
    test_savefile = False
    save_pdg = False
    n_bins = (up_time+down_time)/binwidth
    h2d_time_sig = ROOT.TH2D("h_time_sig", "h_time_sig", int(n_bins), -down_time, up_time, int(n_bins), 0, 1.1)
    h2d_time_bkg = ROOT.TH2D("h_time_bkg", "h_time_bkg", int(n_bins), -down_time, up_time, int(n_bins), 0, 1.1)

    parser = argparse.ArgumentParser(description='DSNB sklearn dataset builder.')
    parser.add_argument("--inputdir", "-d", type=str, default="/workfs/exo/zepengli94/JUNO_DSNB", help="Raw data directory")
    parser.add_argument("--nstart_sig", "-ss", type=int, default=1, help="start num of signal file")
    parser.add_argument("--nstart_bkg", "-sb", type=int, default=1, help="start num of bkg file")
    parser.add_argument("--step", "-s", type=int, default=5, help="how many signal files to put into one output")
    parser.add_argument("--outfile", "-o", type=str, default="try.npz", help="name of outfile")
    arg = parser.parse_args()
    (filelist_sig, filelist_bkg) = GenFilesList(arg.inputdir, arg.nstart_sig, arg.nstart_bkg, arg.step)

    if test_savefile:
        TestnpzOuput(arg.outfile)
        exit()

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
    if save_pdg:
        (data_save_bkg, v_equen_bkg, v_vertex_bkg, v_pdg_bkg, v_px_bkg, v_py_bkg, v_pz_bkg) = LoadData(bkgchain, "bkg", h2d_time_bkg)
        print(f"check pdg:{len(v_pdg_bkg)}, p:{len(v_px_bkg), len(v_py_bkg), len(v_pz_bkg)}")
    else:
        (data_save_bkg, v_equen_bkg, v_vertex_bkg ) = LoadData(bkgchain, "bkg", h2d_time_bkg)
    # print(f"shape:   (data_save_sig:{data_save_sig.shape}), (v_equen_sig:{v_equen_sig}), (v_vertex_sig:{v_vertex_sig})")
    if save_pdg:
        np.savez(arg.outfile, sig_NoWeightE=data_save_sig["NoWeightE"], sig_WeightE=data_save_sig["WeightE"], bkg_NoWeightE=data_save_bkg["NoWeightE"], bkg_WeightE=data_save_bkg["WeightE"], sig_vertex=v_vertex_sig, bkg_vertex=v_vertex_bkg, sig_equen=v_equen_sig, bkg_equen=v_equen_bkg,
             bkg_pdg=np.array(v_pdg_bkg), bkg_px=np.array(v_px_bkg), bkg_py=np.array(v_py_bkg), bkg_pz=np.array(v_pz_bkg))
    else:
        np.savez(arg.outfile, sig_NoWeightE=data_save_sig["NoWeightE"], sig_WeightE=data_save_sig["WeightE"], bkg_NoWeightE=data_save_bkg["NoWeightE"], bkg_WeightE=data_save_bkg["WeightE"], sig_vertex=v_vertex_sig, bkg_vertex=v_vertex_bkg, sig_equen=v_equen_sig, bkg_equen=v_equen_bkg)
    if plot_result:
        # print(f"sig_data: {data_save_sig.shape},\n bkg_data: {data_save_bkg.shape}")
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

