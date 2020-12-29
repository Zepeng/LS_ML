import uproot as up
import ROOT
import numpy as np
import argparse
import matplotlib.pyplot as plt
def LoadData(tchain:ROOT.TChain, name_type:str, h2d:ROOT.TH2D):
    bins_hist = range(-down_time, up_time, binwidth)
    for i in range(tchain.GetEntries()):
    # for i in range(1,10):
        # plt.figure(name_type+str(i))
        tchain.GetEntry(i)
        pmtids = tchain.PMTID
        npes = tchain.Charge
        hittime = tchain.Time
        eqen = tchain.Eqen
        hist, bin_edges = np.histogram(hittime, bins=bins_hist)
        hist = hist/hist.max()
        for j_time in range(len(hist)):
            h2d.Fill(bin_edges[j_time], hist[j_time])
        # plt.plot(hist, label=name_type+str(i))
if __name__ == "__main__":
    up_time = 1000
    down_time = 200
    binwidth = 2
    n_bins = (up_time+down_time)/binwidth
    h2d_time_sig = ROOT.TH2D("h_time_sig", "h_time_sig", int(n_bins), -down_time, up_time, int(n_bins), 0, 1.1)
    h2d_time_bkg = ROOT.TH2D("h_time_bkg", "h_time_bkg", int(n_bins), -down_time, up_time, int(n_bins), 0, 1.1)

    parser = argparse.ArgumentParser(description='DSNB sklearn dataset builder.')
    name_sig_files = "/workfs/exo/zepengli94/JUNO_DSNB/DSNB/data/dsnb_00000[1-3].root"
    name_bkg_files = "/workfs/exo/zepengli94/JUNO_DSNB/AtmNu/data/atm_00000[1-9].root"
    sigchain = ROOT.TChain('psdtree')
    sigchain.Add(name_sig_files)
    # sigchain.Add('%s/*root' % sig_dir)
    bkgchain = ROOT.TChain('psdtree')
    bkgchain.Add(name_bkg_files)
    # bkgchain.Add('%s/*root' % bkg_dir)
    LoadData(sigchain, "sig", h2d_time_sig)
    LoadData(bkgchain, "bkg", h2d_time_bkg)
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
    # plt.legend()
    # plt.show()

