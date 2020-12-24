import uproot as up
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import ROOT
import math
from scipy.interpolate import RegularGridInterpolator
import pickle
from scipy.interpolate import griddata


class PMTIDMap():
    # The position of each pmt is stored in a root file different from the data file.
    NumPMT = 0
    pmtmap = {}
    maxpmtid = 17612
    thetaphi_dict = {}
    thetas = []

    # read the PMT map from the root file.
    def __init__(self, csvmap):
        pmtcsv = open(csvmap, 'r')
        for line in pmtcsv:
            pmt_instance = (line.split())
            self.pmtmap[str(pmt_instance[0])] = (
                int(pmt_instance[0]), float(pmt_instance[1]), float(pmt_instance[2]), float(pmt_instance[3]),
                float(pmt_instance[4]), float(pmt_instance[5]))
        self.maxpmtid = len(self.pmtmap)

    def IdToPos(self, pmtid):
        return self.pmtmap[str(pmtid)]

    # Build a dictionary of the PMT location with theta and phi.
    def CalcDict(self):
        thetas = []
        thetaphi_dict = {}
        thetaphis = []
        for key in self.pmtmap:
            (pmtid, x, y, z, theta, phi) = self.pmtmap[key]
            if theta not in thetas:
                thetas.append(theta)
            thetaphis.append((theta, phi))
        for theta in thetas:
            thetaphi_dict[str(theta)] = []
        for (theta, phi) in thetaphis:
            thetaphi_dict[str(theta)].append(phi)
        for key in thetaphi_dict:
            thetaphi_dict[key] = np.sort(thetaphi_dict[key])
        self.thetaphi_dict = thetaphi_dict
        self.thetas = np.sort(thetas)

    def CalcThetaPhiGrid(self):
        thetas = set()
        phis = set()
        for key in self.pmtmap:
            (pmtid, x, y, z, theta, phi) = self.pmtmap[key]
            thetas.add(theta)
            phis.add(phi)
        thetas = (np.array(list(thetas)) - 90.) * np.pi / 180.
        phis = (np.array(list(phis)) - 180.) * np.pi / 180.
        self.thetas = np.sort(thetas)
        self.phis = np.sort(phis)
        print(f"len(thetas) : {len(self.thetas)}, len(phis) : {len(self.phis)}")
        print(f"thetas -- max: {np.max(self.thetas)}, min: {np.min(self.thetas)}")
        print(f"phis   -- max: {np.max(self.phis)}, min: {np.min(self.phis)}")

    def CalcThetaPhiPmtPoints(self):
        self.points_thetaphi = []  # NOT using set here is because we believe that in the pmt map file , there is no duplicate points
        self.thetas_set = set()
        for key in self.pmtmap:
            (pmtid, x, y, z, theta, phi) = self.pmtmap[key]
            theta = (theta - 90.) * np.pi / 180.
            phi   = (phi - 180.) * np.pi / 180.
            self.points_thetaphi.append((theta, phi))
            self.thetas_set.add(theta)
        self.thetas_set = np.array(list(self.thetas_set))
        self.points_thetaphi = np.array(list(self.points_thetaphi))
        self.thetas = self.points_thetaphi[:, 0]
        self.phis = self.points_thetaphi[:, 1]
        # Get PMTids in the border of theta-phi planar

        self.pmtid_border_right = np.zeros((len(self.thetas_set)), dtype=np.int)  # because theta of pmts is aligned(map to self.thetas_set which has no depulicate theta)
        self.pmtid_border_left = np.zeros((len(self.thetas_set)), dtype=np.int)
        for i, theta in enumerate(self.thetas_set):
            indices_SameTheta = np.where(self.points_thetaphi[:, 0] == theta)[0]
            i_max_phi_SameTheta = indices_SameTheta[np.argmax(self.points_thetaphi[indices_SameTheta, 1])]
            i_min_phi_SameTheta = indices_SameTheta[np.argmin(self.points_thetaphi[indices_SameTheta, 1])]
            self.pmtid_border_left[i] = i_min_phi_SameTheta
            self.pmtid_border_right[i]= i_max_phi_SameTheta
        # print(self.thetas_set, "max index :", self.pmtid_border_right, "min index : ", self.pmtid_border_left)

    def CalcThetaPhiSquareGrid(self, n_grid: int):
        self.thetas = np.linspace(-np.pi * 0.5, np.pi * 0.5, n_grid)
        self.phis = np.linspace(-np.pi, np.pi, n_grid)

    def CalcBin(self, pmtid):
        if pmtid > self.maxpmtid:
            print('Wrong PMT ID')
            return (0, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        xbin = int(phi * 128. / 360)
        ybin = int(theta * 128. / 180)
        # print(pmtid, x, y, z, theta, phi, xbin, ybin)
        return (xbin, ybin)

    def CalcBin_ThetaPhiImage(self, pmtid):
        if pmtid > self.maxpmtid:
            print('Wrong PMT ID')
            return (0, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        # Using xbin and ybin, PMTs can be mapped into a image, which like a oval
        xbin = np.where(self.thetas == theta)[0]
        # xbin = np.where(self.thetaphi_dict[str(theta)] == phi)[0] + 112 - int(len(self.thetaphi_dict[str(theta)])/2)# When the theta is close to pi/2, we got the max length of phis
        ybin = np.where(self.phis == phi)[0]
        # print((xbin, ybin))
        return (xbin, ybin)


def save2root(outfile, pmtinfos, types, eqen_batch, vertices):
    n_evts = len(pmtinfos)
    print(f"n_events in {outfile} : {n_evts}")
    if outfile == '':
        outfile = "data_fake.root"
    f = ROOT.TFile(outfile, "recreate")
    t = ROOT.TTree("data_tree", "data_tree")
    pmtinfos2tree = np.array(pmtinfos[0], dtype=np.float64)
    types2tree = np.array(types[0], dtype=np.int32)
    eqen2tree = np.array(eqen_batch[0], dtype=np.float64)
    vertices2tree = np.array(vertices[0], dtype=np.float64)
    shape_pmtinfos2tree = pmtinfos2tree.shape
    # t.Branch("pmtinfos", pmtinfos2tree, "pmtinfos["+str(shape_pmtinfos2tree[0])+"]["+str(shape_pmtinfos2tree[1])+"]["+str(shape_pmtinfos2tree[2])+"]/F")
    t.Branch("eventtype", types2tree, "eventtype/I")
    # t.Branch("eqen", eqen2tree, "eqen/F")
    # t.Branch("vertex", vertices2tree, "vertex[3]/F")
    for i_evt in range(n_evts):
        pmtinfos2tree = np.array(pmtinfos[i_evt], dtype=np.float64)
        types2tree = np.array(types[i_evt], dtype=np.int32)
        eqen2tree = eqen_batch[i_evt]
        vertices2tree = vertices[i_evt]
        t.Fill()
    f.Write()
    f.Close()


def save2npz(outfile, pmtinfos, types, eqen_batch, vertices):
    if outfile == '':
        np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
    else:
        np.savez(outfile, pmtinfo=pmtinfos, eventtype=types,
                 eqen=eqen_batch, vertex=vertices)


def roottonpz(mapfile, rootfile, outfile='', eventtype='sig', batchsize=100):
    # The csv file of PMT map must have the same tag as the MC production.
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    uptree = up.open(rootfile)['psdtree']
    pmtids = uptree.array('PMTID')
    npes = uptree.array('Charge')
    hittime = uptree.array('Time')
    eqens = uptree.array('eqen')
    nbatches = int(len(pmtids) / batchsize)
    if len(pmtids) > batchsize * nbatches:
        nbatches += 1

    for batch in range(nbatches):
        pmtinfos = []
        types = []
        eqen_batch = []
        for batchentry in range(batchsize):
            # save charge and hittime to 3D array
            event2dimg = np.zeros((2, 128, 128), dtype=np.float16)
            i = batchsize * batch + batchentry
            if i >= len(pmtids):
                continue
            for j in range(len(pmtids[i])):
                (xbin, ybin) = pmtmap.CalcBin(pmtids[i][j])
                event2dimg[0, xbin, ybin] += npes[i][j]
                if event2dimg[1, xbin, ybin] == 0:
                    event2dimg[1, xbin, ybin]
                else:
                    event2dimg[1, xbin, ybin] = min(hittime[i][j], event2dimg[1, xbin, ybin])
            pmtinfos.append(event2dimg)
            if eventtype == 'sig':
                types.append(1)
            else:
                types.append(0)
            eqen_batch.append(eqens[i])

        if outfile == '':
            np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
        else:
            np.savez(outfile + str(batch) + 'npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types),
                     eqen=np.array(eqen_batch))


def chaintonpz(mapfile, sig_dir, bkg_dir, outfile='', batch_num=100, batchsize=500):
    # The csv file of PMT map must have the same tag as the MC production.
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    sigchain = ROOT.TChain('psdtree')
    # sigchain.Add('%s/*00001.root' % sig_dir)
    sigchain.Add(sig_dir)

    bkgchain = ROOT.TChain('psdtree')
    # bkgchain.Add('%s/*00001.root' % bkg_dir)
    bkgchain.Add(bkg_dir)

    print("Load Raw Data Successfully!!")
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    for batchentry in range(int(batchsize / 2)):
        # save charge and hittime to 3D array
        i = int(batchsize / 2) * batch_num + batchentry
        if batchentry % 10 == 0:
            print("processing batchentry : ", batchentry)
        if i >= sigchain.GetEntries() or i >= bkgchain.GetEntries():
            continue
        sigchain.GetEntry(i)
        bkgchain.GetEntry(i)
        pmtids = sigchain.PMTID
        npes = sigchain.Charge
        hittime = sigchain.Time
        eqen = sigchain.Eqen
        # print("pmtids:   ", len(pmtids)) # 24154
        # print("hittime:  ", len(hittime)) # 24154
        # print("npes:   ", len(eqen)) # 1
        event2dimg = np.zeros((2, 128, 128), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            if event2dimg[1, xbin, ybin] < 0.1:
                event2dimg[1, xbin, ybin] = hittime[j]
            else:
                event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])
        pmtinfos.append(event2dimg)
        types.append(1)
        eqen_batch.append(eqen)
        vertices.append([sigchain.X, sigchain.Y, sigchain.Z])
        pmtids = bkgchain.PMTID
        npes = bkgchain.Charge
        hittime = bkgchain.Time
        eqen = bkgchain.Eqen
        event2dimg = np.zeros((2, 128, 128), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            if event2dimg[1, xbin, ybin] < 0.1:
                event2dimg[1, xbin, ybin] = hittime[j]
            else:
                event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])
        pmtinfos.append(event2dimg)
        types.append(0)
        vertices.append([bkgchain.X, bkgchain.Y, bkgchain.Z])
        eqen_batch.append(eqen)

    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    if outfile == '':
        np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
    else:
        np.savez(outfile, pmtinfo=np.array(pmtinfos)[indices], eventtype=np.array(types)[indices],
                 eqen=np.array(eqen_batch)[indices], vertex=np.array(vertices)[indices])


def GetS2cnnData(mapfile, sig_dir, bkg_dir, outfile='', start_entries=0):
    # The csv file of PMT map must have the same tag as the MC production.
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    sigchain = ROOT.TChain('psdtree')
    # sigchain.Add('%s/*00001.root' % sig_dir)
    sigchain.Add(sig_dir)

    bkgchain = ROOT.TChain('psdtree')
    # bkgchain.Add('%s/*00001.root' % bkg_dir)
    bkgchain.Add(bkg_dir)

    print("Load Raw Data Successfully!!")
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    # batchsize = bkgchain.GetEntries()  # because the entries in bkg file is fewer than in sig files ,so we set the batch size as entries contained in one bkg file
    batchsize = 270  # because the entries in bkg file is fewer than in sig files ,so we set the batch size as entries contained in one bkg file
    for batchentry in range(batchsize):
        # save charge and hittime to 3D array
        i_sig = start_entries + batchentry
        if batchentry % 10 == 0:
            print("processing batchentry : ", batchentry)
        if i_sig >= sigchain.GetEntries() or batchentry > bkgchain.GetEntries():
            print("the batchsize is greater than bkgchain's  Entries , so we decided not to save the file")
            exit(1)
        sigchain.GetEntry(i_sig)
        bkgchain.GetEntry(batchentry)
        pmtids = sigchain.PMTID
        npes = sigchain.Charge
        hittime = sigchain.Time
        eqen = sigchain.Eqen
        x = sigchain.X
        y = sigchain.Y
        z = sigchain.Z
        # print(f"sig   E:{eqen},R:{x ** 2 + y ** 2 + z ** 2}")
        # if eqen <= 30 and eqen >= 11 and x ** 2 + y ** 2 + z ** 2 <= 256000000:  # 16m*16m
        # print("pmtids:   ", len(pmtids)) # 24154
        # print("hittime:  ", len(hittime)) # 24154
        # print("npes:   ", len(eqen)) # 1
        event2dimg = np.zeros((2, 128, 128), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            if event2dimg[1, xbin, ybin] < 0.1:
                event2dimg[1, xbin, ybin] = hittime[j]
            else:
                event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])
        pmtinfos.append(event2dimg)
        types.append(1)
        eqen_batch.append(eqen)
        vertices.append([sigchain.X, sigchain.Y, sigchain.Z])
        pmtids = bkgchain.PMTID
        npes = bkgchain.Charge
        hittime = bkgchain.Time
        eqen = bkgchain.Eqen
        x = bkgchain.X
        y = bkgchain.Y
        z = bkgchain.Z
        # print(f"bkg   E:{eqen},R:{x ** 2 + y ** 2 + z ** 2}")
        # if eqen <= 30 and eqen >= 11 and x ** 2 + y ** 2 + z ** 2 <= 256000000:  # 16m*16m
        event2dimg = np.zeros((2, 128, 128), dtype=np.float16)

        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            if event2dimg[1, xbin, ybin] < 0.1:
                event2dimg[1, xbin, ybin] = hittime[j]
            else:
                event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])
        pmtinfos.append(event2dimg)
        types.append(0)
        vertices.append([bkgchain.X, bkgchain.Y, bkgchain.Z])
        eqen_batch.append(eqen)

    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    pmtinfos = np.array(pmtinfos)[indices]
    types = np.array(types, dtype=np.int32)[indices]
    eqen_batch = np.array(eqen_batch)[indices]
    vertices = np.array(vertices)[indices]

    save2npz(outfile, pmtinfos, types, eqen_batch, vertices)
    # save2root(outfile, pmtinfos, types, eqen_batch, vertices)
    # (pmtinfos_load , types_load, eqen_load, vertex_load) = LoadRoot(outfile)
    # print("pmtinfos load:",pmtinfos_load)
    # print("pmtinfos     :", pmtinfos)
    # print("types_save: ",types)
    # print("types_load: ",types_load)


def PlotRawSignal(event2dimage, x, y, z):
    fig_hittime = plt.figure("hittime")
    ax = fig_hittime.add_subplot(111, projection='3d')
    indices = (event2dimage[1] != 0.)
    # print(indices)
    img_hittime = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage[1][indices], cmap=plt.hot(), s=1)
    # img_hittime = ax.scatter(x, y, z, c=event2dimage[1], cmap=plt.hot(), s=1)
    fig_hittime.colorbar(img_hittime)

    fig_eqen = plt.figure("eqen")
    ax = fig_eqen.add_subplot(111, projection='3d')
    indices = (event2dimage[0] != 0)
    img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage[0][indices], cmap=plt.hot(), s=1)
    # img_eqen = ax.scatter(x, y, z, c=event2dimage[0], cmap=plt.hot(), s=1)
    fig_eqen.colorbar(img_eqen)


def PlotIntepSignal(event2dimage_intep, x, y, z):
    fig_hittime = plt.figure("hittime_intep")
    ax = fig_hittime.add_subplot(111, projection='3d')
    # indices = (event2dimage_intep[1] != 0.)
    # img_hittime = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[1][indices], cmap=plt.hot(), s=1)
    img_hittime = ax.scatter(x, y, z, c=event2dimage_intep[1], cmap=plt.hot(), s=1)
    fig_hittime.colorbar(img_hittime)

    fig_eqen = plt.figure("eqen_intep")
    ax = fig_eqen.add_subplot(111, projection='3d')
    # indices = (event2dimage_intep[0] != 0)
    # img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[0][indices], cmap=plt.hot(), s=1)
    img_eqen = ax.scatter(x, y, z, c=event2dimage_intep[0], cmap=plt.hot(), s=1)
    fig_eqen.colorbar(img_eqen)

def PlotSigPlanar(thetas, phis, sig_r2, name_fig:str="sig_planar"):
    fig_planar = plt.figure(name_fig)
    ax = fig_planar.add_subplot(111)
    img_planar = ax.scatter(phis, thetas, c=sig_r2, cmap=plt.hot(), s=1)
    plt.xlabel("phi")
    plt.ylabel("theta")
    fig_planar.colorbar(img_planar)

def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)  # longitude as same as phi
    xy2 = x ** 2 + y ** 2
    lat = np.arctan2(z, np.sqrt(xy2))  # latitude as same as theta
    return lat, long

def Expand2dPlanarBySphere(thetas, phis, sig_r2:np.ndarray, pmtmap:PMTIDMap):
    sig_r2_expand = sig_r2.copy()
    thetas_expand = thetas.copy()
    phis_expand   = phis.copy()
    plot_expanded_planar:bool = False

    # PlotSigPlanar(thetas, phis, sig_r2, name_fig="raw_sig")
    # sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2[pmtmap.pmtid_border_right]))
    # thetas_expand = np.concatenate((thetas_expand, pmtmap.thetas_set))
    # phis_expand   = np.concatenate((phis_expand  , phis[pmtmap.pmtid_border_right]-2*np.pi))
    #
    # sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2[pmtmap.pmtid_border_left]))
    # thetas_expand = np.concatenate((thetas_expand, pmtmap.thetas_set))
    # phis_expand   = np.concatenate((phis_expand  , phis[pmtmap.pmtid_border_left]+2*np.pi))
    # PlotSigPlanar(thetas_expand, phis_expand, sig_r2_expand, name_fig="add left edge")

    sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2))
    thetas_expand = np.concatenate((thetas_expand, thetas))
    phis_expand   = np.concatenate((phis_expand  , phis-2*np.pi))

    sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2))
    thetas_expand = np.concatenate((thetas_expand, thetas))
    phis_expand   = np.concatenate((phis_expand  , phis+2*np.pi))
    if plot_expanded_planar:
        PlotSigPlanar(thetas, phis, sig_r2, name_fig="raw_sig")
        PlotSigPlanar(thetas_expand, phis_expand, sig_r2_expand, name_fig="expanded sig")
        plt.show()
        exit()
    return (thetas_expand, phis_expand, sig_r2_expand)


def interp_pmt2mesh(sig_r2, thetas, phis, V, pmtmap, method="linear", do_calcgrid=False, dtype=np.float32):
    ele, azi = xyz2latlong(V)
    check_interp_range:bool = False
    if check_interp_range:
        print("###########Checking whether raw data range is matched with the interplotation range############")
        print(f"ele range: {np.min(ele)}--{np.max(ele)}")
        print(f"azi range: {np.min(azi)}--{np.max(azi)}")
        print(f"thetas -- max: {np.max(thetas)}, min: {np.min(thetas)}")
        print(f"phis   -- max: {np.max(phis)}, min: {np.min(phis)}")
        s2 = np.array([ele, azi]).T
        print("s2:   ", s2.shape)
        print("sig_r2:   ", sig_r2.shape)
        print("(theta, phis):  ", (thetas[:10], phis[:10]))
        print("###############################################################################################")
    (thetas, phis, sig_r2) = Expand2dPlanarBySphere(thetas, phis, sig_r2, pmtmap)
    # sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1) #aims to add sig_r2 images one column
    if do_calcgrid:
        intp = RegularGridInterpolator((thetas, phis), sig_r2, method=method)
        sig_s2 = intp(s2).astype(dtype)
    else:
        indices = (sig_r2 != 0.)
        sig_s2 = griddata((thetas[indices], phis[indices]), sig_r2[indices], (ele, azi), method=method)
    # print("sig_s2 : ", sig_s2, "shape: ", sig_s2.shape)  # sig_s2 :  (642,)
    return sig_s2


def GetugscnnData(mapfile, sig_dir, bkg_dir, outfile='', start_entries=0):
    # The csv file of PMT map must have the same tag as the MC production.
    plot_result_sig: bool = False
    plot_result_bkg: bool = True
    max_n_points_grid: bool = True
    do_calcgrid: bool = False
    pmtmap = PMTIDMap(mapfile)
    file_mesh = "/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/ugscnn/mesh_files/icosphere_5.pkl"
    p = pickle.load(open(file_mesh, "rb"))
    V = p['V']
    # pmtmap.CalcDict()
    n_grid = 128
    if max_n_points_grid:
        if do_calcgrid:
            pmtmap.CalcThetaPhiGrid()
        else:
            pmtmap.CalcThetaPhiPmtPoints()
    else:
        pmtmap.CalcThetaPhiSquareGrid(n_grid)
    if plot_result_sig or plot_result_bkg:
        if max_n_points_grid:
            if do_calcgrid:
                PHIS, THETAS = np.meshgrid(pmtmap.phis,
                                           pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            else:
                PHIS, THETAS = pmtmap.phis, pmtmap.thetas
        else:
            PHIS, THETAS = np.meshgrid(pmtmap.phis,
                                       pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            # print(f"thetas:{pmtmap.thetas}")
            # print(f"grid(thetas): {THETAS}")
        x_raw_grid = np.cos(THETAS) * np.cos(PHIS)
        y_raw_grid = np.cos(THETAS) * np.sin(PHIS)
        z_raw_grid = np.sin(THETAS)
        x_V, y_V, z_V = V[:, 0], V[:, 1], V[:, 2]

    sigchain = ROOT.TChain('psdtree')
    # sigchain.Add('%s/*00001.root' % sig_dir)
    sigchain.Add(sig_dir)

    bkgchain = ROOT.TChain('psdtree')
    # bkgchain.Add('%s/*00001.root' % bkg_dir)
    bkgchain.Add(bkg_dir)

    print("Load Raw Data Successfully!!")
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    # batchsize = bkgchain.GetEntries()  # because the entries in bkg file is fewer than in sig files ,so we set the batch size as entries contained in one bkg file
    batchsize = 270
    for batchentry in range(batchsize):
        # save charge and hittime to 3D array
        i_sig = start_entries + batchentry
        if batchentry % 10 == 0:
            print("processing batchentry : ", batchentry)
        if i_sig >= sigchain.GetEntries() or batchentry > bkgchain.GetEntries():
            print("the batchsize is greater than bkgchain's  Entries , so we decided not to save the file")
            exit(1)
        sigchain.GetEntry(i_sig)
        bkgchain.GetEntry(batchentry)
        pmtids = sigchain.PMTID
        npes = sigchain.Charge
        hittime = sigchain.Time
        eqen = sigchain.Eqen
        x = sigchain.X
        y = sigchain.Y
        z = sigchain.Z

        pmtids = np.array(pmtids)
        # print(f"pmtids.shape {len(pmtids)}")
        # print(f"sig   E:{eqen},R:{x ** 2 + y ** 2 + z ** 2}")
        # if eqen <= 30 and eqen >= 11 and x ** 2 + y ** 2 + z ** 2 <= 256000000:  # 16m*16m
        # print("pmtids:   ", len(pmtids)) # 24154
        # print("hittime:  ", len(hittime)) # 24154
        # print("npes:   ", len(eqen)) # 1

        pmtinfos = []
        # save charge and hittime to 3D array
        # event2dimg = np.zeros((2, 225, 124), dtype=np.float16)
        if do_calcgrid == False:
            event2dimg = np.zeros((2, len(pmtmap.thetas)), dtype=np.float16)
        else:
            event2dimg = np.zeros((2, len(pmtmap.thetas), len(pmtmap.phis)), dtype=np.float16)

        # event2dimg = np.zeros((2, n_grid, n_grid), dtype=np.float16)
        event2dimg_interp = np.zeros((2, len(V)), dtype=np.float32)
        for j in range(len(pmtids)):
            if pmtids[j] > 17612:
                continue
            if max_n_points_grid:
                if do_calcgrid:
                    (xbin, ybin) = pmtmap.CalcBin_ThetaPhiImage(pmtids[j])
                else:
                    i_pmt = pmtids[j]
            else:
                (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            # if ybin>124:
            #     print(pmtids[i][j])
            if do_calcgrid == False:
                event2dimg[0, i_pmt] += npes[j]
                # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
                if event2dimg[1, i_pmt] == 0:
                    event2dimg[1, i_pmt] = hittime[j]
                else:
                    event2dimg[1, i_pmt] = min(hittime[j], event2dimg[1, i_pmt])
            else:
                event2dimg[0, xbin, ybin] += npes[j]
                # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
                if event2dimg[1, xbin, ybin] == 0:
                    event2dimg[1, xbin, ybin] = hittime[j]
                else:
                    event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])

        event2dimg_interp[0] = interp_pmt2mesh(event2dimg[0], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="linear")
        event2dimg_interp[1] = interp_pmt2mesh(event2dimg[1], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="nearest")
        if plot_result_sig:
            PlotRawSignal(event2dimg, x_raw_grid, y_raw_grid, z_raw_grid)
            PlotIntepSignal(event2dimg_interp, x_V, y_V, z_V)
            plt.show()
            exit()
        pmtinfos.append(event2dimg_interp)
        types.append(1)
        eqen_batch.append(eqen)
        vertices.append([sigchain.X, sigchain.Y, sigchain.Z])

        pmtids = bkgchain.PMTID
        npes = bkgchain.Charge
        hittime = bkgchain.Time
        eqen = bkgchain.Eqen
        x = bkgchain.X
        y = bkgchain.Y
        z = bkgchain.Z

        if do_calcgrid == False:
            event2dimg = np.zeros((2, len(pmtmap.thetas)), dtype=np.float16)
        else:
            event2dimg = np.zeros((2, len(pmtmap.thetas), len(pmtmap.phis)), dtype=np.float16)

        # event2dimg = np.zeros((2, n_grid, n_grid), dtype=np.float16)
        event2dimg_interp = np.zeros((2, len(V)), dtype=np.float32)
        for j in range(len(pmtids)):
            if pmtids[j] > 17612:
                continue
            if max_n_points_grid:
                if do_calcgrid:
                    (xbin, ybin) = pmtmap.CalcBin_ThetaPhiImage(pmtids[j])
                else:
                    i_pmt = pmtids[j]
            else:
                (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            # if ybin>124:
            #     print(pmtids[i][j])
            if do_calcgrid == False:
                event2dimg[0, i_pmt] += npes[j]
                # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
                if event2dimg[1, i_pmt] == 0:
                    event2dimg[1, i_pmt] = hittime[j]
                else:
                    event2dimg[1, i_pmt] = min(hittime[j], event2dimg[1, i_pmt])
            else:
                event2dimg[0, xbin, ybin] += npes[j]
                # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
                if event2dimg[1, xbin, ybin] == 0:
                    event2dimg[1, xbin, ybin] = hittime[j]
                else:
                    event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])

        event2dimg_interp[0] = interp_pmt2mesh(event2dimg[0], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="linear")
        event2dimg_interp[1] = interp_pmt2mesh(event2dimg[1], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="nearest")
        if plot_result_bkg:
            PlotRawSignal(event2dimg, x_raw_grid, y_raw_grid, z_raw_grid)
            PlotIntepSignal(event2dimg_interp, x_V, y_V, z_V)
            plt.show()
            exit()
        pmtinfos.append(event2dimg_interp)
        types.append(0)
        vertices.append([bkgchain.X, bkgchain.Y, bkgchain.Z])
        eqen_batch.append(eqen)

    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    pmtinfos = np.array(pmtinfos)[indices]
    types = np.array(types, dtype=np.int32)[indices]
    eqen_batch = np.array(eqen_batch)[indices]
    vertices = np.array(vertices)[indices]

    save2npz(outfile, pmtinfos, types, eqen_batch, vertices)


def LoadRoot(infile):
    # t.Branch("pmtinfos", pmtinfos2tree, "pmtinfos["+str(shape_pmtinfos2tree[0])+"]["+str(shape_pmtinfos2tree[1])+"]["+str(shape_pmtinfos2tree[2])+"]/D")
    # t.Branch("eventtype", types2tree, "eventtype/I")
    # t.Branch("eqen", eqen2tree, "eqen/D")
    # t.Branch("vertex", vertices2tree, "vertex[3]/D")
    data_chain = ROOT.TChain("data_tree")
    data_chain.Add(infile)
    pmtinfos = []
    eventtype = []
    eqen_batch = []
    vertices = []
    for i in range(data_chain.GetEntries()):
        data_chain.GetEntry(i)
        # pmtinfos.append(np.reshape(data_chain.pmtinfos,(2,128,128)))
        eventtype.append(data_chain.eventtype)
        # eqen_batch.append(data_chain.eqen)
        # vertices.append(data_chain.vertex)
    pmtinfos = np.array(pmtinfos)
    eventtype = np.array(eventtype)
    eqen_batch = np.array(eqen_batch)
    vertices = np.array(vertices)
    return (pmtinfos, eventtype, eqen_batch, vertices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JUNO ML dataset builder.')
    parser.add_argument('--pmtmap', type=str, help='csc file of PMT map in JUNO.')
    parser.add_argument('--sigdir', '-s', type=str, help='Input root file.')
    parser.add_argument('--bkgdir', '-b', type=str, help='Output root file.')
    parser.add_argument('--eventtype', '-t', type=str, help='Event type.')
    parser.add_argument('--outfile', '-o', type=str, help='Output root file.')
    parser.add_argument('--batch', '-n', type=int, help='Batch number.')
    parser.add_argument('--StartEntries', '-e', type=int, help='Start Entry of sig_file.')
    args = parser.parse_args()
    # chaintonpz(args.pmtmap, args.infile, args.outfile, args.eventtype)
    # chaintonpz(args.pmtmap, args.sigdir, args.bkgdir, args.outfile, batch_num = args.batch)
    # GetS2cnnData(args.pmtmap, args.sigdir, args.bkgdir, args.outfile, args.StartEntries)
    GetugscnnData(args.pmtmap, args.sigdir, args.bkgdir, args.outfile, args.StartEntries)
    # '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv', '/junofs/users/lizy/public/deeplearning/J19v1r0-Pre3/samples/train/eplus_ekin_0_10MeV/0/root_data/sample_0.root')
