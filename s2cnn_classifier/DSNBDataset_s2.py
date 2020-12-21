import uproot as up
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import ROOT
import math


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

    def CalcBin(self, pmtid):
        if pmtid > self.maxpmtid:
            print('Wrong PMT ID')
            return (0, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        xbin = int(phi * 128. / 360)  # np.where(self.thetas == theta)[0]
        ybin = int(theta * 128. / 180)
        # print(pmtid, x, y, z, theta, phi, xbin, ybin)
        return (xbin, ybin)


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


def Root2npz(mapfile, sig_dir, bkg_dir, outfile='', start_entries=0):
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
    batchsize = bkgchain.GetEntries()  # because the entries in bkg file is fewer than in sig files ,so we set the batch size as entries contained in one bkg file
    for batchentry in range(batchsize):
        # save charge and hittime to 3D array
        i_sig = start_entries + batchentry
        if batchentry % 10 == 0:
            print("processing batchentry : ", batchentry)
        if i_sig >= sigchain.GetEntries():
            continue
        sigchain.GetEntry(i_sig)
        bkgchain.GetEntry(batchentry)
        pmtids = sigchain.PMTID
        npes = sigchain.Charge
        hittime = sigchain.Time
        eqen = sigchain.Eqen
        x = sigchain.X
        y = sigchain.Y
        z = sigchain.Z
        if eqen <= 30 and eqen >= 11 and x ** 2 + y ** 2 + z ** 2 <= 256: #16m*16m
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
        if eqen <= 30 and eqen >= 11 and x ** 2 + y ** 2 + z ** 2 <= 256:  # 16m*16m
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

    print(f"n_events in {outfile} : {len(pmtinfos)}")
    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    if outfile == '':
        np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
    else:
        np.savez(outfile, pmtinfo=np.array(pmtinfos)[indices], eventtype=np.array(types)[indices],
                 eqen=np.array(eqen_batch)[indices], vertex=np.array(vertices)[indices])


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
    Root2npz(args.pmtmap, args.sigdir, args.bkgdir, args.outfile, args.StartEntries)
    # '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv', '/junofs/users/lizy/public/deeplearning/J19v1r0-Pre3/samples/train/eplus_ekin_0_10MeV/0/root_data/sample_0.root')
