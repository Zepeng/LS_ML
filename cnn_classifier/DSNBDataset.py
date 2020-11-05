import uproot as up
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import ROOT

class PMTIDMap():
    #The position of each pmt is stored in a root file different from the data file.
    NumPMT = 0
    pmtmap = {}
    maxpmtid = 17612
    thetaphi_dict = {}
    thetas = []

    #read the PMT map from the root file.
    def __init__(self, csvmap):
        pmtcsv = open(csvmap, 'r')
        for line in pmtcsv:
            pmt_instance = (line.split())
            self.pmtmap[str(pmt_instance[0])] = ( int(pmt_instance[0]), float(pmt_instance[1]), float(pmt_instance[2]), float(pmt_instance[3]), float(pmt_instance[4]), float(pmt_instance[5]))
        self.maxpmtid = len(self.pmtmap)

    def IdToPos(self, pmtid):
        return self.pmtmap[str(pmtid)]

    #Build a dictionary of the PMT location with theta and phi.
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
        ybin = np.where(self.thetas == theta)[0]
        xbin = np.where(self.thetaphi_dict[str(theta)] == phi)[0] + 112 - int(len(self.thetaphi_dict[str(theta)])/2)
        return(xbin, ybin)

def roottonpz(mapfile, rootfile, outfile='', eventtype='sig', batchsize = 100):
    # The csv file of PMT map must have the same tag as the MC production.
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    uptree = up.open(rootfile)['psdtree']
    pmtids = uptree.array('PMTID')
    npes   = uptree.array('Charge')
    hittime= uptree.array('Time')
    edeps  = uptree.array('Edep')
    nbatches = int(len(pmtids)/batchsize)
    if len(pmtids) > batchsize*nbatches:
        nbatches += 1

    for batch in range(nbatches):
        pmtinfos = []
        types = []
        edep_batch = []
        for batchentry in range(batchsize):
            #save charge and hittime to 3D array
            event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
            i = batchsize*batch + batchentry
            if i >= len(pmtids):
                continue
            for j in range(len(pmtids[i])):
                (xbin, ybin) = pmtmap.CalcBin(pmtids[i][j])
                event2dimg[0, xbin, ybin] += npes[i][j]
                event2dimg[1, xbin, ybin] += hittime[i][j]
            pmtinfos.append(event2dimg)
            if eventtype == 'sig':
                types.append(1)
            else:
                types.append(0)
            edep_batch.append(edeps[i])

        if outfile == '':
            np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
        else:
            np.savez(outfile + str(batch) + 'npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types), edep=np.array(edep_batch))

def chaintonpz(mapfile, sig_dir, bkg_dir, outfile='', batch_num = 100, batchsize = 500):
    # The csv file of PMT map must have the same tag as the MC production.
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    sigchain = ROOT.TChain('psdtree')
    sigchain.Add('%s/*root' % sig_dir)
    bkgchain = ROOT.TChain('psdtree')
    bkgchain.Add('%s/*root' % bkg_dir)
    #pmtids = uptree.array('PMTID')
    #npes   = uptree.array('Charge')
    #hittime= uptree.array('Time')
    #edeps  = uptree.array('Edep')
    #nbatches = int(len(pmtids)/batchsize)

    pmtinfos = []
    types = []
    edep_batch = []
    for batchentry in range(int(batchsize/2)):
        #save charge and hittime to 3D array
        i = int*(batchsize/2)*batch_num + batchentry
        if i >= sigchain.GetEntries() or i >= bkgchain.GetEntries():
            continue
        sigchain.GetEntry(i)
        bkgchain.GetEntry(i)
        pmtids = sigchain.PMTID
        npes = sigchain.Charge
        hittime = sigchain.Time
        edep = sigchain.Edep
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            event2dimg[1, xbin, ybin] += hittime[j]
        pmtinfos.append(event2dimg)
        types.append(1)
        edep_batch.append(edep)
        pmtids = bkgchain.PMTID
        npes = bkgchain.Charge
        hittime = bkgchain.Time
        edep = bkgchain.Edep
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            event2dimg[1, xbin, ybin] += hittime[j]
        pmtinfos.append(event2dimg)
        types.append(0)
        edep_batch.append(edep)

    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    if outfile == '':
        np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
    else:
        np.savez(outfile , pmtinfo=np.array(pmtinfos)[indices], eventtype=np.array(types)[indices], edep=np.array(edep_batch)[indices])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JUNO ML dataset builder.')
    parser.add_argument('--pmtmap', type=str, help='csc file of PMT map in JUNO.')
    parser.add_argument('--sigdir', '-s', type=str, help='Input root file.')
    parser.add_argument('--bkgdir', '-b', type=str, help='Output root file.')
    parser.add_argument('--eventtype', '-t', type=str, help='Event type.')
    parser.add_argument('--outfile', '-o', type=str, help='Output root file.')
    parser.add_argument('--batch', '-n', type=int, help='Batch number.')
    args = parser.parse_args()
    #chaintonpz(args.pmtmap, args.infile, args.outfile, args.eventtype)
    chaintonpz(args.pmtmap, args.sigdir, args.bkgdir, args.outfile, batch_num = args.batch)
    #'/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J19v1r1-Pre4/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv', '/junofs/users/lizy/public/deeplearning/J19v1r0-Pre3/samples/train/eplus_ekin_0_10MeV/0/root_data/sample_0.root')
