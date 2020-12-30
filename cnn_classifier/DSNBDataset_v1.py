import uproot as up
import numpy as np
import scipy
#import matplotlib.pyplot as plt
import argparse

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
    eqens  = uptree.array('eqen')
    nbatches = int(len(pmtids)/batchsize)
    if len(pmtids) > batchsize*nbatches:
        nbatches += 1

    for batch in range(nbatches):
        pmtinfos = []
        types = []
        eqen_batch = []
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
            eqen_batch.append(eqens[i])

        if outfile == '':
            np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
        else:
            np.savez(outfile + str(batch) + 'npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types), eqen=np.array(eqen_batch))

def chaintonpz(mapfile, sig_dir, bkg_dir, outfile='', batch_num = 100, batchsize = 500):
    # The csv file of PMT map must have the same tag as the MC production.
    print('Start processing')
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    print('Read ROOT files.')
    sigchain = ROOT.TChain('psdtree')
    sigchain.Add('%s/*root' % sig_dir)
    bkgchain = ROOT.TChain('psdtree')
    bkgchain.Add('%s/*root' % bkg_dir)

    print('Build outout numpy array')
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    for batchentry in range(int(batchsize/2)):
        #save charge and hittime to 3D array
        i = int(batchsize/2)*batch_num + batchentry
        if i >= sigchain.GetEntries() or i >= bkgchain.GetEntries():
            continue
        sigchain.GetEntry(i)
        bkgchain.GetEntry(i)
        print('entry', i)
        pmtids = sigchain.PMTID
        npes = sigchain.Charge
        hittime = sigchain.Time
        eqen = sigchain.Eqen
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            event2dimg[1, xbin, ybin] += hittime[j]
        pmtinfos.append(event2dimg)
        types.append(1)
        eqen_batch.append(eqen)
        vertices.append([sigchain.X, sigchain.Y, sigchain.Z])
        pmtids = bkgchain.PMTID
        npes = bkgchain.Charge
        hittime = bkgchain.Time
        eqen = bkgchain.Eqen
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids)):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
            event2dimg[0, xbin, ybin] += npes[j]
            event2dimg[1, xbin, ybin] += hittime[j]
        pmtinfos.append(event2dimg)
        types.append(0)
        vertices.append([bkgchain.X, bkgchain.Y, bkgchain.Z])
        eqen_batch.append(eqen)

    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    if outfile == '':
        np.save('data_fake.npz', pmtinfo=np.array(pmtinfos), eventtype=np.array(types))
    else:
        np.savez(outfile , pmtinfo=np.array(pmtinfos)[indices], eventtype=np.array(types)[indices], eqen=np.array(eqen_batch)[indices], vertex=np.array(vertices)[indices])

def uptotar(mapfile, infile, outfile='', eventtype=0):
    # The csv file of PMT map must have the same tag as the MC production.
    import tarfile, os
    import io   # python3 version

    print('Start processing')
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    print('Read ROOT files.', infile)

    print('Build outout numpy array')
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    branches = ['PMTID', 'Charge', 'Time', 'Eqen', 'X', 'Y', 'Z']
    import uproot
    df = uproot.open(infile)['psdtree']
    #save charge and hittime to 3D array
    print(df.keys())
    pmtids = df.array(b'PMTID')
    npes = df.array(b'Charge')
    hittime = df.array(b'Time')
    eqen = df.array(b'Eqen')
    Xs = df.array(b'X')
    Ys = df.array(b'Y')
    Zs = df.array(b'Z')
    tar = tarfile.TarFile(outfile, 'w')
    for entry in range(df.numentries):
        print(entry)
        abuf = io.BytesIO()
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids[entry])):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[entry][j])
            event2dimg[0, xbin, ybin] += npes[entry][j]
            event2dimg[1, xbin, ybin] += hittime[entry][j]
        np.savez( abuf, pmtinfo=event2dimg, eventtype=np.array(eventtype), eqen=np.array(eqen[entry]), vertex=np.array([Xs[entry], Ys[entry], Zs[entry]]))
        abuf.seek(0)
        info = tarfile.TarInfo(name=os.path.basename(outfile) + str(entry) + '.npz')
        info.size=len(abuf.getbuffer())
        tar.addfile(tarinfo=info, fileobj=abuf)
    tar.close()

def uptohdf(mapfile, infile, outfile='', eventtype=0):
    # The csv file of PMT map must have the same tag as the MC production.
    print('Start processing')
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    print('Build outout numpy array')
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    branches = ['PMTID', 'Charge', 'Time', 'Eqen', 'X', 'Y', 'Z']
    print('Read ROOT file:', infile)
    import uproot
    df = uproot.open(infile)['psdtree']
    #save charge and hittime to 3D array
    print(df.keys())
    pmtids = df.array(b'PMTID')
    npes = df.array(b'Charge')
    hittime = df.array(b'Time')
    eqen = df.array(b'Eqen')
    Xs = df.array(b'X')
    Ys = df.array(b'Y')
    Zs = df.array(b'Z')
    import h5py    # HDF5 support
    import six
    import os, time
    print("Write a HDF5 file")
    fileName = outfile
    timestamp = u'%s' % time.ctime()

    # create the HDF5 file
    f = h5py.File(fileName, "w")
    # point to the default data to be plotted
    f.attrs[u'default']          = u'entry'
    # give the HDF5 root some more attributes
    f.attrs[u'file_name']        = fileName
    f.attrs[u'file_time']        = timestamp
    f.attrs[u'creator']          = u'DSNBDataset_v1.py'
    f.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
    f.attrs[u'h5py_version']     = six.u(h5py.version.version)
    junodata = f.create_group(u'juno_data')
    for entry in range(df.numentries):
        print(entry)
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids[entry])):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[entry][j])
            event2dimg[0, xbin, ybin] += npes[entry][j]
            event2dimg[1, xbin, ybin] += hittime[entry][j]
        dset = junodata.create_dataset(os.path.basename(outfile) + str(entry), data=event2dimg, dtype='float16')
        dset.attrs[u'tag'] = eventtype
        dset.attrs[u'vertex'] = [Xs[entry], Ys[entry], Zs[entry]]
        dset.attrs[u'eqen'] = eqen[entry]
    f.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JUNO ML dataset builder.')
    parser.add_argument('--pmtmap', type=str, help='csc file of PMT map in JUNO.')
    parser.add_argument('--infile', '-i', type=str, help='Input root file.')
    parser.add_argument('--eventtype', '-t', type=int, default=0, help='Event type.')
    parser.add_argument('--outfile', '-o', type=str, help='Output root file.')
    parser.add_argument('--batch', '-n', type=int, help='Batch number.')
    args = parser.parse_args()
    uptohdf(args.pmtmap, args.infile, args.outfile, args.eventtype)
