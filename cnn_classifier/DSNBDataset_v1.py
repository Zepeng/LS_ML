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
            return (-1, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        ybin = np.where(self.thetas == theta)[0]
        xbin = np.where(self.thetaphi_dict[str(theta)] == phi)[0] + 112 - int(len(self.thetaphi_dict[str(theta)])/2)
        return(xbin, ybin)

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
    # give the HDF5 root some more attributes
    f.attrs[u'sourcefile']       = infile
    f.attrs[u'file_name']        = fileName
    f.attrs[u'file_time']        = timestamp
    f.attrs[u'creator']          = u'DSNBDataset_v1.py'
    f.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
    f.attrs[u'h5py_version']     = six.u(h5py.version.version)
    junodata = f.create_group(u'juno_data')
    for entry in range(df.numentries):
        event2dimg = np.zeros((2, 225, len(pmtmap.thetas)), dtype=np.float16)
        for j in range(len(pmtids[entry])):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[entry][j])
            if xbin == -1:
                continue
            event2dimg[0, xbin, ybin] += npes[entry][j]
            event2dimg[1, xbin, ybin] += hittime[entry][j]
        dset = junodata.create_dataset(os.path.basename(outfile).replace('h5',str(entry)), data=event2dimg, dtype='f2')
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
