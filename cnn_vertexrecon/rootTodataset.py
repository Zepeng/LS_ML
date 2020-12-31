import uproot as up
import numpy as np
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
            #print('Wrong PMT ID')
            return (0, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        ybin = np.where(self.thetas == theta)[0]
        xbin = np.where(self.thetaphi_dict[str(theta)] == phi)[0] + 113 - int(len(self.thetaphi_dict[str(theta)])/2)
        return(xbin, ybin)

def uptohdf(mapfile, infile, outfile=''):
    # The csv file of PMT map must have the same tag as the MC production.
    print('Start processing')
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    print('Read ROOT file:', infile)
    uptree = up.open(infile)['evt']
    pmtids = uptree.array('pmtID')
    edepxs = uptree.array('edepX')
    edepys = uptree.array('edepY')
    edepzs = uptree.array('edepZ')
    edeps  = uptree.array('edep')
    npes   = uptree.array('nPE')
    hittime= uptree.array('hitTime')
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
    f.attrs[u'creator']          = u'rootTodataset.py'
    f.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
    f.attrs[u'h5py_version']     = six.u(h5py.version.version)
    junodata = f.create_group(u'juno_data')
    for entry in range(uptree.numentries):
        print(entry)
        event2dimg = np.zeros((2, 225, 126), dtype=np.float16)
        for j in range(len(pmtids[entry])):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[entry][j])
            event2dimg[0, xbin, ybin] += npes[entry][j]
            event2dimg[1, xbin, ybin] += hittime[entry][j]
        dset = junodata.create_dataset(os.path.basename(outfile) + str(entry), data=event2dimg, dtype='float16')
        dset.attrs[u'vertex'] = [edepxs[entry], edepys[entry], edepzs[entry]]
        dset.attrs[u'edep'] = edeps[entry]
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JUNO ML dataset builder.')
    parser.add_argument('--pmtmap', type=str, help='csc file of PMT map in JUNO.')
    parser.add_argument('--infile', '-i', type=str, help='Input root file.')
    parser.add_argument('--outfile', '-o', type=str, help='Output root file.')
    args = parser.parse_args()
    uptohdf(args.pmtmap, args.infile, args.outfile)
