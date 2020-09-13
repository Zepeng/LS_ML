import uproot as up
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
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

def roottojson(mapfile, rootfile, outfile=''):
    # The csv file of PMT map must have the same tag as the MC production.
    pmtmap = PMTIDMap(mapfile)
    pmtmap.CalcDict()

    uptree = up.open(rootfile)['data']
    pmtids = uptree.array('pmtID')
    initxs = uptree.array('initX')
    initys = uptree.array('initY')
    initzs = uptree.array('initZ')
    npes   = uptree.array('npe')
    hittime= uptree.array('hittime')
    dataset = []
    for i in range(len(pmtids)):
        #use a dictionary to save the array and vertex information
        eventdict = {}
        #save charge and hittime to 3D array
        event2dimg = np.zeros((2, 225, 124), dtype=np.float)
        for j in range(len(pmtids[i])):
            (xbin, ybin) = pmtmap.CalcBin(pmtids[i][j])
            event2dimg[0, xbin, ybin] += npes[i][j]
            event2dimg[1, xbin, ybin] += hittime[i][j]
        eventdict['pmtinfo'] = event2dimg.tolist()
        eventdict['vertex'] = np.array([initxs[i], initys[i], initzs[i]]).tolist()
        dataset.append(eventdict)

    if outfile == '':
        with open('data_fake.json', 'w') as data_file:
            json.dump(dataset, data_file)
    else:
        with open(outfile, 'w') as data_file:
            json.dump(dataset, data_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JUNO ML dataset builder.')
    parser.add_argument('--pmtmap', type=str, help='csc file of PMT map in JUNO.')
    parser.add_argument('--infile', '-i', type=str, help='Input root file.')
    parser.add_argument('--outfile', '-o', type=str, help='Output root file.')
    args = parser.parse_args()
    roottojson(args.pmtmap, args.infile, args.outfile)
    #'/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J19v1r1-Pre4/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv', '/junofs/users/lizy/public/deeplearning/J19v1r0-Pre3/samples/train/eplus_ekin_0_10MeV/0/root_data/sample_0.root')
