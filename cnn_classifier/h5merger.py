#!/usr/bin/env python
'''Reads NeXus HDF5 files using h5py and prints the contents'''

import h5py    # HDF5 support
import glob
import time
import csv
filelist = glob.glob('/scratchfs/exo/zepengli94/outfiles/atm_000745.h5')
t1 = time.time()
csvfile = open('dataset_info.csv', 'w')
fieldnames = ['dsetname']
writer = csv.DictWriter(csvfile, fieldnames)
with h5py.File('test.h5', 'w') as fid:
    junodata = fid.create_group(u'juno_data')
    for fileName in filelist:
        print(fileName, time.time() - t1)
        f = h5py.File(fileName,  "r")
        dset = f['juno_data']
        for item in dset.keys():
            writer.writerow({'dsetname':item})
            dset_copy = fid.create_dataset(item, data=dset[item])
            dset_copy.attrs[u'tag'] = dset[item].attrs[u'tag']
            dset_copy.attrs[u'vertex'] = dset[item].attrs[u'vertex']
            dset_copy.attrs[u'eqen'] = dset[item].attrs[u'eqen']
        f.close()
