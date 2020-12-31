#!/usr/bin/env python
'''Merge hdf5 files into a single file'''
import h5py    # HDF5 support
import glob
import time
import csv, argparse

def h5merger(filedir, csvfile, outfile):
    filelist = glob.glob('%s/*.h5' % filedir)
    csvfile = open(csvfile, 'w')
    fieldnames = ['groupname', 'dsetname']
    writer = csv.DictWriter(csvfile, fieldnames)
    with h5py.File(outfile, 'w') as fid:
        junodata = fid.create_group(u'juno_data' )
        for i in range(len(filelist)):
            fileName = filelist[i]
            print(fileName)
            f = h5py.File(fileName,  "r")
            f.copy(f['juno_data'], junodata, name='juno_data_%d' % i)
            dset = f['juno_data']
            for item in dset.keys():
                writer.writerow({'groupname':'juno_data_%d' % i, 'dsetname':item})
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset builder.')
    parser.add_argument('--filedir', '-f', type=str, help='directory of h5 files.')
    parser.add_argument('--outfile', '-o', type=str, help='output h5 file.')
    parser.add_argument('--csvfile', '-c', type=str, help='csv file of dataset info.')
    args = parser.parse_args()
    h5merger(args.filedir, args.csvfile, args.outfile)
