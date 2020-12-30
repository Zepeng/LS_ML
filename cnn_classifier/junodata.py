import os, glob
import json, torch, h5py
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import pandas as pd

class SingleJsonDataset(data.Dataset):
    filename = ''
    length = 0
    def __init__(self, json_file, root_dir, transform = None):
        self.filename = json_file
        with open(json_file, 'r') as datafile:
            self.length = len(json.load(datafile))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.filename, 'r') as datafile:
            sample = json.load(datafile)[idx]
            return torch.from_numpy(np.asarray(sample['pmtinfo'])).to(torch.float32),\
                torch.from_numpy(np.asarray(sample['eventtype'])).to(torch.float32)

class ListDataset(data.Dataset):
    filelist = []
    length = 0
    nevt_file = 0
    def __init__(self, filelist, nevt_file):
        self.filelist = filelist
        self.length = nevt_file*len(self.filelist)
        self.nevt_file = nevt_file

    def __len__(self):
        return len(self.filelist)*self.nevt_file

    def __getitem__(self, idx):
        filename = self.filelist[int(idx/self.nevt_file)]
        batch = np.load(filename)
        pmtinfos = batch['pmtinfo'][idx % self.nevt_file]
        types = batch['eventtype'][idx % self.nevt_file]
        return torch.from_numpy(np.array(pmtinfos)).to(torch.float32),\
                torch.from_numpy(np.array(types)).to(torch.float32)

class BatchDataset(data.Dataset):
    filelist = []
    length = 0
    nevt_file = 0
    def __init__(self, filelist, nevt_file):
        self.filelist = filelist
        self.length = nevt_file*len(self.filelist)
        self.nevt_file = nevt_file

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        batch = np.load(filename)
        pmtinfos = batch['pmtinfo'] #[idx % self.nevt_file]
        types = batch['eventtype'] #[idx % self.nevt_file]
        edeps    = batch['edep']
        return torch.from_numpy(np.array(pmtinfos)).to(torch.float32),\
                torch.from_numpy(np.array(types)).to(torch.float32),\
                torch.from_numpy(np.array(edeps)).to(torch.float32)

class H5Dataset(data.Dataset):
    def __init__(self, h5_path, csv_path):
        self.to_tensor = transforms.ToTensor()
        csv_info = pd.read_csv(csv_path, header=None)
        self.groupname = np.asarray(csv_info.iloc[:,0])
        self.datainfo = np.asarray(csv_info.iloc[:,1])
        self.h5file = h5py.File(h5_path, 'r')
        self.h5dset = self.h5file['juno_data']

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        dset_entry = self.h5dset[self.groupname[idx]][self.datainfo[idx]]
        eventtype = dset_entry.attrs[u'tag']
        vertex = dset_entry.attrs[u'vertex']
        eqen = dset_entry.attrs[u'eqen']
        pmtinfo = np.array(dset_entry)
        return torch.from_numpy(pmtinfo).type(torch.FloatTensor), eventtype, eqen


def test():
    dataset = H5Dataset('test1.h5', 'dataset_info.csv')
    print(0, dataset[0][0].shape)

if __name__ == '__main__':
    test()
