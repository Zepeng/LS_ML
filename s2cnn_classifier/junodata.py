import os, glob
import json, torch
import numpy as np
import torch.utils.data as data

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

def test():
    list_of_datasets = []
    filelist = glob.glob('./npz_files/*.npz')
    dataset = BatchDataset(filelist, 500)
    for i in range(len(filelist)):
        print(filelist[i])
        print(dataset[i][2][0])
    #print(i, dataset[i][0].shape)

if __name__ == '__main__':
    test()
