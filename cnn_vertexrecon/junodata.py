import os, glob
import json, torch
import numpy as np
import torch.utils.data as data


class SingleJsonDataset(data.Dataset):
    filename = ''
    length = 0

    def __init__(self, json_file, root_dir, transform=None):
        self.filename = json_file
        with open(json_file, 'r') as datafile:
            self.length = len(json.load(datafile))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.filename, 'r') as datafile:
            sample = json.load(datafile)[idx]
            return torch.from_numpy(np.asarray(sample['pmtinfo'])).to(torch.float32), \
                   torch.from_numpy(np.asarray(sample['vertex'])).to(torch.float32)


class ListDataset(data.Dataset):
    filelist = []
    length = 0
    nevt_file = 0

    def __init__(self, filelist, nevt_file):
        self.filelist = filelist
        self.length = nevt_file * len(self.filelist)
        self.nevt_file = nevt_file

    def __len__(self):
        return len(self.filelist) * self.nevt_file

    def __getitem__(self, idx):
        filename = self.filelist[int(idx / self.nevt_file)]
        batch = np.load(filename)
        pmtinfos = batch['pmtinfo'][idx % self.nevt_file]
        vertices = batch['vertex'][idx % self.nevt_file]
        return torch.from_numpy(np.array(pmtinfos)).to(torch.float32), \
               torch.from_numpy(np.array(vertices)).to(torch.float32)


class BatchDataset(data.Dataset):
    filelist = []
    length = 0
    nevt_file = 0

    def __init__(self, filelist, nevt_file):
        self.filelist = filelist
        self.length = nevt_file * len(self.filelist)
        self.nevt_file = nevt_file

    # The function in this format{__xxx__(self)} is not used by the user by python itself,
    # which means we cannot use BatchDataset.__len__() to invoke it but len(BatchDataset)
    # and BatchDataset.__getitem__() means BatchDataset[i]
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        batch = np.load(filename)
        pmtinfos = batch['pmtinfo']  # [idx % self.nevt_file]
        vertices = batch['vertex']  # [idx % self.nevt_file]
        edeps = batch['edep']
        # print("pmtinfos:   ", pmtinfos[:1])
        # print("pmtinfos id0:", pmtinfos[0][0][0])
        # print("vertices:   ", vertices[:1])
        # print("edeps:      ", edeps[:1])
        # print("pmtinfos.shape:",pmtinfos.shape)
        return torch.from_numpy(np.array(pmtinfos)).to(torch.float32), \
               torch.from_numpy(np.array(vertices)).to(torch.float32), \
               torch.from_numpy(np.array(edeps)).to(torch.float32)


def test():
    list_of_datasets = []
    filelist = glob.glob('./npz_files/sample_0.npz')[:1000]
    dataset = BatchDataset(filelist, 500)
    print(dataset[0][2].shape)
    # print(i, dataset[i][0].shape)


if __name__ == '__main__':
    test()
