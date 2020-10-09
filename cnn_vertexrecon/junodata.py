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
                torch.from_numpy(np.asarray(sample['vertex'])).to(torch.float32)

class ListDataset(data.Dataset):
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
        pmtinfos = []
        vertices = []
        with open(filename, 'r') as datafile:
            r_json = json.load(datafile)
            for sample in r_json:
                pmtinfos.append(np.asarray(sample['pmtinfo']))
                vertices.append(np.asarray(sample['vertex']))
        return torch.from_numpy(np.array(pmtinfos)).to(torch.float32),\
                torch.from_numpy(np.array(vertices)).to(torch.float32)

def test():
    list_of_datasets = []
    filelist = glob.glob('./json_files/*.json')[:1000]
    #for j in filelist:
    #    if not j.endswith('.json'):
    #        continue  # skip non-json files
    #    list_of_datasets.append(SingleJsonDataset(json_file=j, root_dir='./', transform=None))
    ## once all single json datasets are created you can concat them into a single one:
    #multiple_json_dataset = data.ConcatDataset(list_of_datasets)
    dataset = ListDataset(filelist, 500)
    print(dataset[0][0])
    #for i in range(500):
    #    print(dataset[i][0].shape)

if __name__ == '__main__':
    test()
