import os
import json, torch
import numpy as np
import torch.utils.data as data

class SingleJsonDataset(data.Dataset):
    def __init__(self, json_file, root_dir, transform = None):
        datafile = open(json_file, 'r')
        self.dataset = json.load(datafile)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return torch.from_numpy(np.asarray(sample['pmtinfo'])).to(torch.float32),\
                torch.from_numpy(np.asarray(sample['vertex'])).to(torch.float32)

def test():
    list_of_datasets = []
    filelist = ['data_fake.json']
    for j in filelist:
        if not j.endswith('.json'):
            continue  # skip non-json files
        list_of_datasets.append(SingleJsonDataset(json_file=j, root_dir='./', transform=None))
    # once all single json datasets are created you can concat them into a single one:
    multiple_json_dataset = data.ConcatDataset(list_of_datasets)
    print(multiple_json_dataset[0][0].shape)

#test()
