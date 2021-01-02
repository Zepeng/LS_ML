import torch, h5py
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import pandas as pd

class H5Dataset(data.Dataset):
    def __init__(self, h5_path, csv_path):
        self.to_tensor = transforms.ToTensor()
        csv_info = pd.read_csv(csv_path, header=None)
        self.groupname = np.asarray(csv_info.iloc[:,0])
        self.datainfo = np.asarray(csv_info.iloc[:,1])
        self.h5file = h5py.File(h5_path, 'r')

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        dset_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        vertex = dset_entry.attrs[u'vertex']
        eqen = dset_entry.attrs[u'edep']
        pmtinfo = np.array(dset_entry)
        return torch.from_numpy(pmtinfo).to(torch.float32), \
                torch.from_numpy(vertex).to(torch.float32), eqen


def test():
    dataset = H5Dataset('test.h5', 'test.csv')
    print(0, dataset[0][0].shape)

if __name__ == '__main__':
    test()
