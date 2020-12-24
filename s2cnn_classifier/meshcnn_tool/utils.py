import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip

def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor([m.row, m.col])
    v = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))

def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape newlen x #V
    """
    batch_size, in_chan, nv = list(den.size())
    new_len = sp.size()[0]
    den = den.permute(2, 1, 0).contiguous().view(nv, -1)
    res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
    return res

def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x) # longitude as same as phi
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))  # latitude as same as theta
    return lat, long

def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    # print("sig_r2 Before : ", sig_r2.shape)  #sig_r2 Before :  (60, 60)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1) #aims to add sig_r2 images one column
    # print("sig_r2 After : ", sig_r2.shape) #sig_r2 After :  (60, 61)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    # print("ele and azi shape: ",ele.shape, azi.shape)
    # print("Before tran: ",np.array([ele, azi]), np.array([ele, azi]).shape) #(2, 642)
    s2 = np.array([ele, azi]).T
    # print("s2: ",s2, s2.shape) #(642, 2)
    sig_s2 = intp(s2).astype(dtype)
    # print("sig_s2 : ",sig_s2.shape) #sig_s2 :  (642,)
    return sig_s2

class MNIST_S2_Loader(Dataset):
    """Data loader for spherical MNIST dataset."""

    def __init__(self, data_zip, partition="train"):
        """
        Args:
            data_zip: path to zip file for data
            partition: train or test
        """
        assert(partition in ["train", "test"])
        self.data_dict = pickle.load(gzip.open(data_zip, "rb"))
        if partition == "train":
            self.x = self.data_dict["train_inputs"]/255
            self.y = self.data_dict["train_labels"]
        else:
            self.x = self.data_dict["test_inputs"]/255
            self.y = self.data_dict["test_labels"]
        # print("x Before: ",self.x, self.x.shape) #(60000, 2562)
        self.x = (np.expand_dims(self.x, 1) - 0.1307)/0.3081
        # print("x After: ",self.x, self.x.shape) #(60000, 1, 2562)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
