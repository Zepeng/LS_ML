import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import train
import sys
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import argparse
import junodata, vgg, resnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 1d conv net classifier')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--filedir', '-i', type=str, help='directory of dataset files.')
    args = parser.parse_args()
    #transformations = transforms.Compose([transforms.ToTensor()])
    # Data
    print('==> Preparing data..')
    list_of_datasets = []
    import glob
    filelist = glob.glob('%s/*.npz' % args.filedir)
    batch_dataset = junodata.BatchDataset(filelist, 500)

    # Creating data indices for training and validation splits:
    dataset_size = len(batch_dataset)
    indices = list(range(dataset_size))
    validation_split = .1   #set 10% data as validation set
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)  #why use SubsetRandomSampler here: comparing with random_split() ,it is able\
                                                        # to ensure echo batch sees a proportional number of all classes
    validation_sampler = SubsetRandomSampler(val_indices) #we can show the indexs of the Sampler by Sampler.indexs ,it is the indexs about the filelist

    train_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=1, sampler=train_sampler, num_workers=4) #we can use next(iter(train_loader)) to get the dataset which is junodata.BatchDataset in batch
    validation_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=1, sampler=validation_sampler, num_workers=4)

    writer  = SummaryWriter("runs/resnet")

    examples = iter(train_loader)

    # print(pmtinfo.shape)
    # print(pmtinfo[0][:,0,:,:].shape)
    # charge = pmtinfo[0][:,0,:,:]
    # print(charge)
    # hittime = pmtinfo[0][0,1,:,:]

    # for i in range(1):
    pmtinfo, vertex, Edep = examples.next()
    img_grid = torchvision.utils.make_grid( pmtinfo[0] )
    writer.add_image('cnn_pmtinfo', img_grid)
    # img_grid = torchvision.utils.make_grid( hittime[0] )
    # writer.add_image("cnn_pmtinfo", img_grid )
    net = resnet.resnet18()
    writer.add_graph( net , pmtinfo[0])


    writer.close()
    sys.exit()