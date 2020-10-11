import torch
import time
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torch.utils.data as data
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import argparse
import junodata, vgg, resnet

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
best_acc = 0 # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
NUM_EPOCHS = 5

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mean_square_loss(y_true, y_pred):
    '''
    loss function
    '''
    y_tr = y_true[:, 0:3]

    cross_entropy = torch.mean(torch.pow((y_tr - y_pred), 2), 1)
    #cross_entropy = tf.pow(cross_entropy, 0.5)
    cross_entropy = torch.mean(cross_entropy)

    return cross_entropy/len(y_tr)

def dist_acc(y_true, y_pred):
    '''
    accuracy defined as ratio of events with dist to real vertex
    less than 20 cm
    '''
    #print(y_true, y_pred)
    y_tr = y_true[:, 0:3]
    dists = torch.sum(torch.pow((y_tr - y_pred), 2), 1)
    acc = 0
    for dist in dists:
        if dist < 1000000:
            acc += 1
    return acc*1.0/len(y_true)

def train(trainloader, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_acc =0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.reshape(inputs.shape[1:]), targets.reshape(targets.shape[1:])
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = mean_square_loss(outputs, targets)
        acc = dist_acc(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc
        total += targets.size(0)
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*train_acc/total, train_acc, total))
    return train_loss/len(trainloader), 100.*train_acc/total

def test(testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    test_acc = 0
    total = 0
    score = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.reshape(inputs.shape[1:]), targets.reshape(targets.shape[1:])
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = mean_square_loss(outputs, targets)
            acc = dist_acc(outputs, targets)
            test_acc += acc
            test_loss += loss.item()
            total += targets.size(0)
            for m in range(outputs.size(0)):
                score.append([outputs[m], targets[m]])
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*test_acc/total, test_acc, total))

                # Save checkpoint.
            acc = 100.*test_acc/total
            if acc > best_acc:
                print('Saving..')
                state = {'net': net.state_dict(),
                         'acc': acc,
                         'epoch': epoch,
                         }
                if not os.path.isdir('checkpoint_sens' ):
                    os.mkdir('checkpoint_sens' )
                torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
                torch.save(state, './checkpoint_sens/ckpt.t7' )
                best_acc = acc
            return test_loss/len(testloader), 100.*test_acc/total, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 1d conv net classifier')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--filedir', '-i', type=str, help='directory of dataset files.')
    args = parser.parse_args()
    transformations = transforms.Compose([transforms.ToTensor()])
    # Data
    print('==> Preparing data..')
    list_of_datasets = []
    import glob
    filelist = glob.glob('%s/*.npz' % args.filedir)
    batch_dataset = junodata.BatchDataset(filelist, 500)

    # Creating data indices for training and validation splits:
    dataset_size = len(batch_dataset)
    indices = list(range(dataset_size))
    validation_split = .2
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=1, sampler=train_sampler, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=1, sampler=validation_sampler, num_workers=1)

    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3
    batchsize = 50
    batchsize_valid = 500
    start_epoch = 0

    print('==> Building model..')
    net = resnet.resnet18()
    # We use SGD
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    net = net.to(device)
    if args.resume and os.path.exists('./checkpoint_sens/ckpt.t7'):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_sens'), 'Error: no checkpoint directory found!'
        if device == 'cuda':
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7' )
        else:
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7', map_location=torch.device('cpu') )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    y_train_loss = np.zeros(100)
    y_train_acc = np.zeros(100)
    test_score = []
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + 10):
        epoch_start = time.time()
        # set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "Epoch [%d]: "%(epoch)
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                train_ave_loss, train_ave_acc = train(train_loader, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Epoch [%d] train aveloss=%.3f aveacc=%.3f"%(epoch,train_ave_loss,train_ave_acc))
            y_train_loss[epoch] = train_ave_loss
            y_train_acc[epoch]  = train_ave_acc

            # evaluate on validationset
            try:
                valid_loss,prec1, score = test(validation_loader, epoch)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Test[%d]:Result* Prec@1 %.3f\tLoss %.3f"%(epoch,prec1,valid_loss))
            test_score.append(score)
        epoch_elapse = time.time() - epoch_start
        print('Epoch %d used %f time' % (epoch, epoch_elapse))
        print(y_train_loss, y_train_acc)
    print('Total time used is %f' % time.time() - start_time)
        #np.save('test_score_%d.npy' % (start_epoch + 1), test_score)
