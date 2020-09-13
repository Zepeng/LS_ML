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
import junodata, vgg

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

def train(trainloader, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        optimizer.step()
        loss = criterion(outputs, targets)
        loss.backward()
        train_loss += loss.item()
        total += targets.size(0)
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader), 100.*correct/total

def test(testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    score = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            loss = criterion(predicted, targets)
            test_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            softmax = nn.Softmax()
            for m in range(outputs.size(0)):
                score.append([softmax(outputs[m])[1].item(), targets[m].item()])
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

                # Save checkpoint.
            acc = 100.*correct/total
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
            return test_loss/len(testloader), 100.*correct/total, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 1d conv net classifier')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    transformations = transforms.Compose([transforms.ToTensor()])
    # Data
    print('==> Preparing data..')
    list_of_datasets = []
    filelist = ['data_fake.json']
    for j in filelist:
        if not j.endswith('.json'):
            continue  # skip non-json files
        list_of_datasets.append(junodata.SingleJsonDataset(json_file=j, root_dir='./', transform=None))
    # once all single json datasets are created you can concat them into a single one:
    multiple_json_dataset = data.ConcatDataset(list_of_datasets)

    # Creating data indices for training and validation splits:
    dataset_size = len(multiple_json_dataset)
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
    train_loader = torch.utils.data.DataLoader(multiple_json_dataset, batch_size=200, sampler=train_sampler, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(multiple_json_dataset, batch_size=200, sampler=validation_sampler, num_workers=4)

    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3
    batchsize = 50
    batchsize_valid = 500
    start_epoch = 0

    print('==> Building model..')
    net = vgg.vgg16()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
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
    for epoch in range(start_epoch, start_epoch + 10):
        # set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "Epoch [%d]: "%(epoch)
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                rain_ave_loss, train_ave_acc = train(train_loader, epoch)
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
            y_valid_loss[epoch] = valid_loss
            y_valid_acc[epoch]  = prec1
        print(y_train_loss, y_train_acc, y_valid_loss, y_valid_acc)
        np.save('test_score_%d.npy' % (start_epoch + 1), test_score)
