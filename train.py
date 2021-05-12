import os
import sys
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from method import get_grad
from resnet import *
from utils import *
from cifar_wrapper import CIFAR_Wrapper

models_dict = {'resnet20': resnet20}
MEAN = {'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408)}
STD = {'cifar10': (0.2470, 0.2435, 0.2616),
       'cifar100': (0.2675, 0.2565, 0.2761)}

class Trainer():
    def __init__(self, args):
        self.args = args

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),
        ])

        if args.dataset == 'cifar10':
            train_dataset = CIFAR_Wrapper(root=args.data_dir, train=True, download=args.download, transform=transform_train, nclass=10)
            test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=args.download, transform=transform_test)
        elif args.dataset == 'cifar100':
            train_dataset = CIFAR_Wrapper(root=args.data_dir, train=True, download=args.download, transform=transform_train, nclass=100)
            test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=args.download, transform=transform_test)
        else:
            raise NotImplementedError
        self.train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        
        kwargs_model = {'num_classes': train_dataset.nclass}
        try:
            self.model = models_dict[args.model.lower()](**kwargs_model).cuda()
        except:
            try:
                self.model = models.__dict__[args.model.lower()](**kwargs_model).cuda()
            except:
                raise NotImplementedError

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        self.train_loss, self.test_accuracy = [], []

    def train(self, epoch):
        self.model.train()
        self.train_loader.dataset.retransform()
        losses = 0
        top1 = AverageMeter('Top1 Accuracy')
        
        if epoch == 0:
            # Construct Average Gradient (option: initial minibatch grad / zeros)
            self.model.average_grad = []
            image, label = next(iter(self.train_loader))
            image, label = image.cuda(), label.cuda()
            pred_label = self.model(image)
            loss = self.criterion(pred_label, label)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                # self.model.average_grad.append(torch.zeros_like(param))
                self.model.average_grad.append(param.grad.data)

        for i, data in enumerate(tqdm(self.train_loader)):
            image = data[0].cuda()
            label = data[1].cuda()
            
            pred_label = self.model(image)
            loss = self.criterion(pred_label, label)
            losses += loss.item()
            top1_acc = accuracy(pred_label.data, label.data, topk=(1,))[0]
            top1.update(top1_acc.item(), image.size(0))
            
            self.optimizer.zero_grad()
            get_grad(loss, self.model, self.args)
            #loss.backward()
            self.optimizer.step()

        avg_loss = losses / len(self.train_loader)
        self.train_loss.append(avg_loss)
        print('Epoch {} - Train Loss: {:.6f}, Accuracy: {:.2f}'.format(epoch, avg_loss, top1.avg))

        if epoch in self.args.lr_decay:
            self.args.lr *= 0.1
            adjust_learning_rate(self.optimizer, self.args.lr)

    def test(self, epoch):
        self.model.eval()
        losses = AverageMeter('Loss')
        top1 = AverageMeter('Top1 Accuracy')
        top5 = AverageMeter('Top5 Accuracy')

        for i, data in enumerate(self.test_loader):
            image = data[0].cuda()
            label = data[1].cuda()

            pred_label = self.model(image)
            loss = self.criterion(pred_label, label)

            top1_acc, top5_acc = accuracy(pred_label.data, label.data, topk=(1,5))
            losses.update(loss.item(), image.size(0))
            top1.update(top1_acc.item(), image.size(0))
            top5.update(top5_acc.item(), image.size(0))
        
        self.test_accuracy.append((losses.avg, top1.avg, top5.avg))
        print('(Test) Top1 Accuracy: {}, Top5 Accuracy: {}'.format(top1.avg, top5.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    
    parser.add_argument('--method', type=str, default='sgd',
                        choices=['sgd', 'bigcomp19', 'bigcomp20', 'terngrad', 'vrprojection'])
    parser.add_argument('--model', type=str, default='resnet20',
                        help='model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='dataset directory')
    parser.add_argument('--download', action='store_true', default=False,
                        help='download dataset')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of dataloader workers')
    parser.add_argument('--checkname', type=str, default=None,
                        help='checkpoint name')
    # training hyperparameters
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='optimizer momentum')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='optimizer weight decay')
    parser.add_argument('--train-batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='test batch size')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1000,
                        help='random seed')
    parser.add_argument('--lr-decay', type=int, nargs='+', default=[100, 150],
                        help='learning rate decay epoch')
    parser.add_argument('--alpha', type=float, default=0.01, 
                        help='average gradient momentum param for bigcomp20 method')
    # compression hyperparameters
    parser.add_argument('--seed-interval', type=int, default=1,
                        help='interval for seed change')
    parser.add_argument('--conv-cr', type=float, default=0.0625,
                        help='convolutional layers compression ratio (default:1/16)')
    parser.add_argument('--fc-cr', type=float, default=0.0625,
                        help='fc-layers compression ratio (default:1/16)')
    parser.add_argument('--sparsity', type=int, default=2,
                        help='vrprojection random matrix sparsity value')

    args = parser.parse_args()

    if args.checkname is None:
        raise RuntimeError('Please identify the checkpoint (experiment) name')
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    checkpoint_dir = os.path.join('./checkpoint', args.checkname)

    if args.num_epochs is None:
        print('Check epochs and lr decay epochs!')
        epochs_dict = {'cifar10': 200, 'cifar100': 300}
        lr_decay_dict = {'cifar10': [100, 150], 'cifar100': [150, 225]}
        args.num_epochs, args.lr_decay = epochs_dict[args.dataset], lr_decay_dict[args.dataset] 

    print(args)

    ### Train & Val ###
    set_seeds(args.seed)
    trainer = Trainer(args)

    for epoch in range(args.num_epochs):
        trainer.train(epoch)
        trainer.test(epoch)
    
    ### Logging ###
    with open(checkpoint_dir + '.pickle', 'wb') as f:
        pickle.dump(trainer.train_loss, f)
        pickle.dump(trainer.test_accuracy, f)
