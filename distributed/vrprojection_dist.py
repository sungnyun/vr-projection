import copy
import util
import os, argparse, random, shutil, time, builtins, warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision

from models.vgg import *
from models.resnet20 import *
from models.resnet18 import *
from cifar_wrapper import CIFAR_Wrapper
from random_matrix_generator import general_generate_random_ternary_matrix_with_seed

'''
optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
'''

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='resnet18 | resnet20 | vgg16_bn')
parser.add_argument('--epochs', default200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100'])
parser.add_argument('--trainbatch', default=1024, type=int, metavar='N',
                        help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--testbatch', default=512, type=int, metavar='N',
                        help='test batch size')


parser.add_argument('--schedule', type=int, nargs='+', default=[75, 130, 180],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')


parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--conv-cr', default=0.0156254, type=float,
                    help='compression ratio for convolutional layers')
parser.add_argument('--fc-cr', default=0.0625, type=float,
                    help='compression ratio for fc layers')

###########################################################################
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
###########################################################################3

##################################################################################
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:20001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
###################################################################################
# Custom parser
"""---------------------------------------- 파서 작성 ---------------------------------------- """
parser.add_argument('--save_path',default='./res152_softmax1.0/', type=str, help='savepath')
parser.add_argument('--gpu_count',default= 8, type=int, help='use gpu count')
parser.add_argument('--clip_grad', default=False, action='store_true')

"""---------------------------------------------코드 실행---------------------------------------------------- """
# python vrprojection_dist.py --dataset cifar100 --gpu_count 8 --trainbatch 1024 --save_path ./checkpoint/vrprojection_dist_resnet18_8workers --multiprocessing-distributed
"""--------------------------------------------------------------------------------------------------------- """

best_acc1 = 0
args = parser.parse_args()
args.lr *= args.trainbatch / 128
for idx in range(len(args.schedule)):
    args.schedule[idx] *= int(args.epochs / 200)
state = {k: v for k, v in args._get_kwargs()}

MEAN = {'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408)}
STD = {'cifar10': (0.2470, 0.2435, 0.2616),
       'cifar100': (0.2675, 0.2565, 0.2761)}

def clip(tensor):
    shape = tensor.size()
    tensor = tensor.flatten()
    std = (tensor - torch.mean(tensor)) ** 2
    std = torch.sqrt(torch.mean(std))
    c = 2.5 * std.item()
    clipped_tensor = (torch.clamp(tensor, -c, c)).reshape(shape)
    return clipped_tensor


def main():
    if args.gpu is not None: #False#
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1: #False#
        args.world_size = int(os.environ["WORLD_SIZE"])

    ##########################################################################
    
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(args)

    """ used gpu count ex) 4 , all use : torch.cuda.device_count() """
    ngpus_per_node = args.gpu_count
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def _build_tensor(size, value=None):
    if value is None:
        value = size
    return torch.FloatTensor(size, size, size).fill_(value)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1 
    
    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    ###############################################################
    if args.gpu is not None:                                       
        print("Use GPU: {} for training".format(args.gpu))
    ###############################################################
 

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    
    #########################################################################
    if args.seed is not None: #False#
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        #warnings.warn('You have chosen to seed training. '
        #              'This will turn on the CUDNN deterministic setting, '
        #              'which can slow down your training considerably! '
        #              'You may see unexpected behavior when restarting '
        #              'from checkpoints.')
    else:
        raise Exception('Set seed or this method does not work!')
   
    """---------------------------------------- 모델, opimizer,loss 선언 ---------------------------------------- """
    
    
    # model = resnet20(num_classes=100)
    if args.arch == 'resnet18':
        model = resnet18(num_classes=100)
    elif args.arch == 'resnet20':
        model = resnet20(num_classes=100)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(num_classes=100)
    else:
        raise NotImplementedError

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    
    """--------------------------------------------------------------------------------------------------------- """
    
    if args.distributed:#분산에선 True

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
       
        if args.gpu is not None:############################################## args.gpu=3,0,1,2로 not None!! spawn쓰면 None이었던게 배정되버림                       
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.trainbatch = int(args.trainbatch / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:###################분산에선 이걸로 돌아감!!!#########################################
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)# output_device=0)
            
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        model = torch.nn.DataParallel(model).cuda()
    
            
     ###########################################################################################################################
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    #################################################################################################################################
    #cudnn.benchmark = True


    """---------------------------------------- 데이터 및 train 코드 작성. ----------------------------------------------- """

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
        train_dataset = datasets.CIFAR10(root='../', train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='../', train=False, download=False, transform=transform_test)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='/home/osilab/dataset/cifar100', train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='/home/osilab/dataset/cifar100', train=False, download=False, transform=transform_test)

    
    ''' *지우지 말것*   train dataset 넘겨줄것. -> DistributedSampler(train_dataset) '''
    if args.distributed:#####      True      ################################
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=args.seed)
    else:
        train_sampler = None
    '''----------------------------------------------------------------------------'''
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.trainbatch, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.testbatch, shuffle=False, num_workers=args.workers, pin_memory=True)

    
    if args.evaluate:
        validate(test_loader, model, criterion, args)###top1.avg를 리턴한다.
        return
    ##################################################################################
    ''' -------------------------저장 경로 설정-----------------------------'''
    save_path = args.save_path
    if dist.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    ''' -------------------------logger 선언----------------------------'''
    train_logger = util.Logger(os.path.join(save_path, 'train.log'))
    valid_logger = util.Logger(os.path.join(save_path, 'valid.log'))
    train_time_logger = util.Logger(os.path.join(save_path, 'train_time.log'))
    valid_time_logger = util.Logger(os.path.join(save_path, 'valid_time.log'))
    ##################################################################################
    
    average_grad, buffer_svrg, EC_grad = [], [], []
    for param in model.parameters():
        average_grad.append(torch.zeros(param.size()).cuda())
        buffer_svrg.append(torch.zeros(param.size()).cuda())
        EC_grad.append(torch.zeros(param.size()).cuda())
    
    ''' -------------------------학습 시작-----------------------------'''
    for epoch in range(args.start_epoch, args.epochs):
        ''' -------------------------건들지 말기.-----------------------------'''
        if args.distributed:
            train_sampler.set_epoch(epoch)
        ''' -----------------------------------------------------------------'''

        ''' -------------------------learning_rate decay---------------------'''
        adjust_learning_rate(optimizer, epoch)

        ''' -------------------------train-----------------------------'''
        train(average_grad, buffer_svrg, EC_grad, train_loader, model, criterion, optimizer, epoch, args, train_logger, train_time_logger)

        ''' -------------------------valid-----------------------------'''
        acc1 = validate(test_loader, model, criterion, epoch, args, valid_logger, valid_time_logger)

        ''' -------------------------save checkpoint.-----------------------------'''
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            util.save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            print(' * Best Acc@1 {:.3f}'.format(best_acc1))

            
def train(average_grad, buffer_svrg, EC_grad, train_loader, model, criterion, optimizer, epoch, args, logger, time_logger):
    ''' -------------------------averageMeter 선언.-----------------------------'''
    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4f')
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')
    
    ###########################없어도 될 것 같음##########################################
    ''' -------------------------출력 progress 선언.-----------------------------'''
    progress = util.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    #######################################################################################
    
    
    ''' -------------------------학습 시작.-----------------------------'''
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        random_matrix_lst = generate_random_matrixlist(model)

        if args.gpu is not None:######args.gpu= not None#######
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        sgd_origin = []
        for param in model.parameters():
            sgd_origin.append(param.grad.data)
        sgd = copy.deepcopy(sgd_origin)
        
        with torch.no_grad():
            for param_idx, param in enumerate(model.parameters()):
                if len(param.shape) == 4:
                    sh = param.shape
                    compression_length = sh[1] * sh[2] * sh[3]
                    update_param_grad = sgd[param_idx].reshape([sh[0], compression_length])
                    update_param_grad = clip(update_param_grad)

                    u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda(args.gpu)
                    u_t = u.transpose(0, 1)
                    
                    # Compression Gradient with Random Matrix
                    encoding_grad = torch.mm(update_param_grad, u)
                    decoding_grad = torch.mm(encoding_grad, u_t)
                    average_grad[param_idx] = (1/compression_length)*decoding_grad.reshape(sh) + args.momentum*average_grad[param_idx]

                    new_encoding_grad = torch.mm((update_param_grad.reshape(sh) - average_grad[param_idx]).reshape([sh[0], compression_length]), u)                        
                    new_decoding_grad = torch.mm(new_encoding_grad, u_t)
                    buffer_svrg[param_idx] = new_decoding_grad.reshape(sh) + average_grad[param_idx] + args.weight_decay*param.data + args.momentum*buffer_svrg[param_idx]
                    param.grad.data = buffer_svrg[param_idx]                

                    # EC_grad[param_idx] = sgd[param_idx] - decoding_grad.reshape(sh)
            
                elif len(param.shape) == 2:
                    sh = param.shape
                    update_param_grad = sgd[param_idx]
                    update_param_grad = clip(update_param_grad)

                    u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda(args.gpu)
                    u_t = u.transpose(0, 1)

                    # Compression Gradient with Random Matrix
                    encoding_grad = torch.mm(update_param_grad, u)
                    decoding_grad = torch.mm(encoding_grad, u_t)
                    average_grad[param_idx] = (1/sh[1])*decoding_grad.reshape(sh) + args.momentum*average_grad[param_idx]

                    new_encoding_grad = torch.mm((update_param_grad.reshape(sh) - average_grad[param_idx]), u)                        
                    new_decoding_grad = torch.mm(new_encoding_grad, u_t)
                    buffer_svrg[param_idx] = new_decoding_grad + average_grad[param_idx] + args.weight_decay*param.data + args.momentum*buffer_svrg[param_idx]
                    param.grad.data = buffer_svrg[param_idx]

                else:
                    buffer_svrg[param_idx] = clip(sgd[param_idx]) + args.weight_decay * param.data + args.momentum * buffer_svrg[param_idx]
                    param.grad.data = buffer_svrg[param_idx]
        
        # Gradient averaging
        average_gradients(model) 
        optimizer.step()
        
        ''' -------------------------이미지넷 top1, top5 accuracy----------------------------'''
        acc1, acc5, correct = util.accuracy(output, target, topk=(1, 5))

        ''' -------------------------각 GPU log 합쳐주기-----------------------------'''
        reduced_loss = reduce_tensor(loss.data)
        reduced_top1 = reduce_tensor(acc1.data)
        reduced_top5 = reduce_tensor(acc5.data)

        ''' ------------------------- averageMeter에 업데이트 -----------------------------'''
        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_top1.item(), images.size(0))
        top5.update(reduced_top5.item(), images.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        ################################# 이거 잘 이해 안됨  ######################################################    
        ''' ------------------------- gpu 하나로만 출력하기. (rank == 0 : 0번 gpu에서만 출력하도록.)-----------------------------'''
        if dist.get_rank() == 0:
            if i % args.print_freq == 0:
                progress.display(i)
        
    ''' ------------------------- logger 에 업데이트-----------------------------'''
    if dist.get_rank() == 0:
        logger.write([epoch, losses.avg, top1.avg, top5.avg])
        time_logger.write([epoch, batch_time.avg, data_time.avg])
#############################################################################################################   

def validate(test_loader, model, criterion, epoch, args, logger, time_logger):
    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4f')
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')
    progress = util.ProgressMeter(
        len(test_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix='Test: ')

    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            data_time.update(time.time() - end)

            if args.gpu is not None:#####None이다#############
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5, correct = util.accuracy(output, target, topk=(1, 5))

            reduced_loss = reduce_tensor(loss.data)######각 worker에서의 loss를 average ####
            reduced_top1 = reduce_tensor(acc1.data)######각 worker에서의 top1 accuracy를 average ####
            reduced_top5 = reduce_tensor(acc5.data)######각 worker에서의 top5 accuracy를 average ####

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_top1.item(), images.size(0))
            top5.update(reduced_top5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


            if dist.get_rank() == 0:
                if i % args.print_freq == 0:
                    progress.display(i)

        if dist.get_rank() == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    if dist.get_rank() == 0:
        logger.write([epoch, losses.avg, top1.avg, top5.avg])
        time_logger.write([epoch, batch_time.avg, data_time.avg])

    return top1.avg


def generate_random_matrixlist(model):
    random_matrix_lst=[]
    for param_idx, param in enumerate(model.parameters()):
        if len(param.shape) == 4:
            sh = param.shape
            row_d = sh[1] * sh[2] * sh[3]
            u = general_generate_random_ternary_matrix_with_seed(row_d, ratio=1/row_d, s=1)
            random_matrix_lst.append(u)     

        elif len(param.shape) == 2:
            sh = param.shape
            row_d = sh[1]
            u = general_generate_random_ternary_matrix_with_seed(row_d, ratio=1/row_d, s=1)
            random_matrix_lst.append(u)

        else:
            random_matrix_lst.append(None)

    return random_matrix_lst

def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= args.gpu_count

'''-----------------------------GPU에 있는 각 데이터 평균 취하기------------------------'''
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # gpu 갯수로 나눠줌.
    rt /= args.gpu_count
    return rt

'''----------------------------learning_rate deacy 설정 (ex)30 epcoh 마다 decay)-------------------------'''
def adjust_learning_rate(optimizer, epoch):

    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def get_top1(test_accuracy):
    top1_acc_list = []
    for acc in test_accuracy:
        top1_acc_list.append(acc)
    max_top1_acc= np.sort(top1_acc_list)[-1]
    
    return max_top1_acc

def get_top5(test_accuracy):
    top1_acc_list = []
    for acc in test_accuracy:
        top1_acc_list.append(acc[2])
    max_top5_acc= np.sort(top1_acc_list)[-1]
    
    return max_top5_acc


if __name__ == '__main__':
    main()
