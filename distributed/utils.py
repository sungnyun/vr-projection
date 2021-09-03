import operator
import random
import numpy as np
import torch

from random_matrix_generator import general_generate_random_ternary_matrix_with_seed


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def distance(sgd, average_grad):

    diatance_grad=[]
    for sgd_layer, avggrad_layer in zip(sgd, average_grad):
        diatance_grad.append(sgd_layer-avggrad_layer)
    return diatance_grad

def generate_randommatrixlist(model, conv_cr, fc_cr, sparsity):

    random_matrix_lst=[]
    for param_idx, param in enumerate(model.parameters()):
        if len(param.shape) == 4:
            sh = param.shape
            row_d = max(sh[0] * sh[1], sh[2] * sh[3])
            u = general_generate_random_ternary_matrix_with_seed(row_d, ratio=conv_cr, s=sparsity)
            random_matrix_lst.append(u)     
        elif len(param.shape) == 2:
            row_d = max(param.shape[0], param.shape[1])
            u = general_generate_random_ternary_matrix_with_seed(row_d, ratio=fc_cr, s=sparsity)
            random_matrix_lst.append(u)
        elif len(param.shape) == 1:
            random_matrix_lst.append(None)

    return random_matrix_lst

def get_fullbatch_gradient(model, criterion, optimizer, train_loader):

    fullbatch_grad=[]
    for i in model.parameters():
         fullbatch_grad.append(torch.zeros(i.size()).cuda(device))
    
    for i, data in enumerate(train_loader):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        pred_label = model(image)
        loss = criterion(pred_label, label)
        optimizer.zero_grad()
        grad=torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        
        for idx, (fullbatch_grad_layer, grad_layer) in enumerate(zip(fullbatch_grad, grad)):
            fullbatch_grad[idx]=((fullbatch_grad_layer)*i+grad_layer)/(i+1)

    return fullbatch_grad

def get_fullbatch_gradient_information(fullbatch_grad, argmax, maxvalue):

    max_s_t = []
    for grad_layer in fullbatch_grad:
        s_t = torch.max(torch.abs(grad_layer)).item()
        max_s_t.append(s_t)
    
    print('maximum absolute element of each layer is')
    print(max_s_t)
    index, value = max(enumerate(max_s_t), key=operator.itemgetter(1))
    print('argmax absolute element of layer: ', index)
    print('1st maximum absolute element of gradient: ', value)
    second_value=np.sort(max_s_t)[-2]
    print('2nd maximum absolute element of gradient: ', second_value)
    
    argmax.append(index) #argmax 
    maxvalue.append(value) #max absolute value

def get_fullbatch_2norm_square(fullbatch_grad, normalizing_measure):

    normalizing_constant=0
    for idx, grad_layer in enumerate(fullbatch_grad):
        grad_layer=grad_layer.view(1, -1)
        normalizing_constant+=torch.norm(grad_layer, p='fro')**2
                
    normalizing_constant=normalizing_constant.item()
      
    print('normalizing constant: ', normalizing_constant)
    normalizing_measure.append(normalizing_constant)
    
    return normalizing_constant

def get_top1(test_accuracy):
    top1_acc_list = []
    for acc in test_accuracy:
        top1_acc_list.append(acc[1])
    max_top1_acc= np.sort(top1_acc_list)[-1]
    return max_top1_acc

def get_top5(test_accuracy):
    top1_acc_list = []
    for acc in test_accuracy:
        top1_acc_list.append(acc[2])
    max_top5_acc= np.sort(top1_acc_list)[-1]
    return max_top5_acc

def adjust_learning_rate(optimizer, lr):
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
