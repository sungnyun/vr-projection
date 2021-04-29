import copy
import torch
from utils import generate_randommatrixlist

def distance(sgd, average_grad):
    diatance_grad=[]
    for sgd_layer, avggrad_layer in zip(sgd, average_grad):
        diatance_grad.append(sgd_layer-avggrad_layer)
    return diatance_grad


def get_grad(loss, model, args):

    # backpropagation to get raw gradient
    loss.backward()
    
    sgd_origin = []
    for param in model.parameters():
        sgd_origin.append(param.grad.data)
    sgd = copy.deepcopy(sgd_origin)
    distance_grad = distance(sgd, model.average_grad) 

    # generate random matrix R
    random_matrix_lst = generate_randommatrixlist(model, args.conv_cr, args.fc_cr, sparsity=1)
    
    for param_idx, param in enumerate(model.parameters()):
        if len(param.shape) == 4:
            sh = param.shape
            update_param_grad = distance_grad[param_idx].reshape((sh[0] * sh[1], sh[2] * sh[3]))
            u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda()
            u_t = u.transpose(0, 1)

            # Compression Gradient with Random Matrix
            if update_param_grad.shape[0] > update_param_grad.shape[1]:
                encoding_grad = torch.mm(u_t, update_param_grad.data)
                decoding_grad = torch.mm(u, encoding_grad)
            else:
                encoding_grad = torch.mm(update_param_grad.data, u)
                decoding_grad = torch.mm(encoding_grad, u_t)
                
            new_grad = decoding_grad + model.average_grad[param_idx].reshape((sh[0] * sh[1], sh[2] * sh[3]))
            model.average_grad[param_idx] = (1-args.alpha) * model.average_grad[param_idx] +\
                                            args.alpha * (args.conv_cr * decoding_grad.reshape(sh) + model.average_grad[param_idx])
            param.grad.data = new_grad.reshape(sh)

        elif len(param.shape) == 2:
            u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda()
            u_t = u.transpose(0, 1)

            # Compression Gradient with Random Matrix
            if distance_grad[param_idx].shape[0] > distance_grad[param_idx].shape[1]:
                encoding_grad = torch.mm(u_t, distance_grad[param_idx])
                decoding_grad = torch.mm(u, encoding_grad)
            else:
                encoding_grad = torch.mm(distance_grad[param_idx], u)
                decoding_grad = torch.mm(encoding_grad, u_t)
                
            new_grad = decoding_grad + model.average_grad[param_idx]
            model.average_grad[param_idx] = (1-args.alpha) * model.average_grad[param_idx] +\
                                            args.alpha * (args.fc_cr * decoding_grad + model.average_grad[param_idx])
            param.grad.data = new_grad

        else:
            continue
