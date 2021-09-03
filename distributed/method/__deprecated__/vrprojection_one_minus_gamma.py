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
    # distance_grad = distance(sgd, model.average_grad) 

    # generate random matrix R
    random_matrix_lst = generate_randommatrixlist(model, args.conv_cr, args.fc_cr, args.sparsity)
    
    for param_idx, param in enumerate(model.parameters()):
        if len(param.shape) == 4:
            sh = param.shape
            update_param_grad = param.grad.data.reshape((sh[0] * sh[1], sh[2] * sh[3]))
            u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda()
            u_t = u.transpose(0, 1)

            # Compression Gradient with Random Matrix
            if update_param_grad.shape[0] > update_param_grad.shape[1]:
                encoding_grad = torch.mm(u_t, update_param_grad.data)
                decoding_grad = torch.mm(u, encoding_grad)

                model.average_grad[param_idx] = args.conv_cr * decoding_grad.reshape(sh) + (1-args.conv_cr) * model.average_grad[param_idx]
                new_encoding_grad = torch.mm(u_t, (param.grad.data - model.average_grad[param_idx]).reshape([sh[0] * sh[1], sh[2] * sh[3]]))
                new_decoding_grad = torch.mm(u, new_encoding_grad)
            else:
                encoding_grad = torch.mm(update_param_grad.data, u)
                decoding_grad = torch.mm(encoding_grad, u_t)

                model.average_grad[param_idx] = args.conv_cr * decoding_grad.reshape(sh) + (1-args.conv_cr) * model.average_grad[param_idx]
                new_encoding_grad = torch.mm((param.grad.data - model.average_grad[param_idx]).reshape([sh[0] * sh[1], sh[2] * sh[3]]), u)
                new_decoding_grad = torch.mm(new_encoding_grad, u_t)

            param.grad.data = (new_decoding_grad + model.average_grad[param_idx].reshape([sh[0] * sh[1], sh[2] * sh[3]])).reshape(sh)

        elif len(param.shape) == 2:
            u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda()
            u_t = u.transpose(0, 1)

            # Compression Gradient with Random Matrix
            if param.grad.shape[0] > param.grad.shape[1]:
                encoding_grad = torch.mm(u_t, param.grad.data)
                decoding_grad = torch.mm(u, encoding_grad)

                model.average_grad[param_idx] = args.fc_cr * decoding_grad + (1-args.fc_cr) * model.average_grad[param_idx]
                new_encoding_grad = torch.mm(u_t, (param.grad.data - model.average_grad[param_idx]))
                new_decoding_grad = torch.mm(u, new_encoding_grad)
            else:
                encoding_grad = torch.mm(param.grad.data, u)
                decoding_grad = torch.mm(encoding_grad, u_t)

                model.average_grad[param_idx] = args.fc_cr * decoding_grad + (1-args.fc_cr) * model.average_grad[param_idx]
                new_encoding_grad = torch.mm((param.grad.data - model.average_grad[param_idx]), u)
                new_decoding_grad = torch.mm(new_encoding_grad, u_t)
                
            param.grad.data = new_decoding_grad + model.average_grad[param_idx]

        else:
            continue
