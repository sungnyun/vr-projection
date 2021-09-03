import torch
from utils import generate_randommatrixlist

def get_grad(loss, model, args):

    # backpropagation to get raw gradient
    loss.backward()
    
    # generate random matrix R
    random_matrix_lst = generate_randommatrixlist(model, args.conv_cr, args.fc_cr, sparsity=1)
    
    for param_idx, param in enumerate(model.parameters()):
        if len(param.shape) == 4:
            sh = param.shape
            update_param_grad = param.grad.reshape((sh[0] * sh[1], sh[2] * sh[3]))
            u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda()
            u_t = u.transpose(0, 1)

            # Compression Gradient with Random Matrix
            if update_param_grad.shape[0] > update_param_grad.shape[1]:
                encoding_grad = torch.mm(u_t, update_param_grad.data)
                decoding_grad = torch.mm(u, encoding_grad)
            else:
                encoding_grad = torch.mm(update_param_grad.data, u)
                decoding_grad = torch.mm(encoding_grad, u_t)
                
            new_grad = decoding_grad.reshape(sh)
            param.grad.data = new_grad

        elif len(param.shape) == 2:
            u = torch.from_numpy(random_matrix_lst[param_idx]).float().cuda()
            u_t = u.transpose(0, 1)

            # Compression Gradient with Random Matrix
            if param.grad.shape[0] > param.grad.shape[1]:
                encoding_grad = torch.mm(u_t, param.grad.data)
                decoding_grad = torch.mm(u, encoding_grad)
            else:
                encoding_grad = torch.mm(param.grad.data, u)
                decoding_grad = torch.mm(encoding_grad, u_t)
                
            new_grad = decoding_grad
            param.grad.data = new_grad

        else:
            continue
