import torch


def ternarize_grad(grad):
    """
    grad: tuple of gradients in all layers
    """
    max_s_t = []
    for grad_layer in grad:
        s_t = torch.max(torch.abs(grad_layer)).item()
        max_s_t.append(s_t)

    tern_grad = []
    for i, grad_layer in enumerate(grad):
        prob_tensor = torch.div(torch.abs(grad_layer), max_s_t[i])
        if torch.sum(torch.isnan(prob_tensor)) > 0:
            prob_tensor[prob_tensor != prob_tensor] = 0.5

        bernoulli = torch.bernoulli(prob_tensor)
        sign_tensors = torch.sign(grad_layer)

        tern_grad.append(max_s_t[i] * bernoulli * sign_tensors)

    return tern_grad


def get_grad(loss, model, args):
    
    loss.backward()
    # grad = torch.autograd.grad(loss, model.parameters())
    grad = []
    for param in model.parameters():
        grad.append(param.grad.data)
    tern_grad = ternarize_grad(grad)

    for param, tg in zip(model.parameters(), tern_grad):
        param.grad.data = tg
