from .bigcomp19 import get_grad as bigcomp19_get_grad
from .bigcomp20 import get_grad as bigcomp20_get_grad
from .terngrad import get_grad as terngrad_get_grad
from .vrprojection import get_grad as vrprojection_get_grad


def get_grad(loss, model, args):
    if args.method == 'sgd':
        loss.backward()
    elif args.method == 'bigcomp19':
        bigcomp19_get_grad(loss, model, args)
    elif args.method == 'bigcomp20':
        bigcomp20_get_grad(loss, model, args)
    elif args.method == 'terngrad':
        terngrad_get_grad(loss, model, args)
    elif args.method == 'vrprojection':
        vrprojection_get_grad(loss, model, args)
    
