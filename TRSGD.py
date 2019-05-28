import torch
from torch.optim.optimizer import Optimizer, required


class TRSGD(Optimizer):
    """ Revised based on the official implementation of SGD optimizer
    """

    def __init__(self, params, lambda_w, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.lambda_w = lambda_w
        
        super(TRSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TRSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
        
    # def get_lr(self):
    #     return self.param_groups[0]['lr']
    
    def step(self, iteration, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = 0
            nesterov = False

            lr = group['lr'] * (1.0/(iteration+1))  # lr decay in the inner loop iterations
            lam = self.lambda_w * (1/group['lr'])
            # print(lam)

            for k in range(len(group['params'])):
                p = group['params'][k]

                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                param_state = self.state[p]                
                if 'p_old' not in param_state:
                    param_state['p_old'] = torch.zeros_like(p.data)
                p_old = param_state['p_old']
                if iteration == 0:
                    p_old = p_old.copy_(p.data)
                    
                # caculate the gradient of regularizer
                d_p += (p.data - p_old.data) * 2 * lam
                
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-lr, d_p)

        return loss
