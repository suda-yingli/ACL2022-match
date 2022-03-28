import torch
import torch.optim as optim

class MetaOptimizer:
    def __init__(self, params_to_optimize, conf):
        #if meta_opti_type == 'sgd':
        #    self._meta_optimizer = optim.SGD(params_to_optimize, lr=conf.meta_learning_rate, weight_decay=1e-4, momentum=0.9, nesterov=opt.nesterov)
        #else:
        #self.optim = optim.Adam(params_to_optimize, lr=conf.meta_learning_rate, weight_decay=1e-4)
        self.optim = optim.Adam(params_to_optimize, lr=conf.meta_learning_rate, weight_decay=1e-4)
        self.scheduler = \
            torch.optim.lr_scheduler.LambdaLR(self.optim,
                                              lr_lambda=lambda epoch: conf.decay ** (float(epoch) / conf.decay_steps))

    def step(self):
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()



class Optimizer:
    def __init__(self, params_to_optimize, conf):
        self.optim = torch.optim.Adam(params_to_optimize, lr=conf.learning_rate, betas=(conf.beta_1, conf.beta_2),
                                      eps=conf.epsilon, weight_decay=1e-6)
        self.scheduler = \
            torch.optim.lr_scheduler.LambdaLR(self.optim,
                                              lr_lambda=lambda epoch: conf.decay ** (float(epoch) / conf.decay_steps))

    def step(self):
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()

    # def schedule(self):
    #     self.scheduler.step()

    # def zero_grad(self):
    #     self.optim.zero_grad()

    # @property
    # def lr(self):
    #     return self.scheduler.get_lr()

