from torch import optim
from torch.optim import lr_scheduler as lrs


def set_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    kwargs = dict()
    optimizer_type = optim.Adam
    kwargs['lr'] = args.learning_rate

    return optimizer_type(trainable, **kwargs)


def set_scheduler(args, optimizer):
    scheduler = lrs.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    return scheduler
