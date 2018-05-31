import torch

from option import args
from data import data
from logger import logger
from trainer import Trainer_CNN
from model import model

torch.manual_seed(args.seed)
print(len(args.dictionary))
checkpoint = logger.logger(args)
dataloader = data(args)
my_model = model(args, checkpoint)
loader = dataloader.get_loader()

t = Trainer_CNN(args, loader, my_model, checkpoint)
while not t.termination():
    t.train()
    t.test()
