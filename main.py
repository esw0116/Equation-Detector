import torch
from torch import nn
import tqdm

from option import args
from data import data
from logger import logger
from trainer import Trainer
from model import model

torch.manual_seed(args.seed)
checkpoint = logger.logger(args)
dataloader = data(args)
my_model = model(args, checkpoint)
loader = dataloader.get_loader()

t = Trainer(args, loader, my_model, checkpoint)
if not args.test_only:
    t.train()
t.test()

