import torch
from torch import nn
import tqdm

from option import args
from data import data
from trainer import Trainer
from model import model

torch.manual_seed(args.seed)
#checkpoint = logger.logger(args)
loader = data(args)
my_model = model(args, checkpoint)

t = Trainer(args, loader, my_model, checkpoint)
t.train()
t.test()

