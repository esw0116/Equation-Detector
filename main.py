import torch
from torch import nn
import tqdm

from option import args
from data import data
from trainer import ~~
from model import model
from logger import logger

torch.manual_seed(args.seed)
#checkpoint = logger.logger(args)
loader = data(args)
my_model = model(args, checkpoint)

t = Trainer_Mix(args, loader, my_model, checkpoint)
t.train()
t.test()

