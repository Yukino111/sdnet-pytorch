import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from nets.sd import Body
from REPVGG import repvgg_model_convert

model = Body(3, 'tiny', False)
model.load_state_dict(torch.load('logs/'))
model_1 = repvgg_model_convert(model, save_path='detect.pth')