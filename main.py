import argparse
from utils.Context import ContextManager, DatasetManager
from config import const
import models.Inputs as data_in

import torch
import numpy as np
import random 
import os

def setup_seed(seed):
    '''setting random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 
setup_seed(20210823)

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--description', type=str, help='experiments details, used for log name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')

parser.add_argument('--dataset_name', type=str, default='amazon')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--tb', type=bool, help='whether use tensorboard (record metrics)', default=True)
parser.add_argument('--train_tb', type=bool, help='whether use tensorboard to record loss', default=True)
parser.add_argument('--verbose', type=bool, help='whether save model paremeters in tensorborad', default=False)
parser.add_argument('--model', type=str, help='which model to use', default='')

parser.add_argument('--num_negs', type=int, help='# negtive samples for training', default=4)
parser.add_argument('--batch_size', type=int, help='training batch_size', default=256)


args = parser.parse_args()

if args.dataset_name == 'commercial': #Kuaishou Dataset
    const.init_dataset_setting_commercial()
    data_in.init_data_attribute_ls('commercial')
elif args.dataset_name == 'amazon':
    const.init_dataset_setting_amazon()
    data_in.init_data_attribute_ls('amazon')
else:
    raise ValueError(f'Not support dataset: {args.dataset_name}')


cm = ContextManager(args)
dm = DatasetManager(args)

trainer = cm.set_trainer(args, cm, dm)

trainer.train()
trainer.test()