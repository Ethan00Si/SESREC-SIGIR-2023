#coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from models import *

import utils.data as data
from utils import Context as ctxt
import config.const as const_util

import os
import yaml

class Recommender(object):

    def __init__(self, flags_obj, workspace, dm, nc=None):

        self.dm = dm # dataset manager
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.load_model_config()
        self.update_model_config(nc)
        self.set_model()
        self.workspace = workspace

    def set_device(self):

        self.device  = ctxt.ContextManager.set_device(self.flags_obj)

    def load_model_config(self):
        path = 'config/{}_{}.yaml'.format(self.model_name, self.dm.dataset_name)
        f = open(path)
        self.model_config = yaml.load(f, Loader=yaml.FullLoader)


    def update_model_config(self, new_config):
        '''update model config'''
        if new_config is not None:
            self.model_config['git_id'] = new_config['git_id']
            for item in new_config.keys():
                if not item in self.model_config.keys():
                    raise ValueError(f'False config key value: {item}')
            for key in [item for item in new_config.keys() if item in self.model_config.keys()]:
                if type(self.model_config[key]) == dict:
                    self.model_config[key].update(new_config[key])
                else:
                    self.model_config[key] = new_config[key]

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)

    def save_ckpt(self):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'best.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self, assigned_path=None):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = None
        if assigned_path is not None:
            '''specific assigned path'''
            model_path = assigned_path
        else:
            '''default path'''   
            model_path = os.path.join(ckpt_path, 'best.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_dataset(self, *args):

        return getattr(data, f'{self.model_name}_Dataset')(*args)

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])

    def predict(self, sample):
        '''sample: input data and labels'''
        
        sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
        input_data = sample[:-1] # labels in last colomn 
        return self.model.forward(input_data)

    def get_loss(self, sample):
        '''sample: input data and labels'''
        sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
        input_data, label = sample[:-1], sample[-1]
        return self.model.forward(input_data, label)



class SESRec_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)

    def set_model(self):
        
        self.model = SESRec(config=self.model_config)
  
    def predict(self, sample):
        '''sample: input data and labels'''
        sample = [
                    [ 
                        [p.to(self.device) for p in k] 
                        if type(k) == list else k.to(self.device)
                        for k in i
                    ] 
                    if type(i) == list else i.to(self.device) 
                    for i in sample
                ]
        input_data = sample[:-1] # labels in the last colomn 
        return self.model.predict(input_data)


    def get_loss(self, sample, negs):

        neg_items, neg_queries = negs

        sample += [neg_items, neg_queries]

        sample = [
                    [ 
                        [p.to(self.device) for p in k] 
                        if type(k) == list else k.to(self.device)
                        for k in i
                    ] 
                    if type(i) == list else i.to(self.device) 
                    for i in sample
                ]
        
        return self.model.forward(sample)
    