#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from tqdm import tqdm
from utils.data import SESRec_neg_sample_Dataset, get_dataloader
import config.const as const_util

from utils import Context as ctxt
from tester import Tester

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import logging


class Trainer(object):

    def __init__(self, flags_obj, cm,  dm, new_config=None):
        """
        Args:
            flags_obj: arguments in main.py
            cm : context manager
            dm : dataset manager
            new config : update default model config(`./config/model_commercial.yaml`) to tune hyper-parameters
        """

        self.name = flags_obj.name + '_trainer'
        self.cm = cm #context manager
        self.dm = dm #dataset manager
        self.flags_obj = flags_obj
        self.set_recommender(flags_obj, cm.workspace, dm, new_config)
        self.recommender.transfer_model()
        self.lr = self.recommender.model_config['lr']
        self.set_tensorboard(flags_obj.tb)
        self.tester = Tester(flags_obj, self.recommender, self.writer)
        

    def set_recommender(self, flags_obj, workspace, dm, new_config):

        self.recommender = ctxt.ContextManager.set_recommender(flags_obj, workspace, dm, new_config)

    def train(self):

        self.set_dataloader()
        self.set_optimizer()
        self.set_scheduler()
        self.set_esm() #early stop manager

        best_metric = 0
        train_loss = [0.0, 0.0, 0.0] #store every training loss
        
        for epoch in range(self.flags_obj.epochs):

            self.train_one_epoch(epoch, train_loss)
            watch_metric_value = self.validate(epoch)
            if watch_metric_value > best_metric:
                self.recommender.save_ckpt()
                logging.info('save ckpt at epoch {}'.format(epoch))
                best_metric = watch_metric_value
            self.scheduler.step(watch_metric_value)

            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break

    def set_test_dataloader(self):
        raise NotImplementedError

    def test(self, tune_para=None, assigned_model_path = None, load_config=True):
        '''
            test model on test dataset
        Args:
            tune_para
        '''

        self.set_test_dataloader()
        
        if load_config:
            self.recommender.load_ckpt(assigned_path = assigned_model_path)

        results = self.tester.test()

        logging.info('TEST results :')
        self.record_metrics('test', results)
        print('test: ', results)

    def set_dataloader(self):

        raise NotImplementedError

    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()

    def set_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
         mode='max', patience=self.recommender.model_config['patience'], 
         min_lr=self.recommender.model_config['min_lr'])

    def set_esm(self):

        self.esm = ctxt.EarlyStopManager(self.recommender.model_config)

    def set_tensorboard(self, tb=False):
        if tb:
            self.writer = SummaryWriter("{}/tb/{}".format(self.cm.workspace_root, self.cm.exp_name))
        else:
            self.writer = None


    def record_metrics(self, epoch, metric):
        """
        record metrics after each epoch
        """    

        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, metric))
        if self.writer:
            if epoch != 'test':
                for k,v in metric.items():
                        self.writer.add_scalar("training_metric/"+str(k), v, epoch)

    def train_one_epoch(self, epoch, train_loss):

        self.lr = self.train_one_epoch_core(epoch, self.dataloader, self.optimizer, self.lr, train_loss)

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr, train_loss):

        epoch_loss = train_loss[0]

        self.recommender.model.train()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:

            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=100)
        for step, sample in enumerate(tqdm_):

            optimizer.zero_grad()
            loss = self.recommender.get_loss(sample)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if step % (dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step+1, epoch_loss / (step+1+epoch*dataloader.__len__())))
                if self.writer and self.flags_obj.train_tb:
                    self.writer.add_scalar("training_loss",
                                    epoch_loss/(step+1+epoch*dataloader.__len__()), step+1+epoch*dataloader.__len__())

        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss/(step+1+epoch*dataloader.__len__())))

        train_loss[0] = epoch_loss

        return lr

    def validate(self, epoch):

        results = self.tester.test()
        self.record_metrics(epoch, results)
        print(results)
       
        return results['mrr']


    

class SequenceTrainer(Trainer):
    
    def __init__(self, flags_obj, cm, dm, nc):

        super().__init__(flags_obj, cm, dm, nc)

    def set_dataloader(self):

        # training dataloader
        self.dataloader = get_dataloader(
            data_set = self.recommender.get_dataset(const_util.train_file, self.dm, True),
            bs = self.dm.batch_size,
            prefetch_factor = self.dm.batch_size // 8 + 1, num_workers = 8
        )
        # validation dataloader
        self.tester.set_dataloader(
            dst = self.recommender.get_dataset(const_util.valid_file, self.dm, False)
        )

    def set_test_dataloader(self):

        # test dataloader
        self.tester.set_dataloader(
            dst = self.recommender.get_dataset(const_util.test_file, self.dm, False)
        )


class SESRec_Trainer(SequenceTrainer):
    def __init__(self, flags_obj, cm, dm, nc):

        super().__init__(flags_obj, cm, dm, nc)
        self.set_infoNCE()

    def set_infoNCE(self):
        self.w_infoNCE = self.recommender.model_config['infoNCE_weight']
        self.w_trip = self.recommender.model_config['triplet_weight']
        num_neg = self.recommender.model_config['infoNCE_neg_sample']
        self.infoNCE_dataloader = get_dataloader(
            data_set = SESRec_neg_sample_Dataset(const_util.valid_file, self.dm),
            bs = num_neg, prefetch_factor = num_neg // 4 + 1, num_workers = 4
        )
    

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr, train_loss):
        '''training procedure for each epoch'''
    
        epoch_loss = train_loss[0]
        epoch_feat_align_loss = train_loss[1]
        epoch_int_contrast_loss = train_loss[2]

        self.recommender.model.train()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:

            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        iterator = iter(self.infoNCE_dataloader)

        tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=100)
        for step, sample in enumerate(tqdm_):

            optimizer.zero_grad()
            bce_loss, feat_align_loss, int_contrast_loss = self.recommender.get_loss(sample, next(iterator))

            loss = bce_loss + self.w_infoNCE * feat_align_loss + self.w_trip * int_contrast_loss

            loss.backward()
            optimizer.step()
            
            epoch_loss += bce_loss.item()
            epoch_feat_align_loss += feat_align_loss.item()
            epoch_int_contrast_loss += int_contrast_loss.item()

            if step % (dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step+1, epoch_loss / (step+1+epoch*dataloader.__len__())))
                if self.writer and self.flags_obj.train_tb:
                    
                    self.writer.add_scalar("training_loss/BCE",
                                    epoch_loss/(step+1+epoch*dataloader.__len__()),  + epoch * 50)
                    self.writer.add_scalar("training_loss/InfoNCE",
                                    epoch_feat_align_loss/(step+1+epoch*dataloader.__len__()),  step // (dataloader.__len__() // 50) + epoch * 50)
                    self.writer.add_scalar("training_loss/Triplet",
                                    epoch_int_contrast_loss/(step+1+epoch*dataloader.__len__()),  step // (dataloader.__len__() // 50) + epoch * 50)

        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss/(step+1+epoch*dataloader.__len__())))

        train_loss[0] = epoch_loss
        train_loss[1] = epoch_feat_align_loss
        train_loss[2] = epoch_int_contrast_loss

        return lr

