#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from utils.metrics import judger as judge
from utils.data import get_dataloader

import torch

from tqdm import tqdm
import numpy as np


class Tester(object):

    def __init__(self, flags_obj, recommender, writer):

        self.recommender = recommender
        self.flags_obj = flags_obj
        self.judger = judge()
        self.writer = writer
        self.results = {}


    def set_dataloader(self, dst, **kwargs):

        self.num_neg = 99
        if self.flags_obj.dataset_name == 'commercial':
            self.batch_size =  (self.num_neg+1) * 50 
        elif self.flags_obj.dataset_name == 'amazon':
            self.batch_size =  (self.num_neg+1) * 1000 
        self.dataloader =   get_dataloader(
                                dst, self.batch_size,
                                prefetch_factor = self.batch_size // 16 * 2, num_workers = 16,
                                **kwargs
                            )


    @torch.no_grad()
    def test(self):
        """Evaluate the model on validation/test data set
        
        Args:
            total_loss: store total loss for all epochs. only used for validation
            epoch: number of current epoch

        Returns:
            results: dict of evaluation metrics 
        """

        self.recommender.model.eval()

        users, preds, labels, group_preds, group_labels = self._run_eval(
            group_size = (1 + self.num_neg)
        )
        
        res_pairwise = self.judger.cal_metric(
            group_preds, group_labels, ['ndcg@5;10', 'hit@1;5;10','mrr']
        )
        self.results.update(res_pairwise)

        return self.results

    def _run_eval(self, group_size):
        """ making prediction and group results

        Args:
            group_size: 100 for validation/test set(# negative samples is 99); 
        """
        users, preds, labels, group_preds, group_labels = [],[],[],[],[]

        for batch_data in tqdm(iterable=self.dataloader, mininterval=1, ncols=100):
            step_pred = self.recommender.predict(batch_data).squeeze(-1).cpu().numpy()
            step_label = batch_data[-1].cpu().numpy()
            step_user = batch_data[0][0].cpu().numpy() # u_id in user

            users.extend(np.reshape(step_user, -1))
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_label, -1))
            group_preds.extend(np.reshape(step_pred, (-1, group_size)))
            group_labels.extend(np.reshape(step_label, (-1, group_size)))

        # cpu() results in that gpu memories are not auto-collected
        # this command frees memory in Nvidia-smi
        if self.recommender.device != torch.device('cpu'):
            torch.cuda.empty_cache() 
        
        return users, preds, labels, group_preds, group_labels
