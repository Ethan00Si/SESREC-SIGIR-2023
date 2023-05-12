from .data_utils import NpyLoader, TsvLoader, JsonLoader, PickleLoader
import random
from config import const
import torch
import numpy as np


class Sampler(object):
    
    def __init__(self, dataset_file, load_path):

        self.load_path = load_path
        self.tsv_loader = TsvLoader(load_path)
        self.json_loader = JsonLoader(load_path)
        self.pickle_loader = PickleLoader(load_path)

        # user item interactions
        self.record = self.tsv_loader.load(filename=dataset_file, sep='\t')
    
    def sample(self, index, **kwargs):

        raise NotImplementedError



class Pair_Sampler(Sampler):
    def __init__(self, dataset_file, args, is_train):
        super().__init__(dataset_file, args.load_path)
        self.is_train = is_train
        self.dataset_name = args.dataset_name
        self.sample = self.train_sample if is_train else self.test_sample
        self.build(args)

    def build(self, args):
        self.num_negs = 0 if not self.is_train else args.num_negs

        self.user_vocab = self.pickle_loader.load(args.user_vocab)
        self.item_vocab = self.pickle_loader.load(args.item_vocab)

        self.items_with_popular = self.record['i_id'].tolist()

        self.max_rec_his = const.max_rec_his_len

        self._get_user_profile = self.commercial_get_user_profile if self.dataset_name == 'commercial' else self.amazon_get_user_profile
        self._get_item_info = self.commercial_get_item_info if self.dataset_name == 'commercial' else self.amazon_get_item_info

    def commercial_get_user_profile(self, user):
        
        return (self.user_vocab[user]['id'], self.user_vocab[user]['gender'], self.user_vocab[user]['age'], \
            self.user_vocab[user]['src_level'])

    def amazon_get_user_profile(self, user):

        return (self.user_vocab[user]['id'],)

    def commercial_get_item_info(self, item):
        if item == 0:
            # skip padding
            return (0, 0, 0)
        

        return (
            self.item_vocab[item]['id'],  self.item_vocab[item]['type1'], \
            self.item_vocab[item]['cate']
        )

    def amazon_get_item_info(self, item):
        if item == 0:
            # skip padding
            return (0,)
        
        return (self.item_vocab[item]['id'],)


    def _gen_neg_samples(self, postive_item, user_rec_his):
        count = 0
        neg_items = []
        while count < self.num_negs:
            neg_item = random.choice(self.items_with_popular)
            if  neg_item == postive_item or\
                neg_item in neg_items or\
                neg_item in user_rec_his:
                continue
            count += 1
            neg_items.append(neg_item)
            
        return neg_items

    def parse_line(self, index):

        line = self.record.iloc[index]

        user = line['u_id']
        pos_item = line['i_id']
        rec_his_num = int(line['rec_his'])
        src_his_num = int(line['src_his'])
        label = float(line['label'])

        return user, pos_item, rec_his_num, src_his_num, label

    def wrap_item_his_ls(self, item_ls):
        '''
        Args:
            item_ls: list of item original IDs
        '''
        item_ls = list(
           torch.tensor(list(elem), dtype=torch.int64) for elem in zip(*[self._get_item_info(it)  for it in item_ls])
        )

        return item_ls

    def train_sample(self, index):
        
        user, pos_item, rec_his_num, src_his_num, label = self.parse_line(index)

        rec_his = self.user_vocab[user]['rec_his'][:rec_his_num] 

        neg_items = []

        # neg_sample
        if self.num_negs > 0:
            neg_items = self._gen_neg_samples(pos_item, rec_his)

        # parse item features
        neg_items = self.wrap_item_his_ls(neg_items)

        pos_item = list(self._get_item_info(pos_item))
        # pos_item[-1] = torch.tensor(pos_item[-1], dtype=torch.int64)

        # parse user features 
        users = [ attr for attr in self._get_user_profile(user) ]

        # parse recommendation history 
        #      padding history
        rec_his = rec_his[-self.max_rec_his:]
        if len(rec_his) < self.max_rec_his:
            rec_his += [0]*(self.max_rec_his - len(rec_his))

        rec_hiss = self.wrap_item_his_ls(rec_his)

        return users, rec_hiss, pos_item, neg_items

    def test_sample(self, index):

        user, item, rec_his_num, src_his_num, label = self.parse_line(index)

        rec_his = self.user_vocab[user]['rec_his'][:rec_his_num] 

        item = list(self._get_item_info(item))

        # parse user features 
        users = [ attr for attr in self._get_user_profile(user) ]

        # parse recommendation history 
        #      padding history
        rec_his = rec_his[-self.max_rec_his:]
        if len(rec_his) < self.max_rec_his:
            rec_his += [0]*(self.max_rec_his - len(rec_his))

        rec_hiss = self.wrap_item_his_ls(rec_his)

        label = torch.tensor(label, dtype=torch.float32)

        return users, rec_hiss, item, label

class Pair_w_src_Sampler(Pair_Sampler):
    def __init__(self, *args):
        '''pairwise sampler with search data'''
        super().__init__(*args)
        
    def build(self, args):

        super().build(args)
        self.src_session_vocab = self.pickle_loader.load(args.src_session_vocab)

        self.max_src_click_item = const.max_src_click_item
        self.max_src_his = const.max_src_his_len
        self.max_query_word = const.max_words_of_query

    def _get_src_info(self, s_sess_id):
        if s_sess_id == '0':
            # skip padding
            click_item_ls = list(
                                    list(elem) for elem in zip(*[self._get_item_info(it)  for it in [0]*self.max_src_click_item])
                                )
            return (
                0, 0, click_item_ls, [0] * self.max_query_word
            )

        click_item_ls =  self.src_session_vocab[s_sess_id]['click_items'][:self.max_src_click_item]
        if len(click_item_ls) == 0: # if no click item, use the top 1 relevant item to replace
            click_item_ls = self.src_session_vocab[s_sess_id]['relevant_item']
        if len(click_item_ls) < self.max_src_click_item:
            click_item_ls += [0] * (self.max_src_click_item - len(click_item_ls))
        click_item_ls = list(
            list(elem) for elem in zip(*[self._get_item_info(it)  for it in click_item_ls])
        )

        query_words = self.src_session_vocab[s_sess_id]['keyword_seg'][:self.max_query_word]
        if len(query_words) < self.max_query_word:
            query_words += [0] * (self.max_query_word - len(query_words))

        return (
            self.src_session_vocab[s_sess_id]['query_id'], self.src_session_vocab[s_sess_id]['search_source'],
            click_item_ls, query_words
        )

    def wrap_src_his_ls(self, src_his):
        src_his = list(
            list(elem) for elem in zip(*[self._get_src_info(it)  for it in src_his])
        )
        need_tensor = [0,1,3]
        for idx in need_tensor:
            src_his[idx] = torch.tensor(src_his[idx], dtype=torch.int64)
        src_his[2] = [torch.tensor(item_attr, dtype=torch.int64) for item_attr in zip(*src_his[2])] #clicked item list
        
        return src_his

    def train_sample(self, index):

        user, pos_item, rec_his_num, src_his_num, label = self.parse_line(index)

        users, rec_hiss, pos_item, neg_items = super().train_sample(index)

        src_his = self.user_vocab[user]['src_his'][:src_his_num] 
        # parse search history 
        #      padding history
        src_his = src_his[-self.max_src_his:]
        if len(src_his) < self.max_src_his:
            src_his += ['0']*(self.max_src_his - len(src_his))

        src_hiss = self.wrap_src_his_ls(src_his)

        return users, rec_hiss, src_hiss, pos_item, neg_items


    def test_sample(self, index):

        user, item, rec_his_num, src_his_num, label = self.parse_line(index)

        users, rec_hiss, item, label = super().test_sample(index)

        src_his = self.user_vocab[user]['src_his'][:src_his_num] 
        # parse search history 
        #      padding history
        src_his = src_his[-self.max_src_his:]
        if len(src_his) < self.max_src_his:
            src_his += ['0']*(self.max_src_his - len(src_his))

        src_hiss = self.wrap_src_his_ls(src_his)

        return users, rec_hiss, src_hiss, item, label