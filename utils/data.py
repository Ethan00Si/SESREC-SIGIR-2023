import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from .data_sampler import *
from .data_utils import *

class BaseDataset(Dataset):
    
    def __init__(self):
        super(Dataset, self).__init__()


    def __len__(self):

        return self.sampler.record.shape[0]

    def __getitem__(self, index):

        raise NotImplementedError



class Pairwise_Dataset(BaseDataset):
    def __init__(self, dataset_file, flags_obj, is_train):
        super().__init__()
        self.sampler = Pair_Sampler(dataset_file, flags_obj, is_train=is_train)

    def __getitem__(self, index):
        '''
        Args: 
            index: index of interaction 
        Returns:
            users: list of attributes of users,  
                    commericial data: [ [id], [gender], [age], [search active level]]
                    amazon data: [[id]]
            rec_hiss: list of items' attributres in recommendation history, where the shape of tensors is (length of history, ) 
                    commericial data:[ tensor(id), tensor(author id), tensor(type 1), tensor(category)]
                    amazon data: [tensor(id)],
            pos_item: list of postive item's attributes
                    commericial data:[ [id], [author id], [type 1], [category]]
                    amazon data: [[id]]
            neg_items: list of negtive items' attributes, the shape of tensors is (number of negative samples, )
                    commericial data:[ tensor(id), tensor(author id), tensor(type 1), tensor(category)]
                    amazon data: [tensor(id)]
        '''
        users, rec_hiss, pos_item, neg_items = self.sampler.sample(index) #

        return users, rec_hiss, pos_item, neg_items

    


class Pairwise_w_src_Dataset(BaseDataset):
    def __init__(self, dataset_file, flags_obj, is_train):
        super().__init__()
        self.sampler = Pair_w_src_Sampler(dataset_file, flags_obj, is_train)

    def __getitem__(self, index):
        '''
        Args: 
            index: index of interaction file
        Returns:
            users, rec_hiss, pos_item, neg_items: the same as above 
            src_hiss: list of search behaviors' attributes
                [tensor(query ID), tensor(query source), [attributes of clicked items], tensor(query words)]
                - tensor(query ID): (length of history,)
                - tensor(query source): (length of history,)
                - [attributes of clicked items]: [ tensor(id), tensor(author id), tensor(type 1), tensor(category)], 
                    shape of each tensor is (length of history, maximum number of clicked items for one query)
                - tensor(query words): (length of history, maximum number of words for one query)
        '''
        users, rec_hiss, src_hiss, pos_item, neg_items = self.sampler.sample(index)

        return users, rec_hiss, src_hiss, pos_item, neg_items

class SESRec_neg_sample_Dataset(BaseDataset):
    def __init__(self,  dataset_file, flags_obj):
        super().__init__()
        self.sampler = Pair_w_src_Sampler(dataset_file, flags_obj, True)
        self.item_vocab = np.array( list(self.sampler.item_vocab.keys()) )
        np.random.shuffle(self.item_vocab)
        self.query_vocab = np.array( list(self.sampler.src_session_vocab.keys()) )
        np.random.shuffle(self.query_vocab)

        self.dst_name = flags_obj.dataset_name
        self.our_get_item = self.commercial__getitem__ if self.dst_name == 'commercial' else self.amazon__getitem__

    def __len__(self):
        return 10000000000000

    def __getitem__(self, index):
        return self.our_get_item(index)

    def commercial__getitem__(self, index):
        '''
        generate randomly sampled items and queries for the query-item alignment (infoNCE loss)
        Args:
            index: index of specific item and query
        Returns:
            list of item attributes
            list of query attributes
        '''
        
        item = self.item_vocab[index % self.item_vocab.size]
        item_id, item_type1, item_cate = self.sampler._get_item_info(item)

        query = self.query_vocab[index % self.query_vocab.size]
        query_id, _, _, query_words = self.sampler._get_src_info(query)

        return [item_id, item_type1, item_cate], [query_id, torch.tensor(query_words, dtype=torch.int64)]

    def amazon__getitem__(self, index):
        '''
        the same as above
        '''
        
        item = self.item_vocab[index % self.item_vocab.size]
        item_id, = self.sampler._get_item_info(item)

        query = self.query_vocab[index % self.query_vocab.size]
        query_id, _, _, query_words = self.sampler._get_src_info(query)

        return [item_id], [query_id, torch.tensor(query_words, dtype=torch.int64)]
        

class SESRec_Dataset(Pairwise_w_src_Dataset):
    def __init__(self, *args):
        super().__init__(*args)






GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_dataloader(data_set, bs, **kwargs):
    return DataLoader(  data_set, batch_size = bs,
                        shuffle=False, pin_memory = True, 
                        worker_init_fn=worker_init_fn, **kwargs
                    )

