import torch.nn as nn
import torch

from config import const

def init_data_attribute_ls(dataset_name):
    global user_attr_ls, item_attr_ls
    if dataset_name == 'commercial':
        user_attr_ls = ['id', 'gender', 'age', 'src_level']
        item_attr_ls = ['id', 'type1', 'cate']
    elif dataset_name == 'amazon':
        user_attr_ls = ['id']
        item_attr_ls = ['id']


class user_feat(nn.Module):
    def __init__(self):
        super().__init__()

        global user_attr_ls
        self.attr_ls = user_attr_ls

        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'user_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'user_{attr}_num'),
                    embedding_dim = getattr(const, f'user_{attr}_dim')
                )
            )
            self.size += getattr(const, f'user_{attr}_dim')

    def get_emb(self, sample):
        feats_ls = []
        for i, attr in enumerate(sample):
            feats_ls.append(
                getattr(self, f'user_{self.attr_ls[i]}_emb')(attr)
            ) 
        return torch.cat(feats_ls, dim=-1)



class item_feat(nn.Module):
    def __init__(self):
        super().__init__()

        global item_attr_ls
        self.attr_ls = item_attr_ls

        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'item_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'item_{attr}_num'),
                    embedding_dim = getattr(const, f'item_{attr}_dim'),
                    padding_idx = 0 if attr in ['id'] else None
                )
            )
            self.size += getattr(const, f'item_{attr}_dim')
        
    def get_emb(self, sample):
        '''
        Args:
            sample (list of tensors): item features
        
        Returns:
            item embedding
        '''
        feats_ls = []
        for i, attr in enumerate(sample):
            feats_ls.append(
                getattr(self, f'item_{self.attr_ls[i]}_emb')(attr)
            ) 
        return torch.cat(feats_ls, dim=-1)
        
class query_and_item_feat(nn.Module):
    def __init__(self):
        super().__init__()

        self.item_feat = item_feat()

        attr_ls = ['id', 'search_source', 'word_segs']

        for attr in attr_ls:
            setattr(
                self, f'query_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'query_{attr}_num'),
                    embedding_dim = getattr(const, f'query_{attr}_dim'),
                    padding_idx = 0 if attr in ['id', 'words_seg'] else None
                )
            )

        self.query_size =  const.query_word_segs_dim + const.query_id_dim

        self.query_trans = nn.Linear(self.query_size, self.item_feat.size) # align feature dimensinon
        self.item_trans = nn.Linear(self.item_feat.size, self.item_feat.size)
    

    def get_item_emb(self, sample):

        return self.item_trans( self.item_feat.get_emb(sample) )

    def get_search_session_emb(self, sample):
        '''get embedding for search behavior sequences used in training
        
        Args:
            sample: a list of input data

        Returns:
            query embedding (tensor): (Batch, search history length, dimension)
            query source embedding (tensor): (B, src_len, dim), categories of queries, for example, typed-in queries and suggested queries.
            query clicking items embedding: (B, src_len, number of clicking items, dim), itemd clicked by queries. if there is no clicked item, we collect the most relavant item.
            click item mask: (B, src_len, #click_items), 0 means padding
        '''
        
        query_id, search_source, click_item_ls, query_words = sample

        q_id_emb = self.query_id_emb(query_id) # batch, max_src_len, dim
        q_src_source_emb = self.query_search_source_emb(search_source) # batch, max_src_len, dim

        word_mask = torch.where(query_words==0,
                    0, 1).bool() #batch, max_src_len, max_query_word
        q_word_emb = self.query_word_segs_emb(query_words)  
        q_word_emb = torch.sum(torch.mul(q_word_emb, word_mask.unsqueeze(-1)), dim=-2) #batch, max_src_len, dim
        q_word_emb = q_word_emb / (
                                    torch.max( word_mask.sum(-1, keepdim=True),
                                            torch.ones_like( word_mask.sum(-1, keepdim=True) )
                                            )
                                )

        click_item_mask = torch.where(click_item_ls[0]==0,
                    0, 1).bool() #batch, max_src_len, max_click_item
        q_click_item_emb = self.get_item_emb(click_item_ls) #batch, max_src_len, max_click_item, dim


        query_emb = torch.cat([q_id_emb, q_word_emb], -1) # fuse query id and keywords
        query_emb = self.query_trans(query_emb)

        return [query_emb, q_src_source_emb, q_click_item_emb, click_item_mask]

    def get_query_emb(self, sample):
        """get query embedding to calculate infoNCE 

        Args:
            sample (list): [query id, query words]
                - query id (tensor): (B, src_len)
                - query words (tensor): (B, src_len, #words)

        Returns:
            query embedding (tensor): (B, src_len, dim)
        """
        
        query_id, query_words = sample
        
        q_id_emb = self.query_id_emb(query_id) # batch, max_src_len, dim

        word_mask = torch.where(query_words==0,
                    0, 1).bool() #batch, max_src_len, max_query_word
        q_word_emb = self.query_word_segs_emb(query_words)  
        q_word_emb = torch.sum(torch.mul(q_word_emb, word_mask.unsqueeze(-1)), dim=-2) #batch, max_src_len, dim
        q_word_emb = q_word_emb / (
                                    torch.max( word_mask.sum(-1, keepdim=True),
                                            torch.ones_like( word_mask.sum(-1, keepdim=True) )
                                            )
                                )

        query_emb = torch.cat([q_id_emb, q_word_emb], -1) # fuse query id and keywords
        query_emb = self.query_trans(query_emb)

        return query_emb
