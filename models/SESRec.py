import torch
import torch.nn as nn
import torch.nn.functional as F

from .Inputs import query_and_item_feat, user_feat
from .module import FullyConnectedLayer
from config import const
import copy


class SESRec(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.query_item_feat = query_and_item_feat()
        
        
        self.user_feat = user_feat()

        self.rec_his_transformer_layer = TransformerEncoder(
                                            n_layers=config['num_blocks'],
                                            n_heads=config['num_heads'],
                                            hidden_size=self.query_item_feat.item_feat.size,
                                            inner_size=self.query_item_feat.item_feat.size,
                                            hidden_dropout_prob=config['dropout'],
                                        )
        self.src_his_transformer_layer = TransformerEncoder(
                                            n_layers=config['num_blocks'],
                                            n_heads=config['num_heads'],
                                            hidden_size=self.query_item_feat.item_feat.size,
                                            inner_size=self.query_item_feat.item_feat.size,
                                            hidden_dropout_prob=config['dropout'],
                                        )
        
        self.rec_pos_emb = PositionalEmbedding(const.max_rec_his_len, self.query_item_feat.item_feat.size)
        self.src_pos_emb = PositionalEmbedding(const.max_src_his_len, self.query_item_feat.item_feat.size)

        self.interest_contrast = Interest_Contrast(hid_dim=self.query_item_feat.item_feat.size,
                                            margin=config['triplet_margin'])

        self.src_his_att_pooling = Target_Attention(
                                        self.query_item_feat.item_feat.size,
                                        self.query_item_feat.item_feat.size
                                        )
        self.rec_his_att_pooling = Target_Attention( 
                                        self.query_item_feat.item_feat.size,
                                        self.query_item_feat.item_feat.size
                                        )


        self.fc_layer = FullyConnectedLayer(input_size = 7 * self.query_item_feat.item_feat.size + self.user_feat.size,
                                            hidden_unit=config['pred_hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='relu',
                                            dropout=config['dropout'],
                                            )

        self.loss_func = nn.BCELoss()

        self._init_weights()

        # init temperature of infoNCE as specific value
        self.feature_alignment = feature_align(config['infoNCE_temp'], self.query_item_feat.item_feat.size)

    def _init_weights(self):
        # weight initialization xavier_normal (a.k.a glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)


    def src_feat_process(self, src_feat, align_loss_input=None):
        """process search history features and calculate infoNCE loss for query-item alignment

        Args:
            src_feat (list of tensors): 
                query embedding (tensor): (Batch, search history length, dimension)
                query source embedding (tensor): (B, src_len, dim)
                query clicking items embedding: (B, src_len, number of clicking items, dim)
                clicked item mask: (B, src_len, #click_items) 
            align_loss_input (list of tensors), optional(only used for training): 
                randomly sampeld negative samples for infoNCE loss
                    negative item embeddings (tensor): (#neg, dim)
                    negative query embeddings (tensor): (#neg, dim)

        Returns:
            query_his_embedding (tensor): (B, src_len, dim), query history embedding
            click_item_his_emb (tensor): (B, src_len, dim), after mean pooling for all clicked items for each query
            align_loss (tensor): (,) the value of infoNCE loss, only calculated in training 
        """

                                                      #batch,max_src_len,max_click_item  #batch, max_src_len, 1
        query_emb, q_src_source_emb, q_click_item_emb, click_item_mask = src_feat

        mean_click_item_emb = torch.sum(torch.mul(q_click_item_emb, click_item_mask.unsqueeze(-1)), dim=-2) #batch, max_src_len, dim
        mean_click_item_emb = mean_click_item_emb / (
                                                        torch.max( click_item_mask.sum(-1, keepdim=True),
                                                                torch.ones_like( click_item_mask.sum(-1, keepdim=True) )
                                                                )
                                                    )

        query_his_emb = query_emb + q_src_source_emb
        click_item_his_emb = mean_click_item_emb

        if align_loss_input is None:
            return query_his_emb, click_item_his_emb
        else:
            align_loss = self.feature_alignment(align_loss_input, query_emb, click_item_mask, q_click_item_emb)
            return query_his_emb, click_item_his_emb, align_loss
        

    def parse_input_train(self, input_data):
        '''process raw data into dense vectors

        Args:
            input_data (list of tensors): 

        Returns: 
            pos_item_emb (tensor): (B, dim), target item embedding for positive feedbacks
            neg_item_emb(tensor): (B, #neg, dim), target item embedding for negative feedbacks
            rec_his_emb (tensor): (B, rec_len, dim), embedding for each recommendation history
            rec_his_mask (tensor): (B, rec_len), 'True' denotes padded value
            query_his_emb (tensor): (B, src_len, dim), embedding for queries of search history
            click_item_his_emb (tensor): (B, src_len, #click_item, dim), embedding for clicking items of each query in search history
            src_his_mask (tensor): (B, src_len), 'True' denotes padded value
            user_emb (tensor): (B, dim), user profile embedding
            align_loss (tensor): the value of infoNCE loss, only calculated in training
        '''

        #batch, feature
        user, rec_his, src_his, pos_item, neg_item, align_neg_item, align_neg_query = input_data

        user_emb = self.user_feat.get_emb(user)
        pos_item_emb = self.query_item_feat.get_item_emb(pos_item)
        neg_item_emb = self.query_item_feat.get_item_emb(neg_item)
        #batch, sequence, feature
        rec_his_emb = self.query_item_feat.get_item_emb(rec_his)
        rec_his_mask = torch.where(
                            rec_his[0]==0,
                            1, 0).bool()

        align_neg_item = self.query_item_feat.get_item_emb(align_neg_item)
        align_neg_query = self.query_item_feat.get_query_emb(align_neg_query)

        query_his_emb, click_item_his_emb, align_loss = self.src_feat_process( self.query_item_feat.get_search_session_emb(src_his),
                                                         [align_neg_item, align_neg_query]
                                                        )

        src_his_mask = torch.where(
                            src_his[0]==0,
                            1, 0).bool()

        return pos_item_emb, neg_item_emb, rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask, user_emb, align_loss

    def parse_input_test(self, input_data):
        #batch, feature
        user, rec_his, src_his, item = input_data

        user_emb = self.user_feat.get_emb(user)
        item_emb = self.query_item_feat.get_item_emb(item)
        #batch, sequence, feature
        rec_his_emb = self.query_item_feat.get_item_emb(rec_his)
        rec_his_mask = torch.where(
                            rec_his[0]==0,
                            1, 0).bool()

        query_his_emb, click_item_his_emb = self.src_feat_process( self.query_item_feat.get_search_session_emb(src_his) )

        src_his_mask = torch.where(
                            src_his[0]==0,
                            1, 0).bool()

        return item_emb, rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask, user_emb

     
    def his2feats(self, rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask):
        """modeling recommendation and search history into features

        Args:
            rec_his_emb: (B, rec_len, dim), browsing items' embedding
            rec_his_mask: (B, rec_len), "True" means padding
            query_his_emb: (B, src_len, dim), issued queries' embedding
            click_item_his_emb: (B, src_len, dim), clicked items' embedding (after mean pooling)
            src_his_mask: (B, src_len), "True" means padding

        Return:
            rec_his_emb: (B, rec_len, dim)
            src_his_emb: (B, src_len, dim)
        """

        rec_his_emb += self.rec_pos_emb(rec_his_emb)
        query_his_emb += self.src_pos_emb(query_his_emb)

        rec_his_emb = self.rec_his_transformer_layer(rec_his_emb, rec_his_mask) # batch, max_len, dim
        src_his_emb = self.src_his_transformer_layer(query_his_emb+click_item_his_emb, src_his_mask) # batch, max_len, dim

        return rec_his_emb, src_his_emb

     
    def inter_pred(self, user_feats, item_emb):
        '''feature interaction for prediction.
            - aggregate user interests by multiple-interest extraction.
            - make predictions using a two-layer MLP

        Args:
            user_feats (list of tensors): 
                rec_his_emb, src_his_emb: (B, seq_len, dim)
                rec_his_mask, src_his_mask: (B, seq_len)
                user_emb: (B, dim)
                mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg: (B, seq_len)
        
        Returns:
            logits: (B), predictions.
        '''
        rec_his_emb, rec_his_mask, src_his_emb, src_his_mask, user_emb, \
            mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg = user_feats
        
        
        rec_his_target_emb = self.rec_his_att_pooling(rec_his_emb, item_emb, rec_his_mask, mask_rec_pos, mask_rec_neg)
        src_his_target_emb = self.src_his_att_pooling(src_his_emb, item_emb, src_his_mask, mask_src_pos, mask_src_neg)

        inter_input = torch.cat([rec_his_target_emb, src_his_target_emb, user_emb, item_emb], -1)

        return self.fc_layer(inter_input).squeeze(-1) 

    def forward(self, input_data):
        '''forward pass used for training

        Args:
            input data (list of tensors)

        Returns:
            BCEloss: binary cross entropy loss 
            query-item alignment loss: infoNCE loss
            interest contrast loss: triplet loss
        '''
    
        pos_item_emb, neg_item_emb, rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask, user_emb, feat_align_loss\
             = self.parse_input_train(input_data)

        
        rec_his_emb, src_his_emb = self.his2feats(rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask)

        int_contrast_loss, pos_neg_mask = self.interest_contrast(rec_his_emb, src_his_emb, rec_his_mask, src_his_mask)

        mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg = pos_neg_mask

        user_feats = [rec_his_emb, rec_his_mask, src_his_emb, src_his_mask, user_emb,\
                        mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg]
        
        repeat_user_feats = [  torch.repeat_interleave(user_f, neg_item_emb.size(1), dim=0)
                                for user_f in user_feats
                            ]           
                                 
        
        pos_logits = self.inter_pred(user_feats, pos_item_emb)# batch
        neg_logits = self.inter_pred(repeat_user_feats, neg_item_emb.reshape(-1, neg_item_emb.size(-1))) # batch * #neg

        logits = torch.cat([pos_logits, neg_logits], 0)
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:pos_logits.size(0)] = 1.0

        return self.loss_func(logits, labels), feat_align_loss, int_contrast_loss

    def predict(self, input_data):
        '''make predictions for testing

        Args:
            input data (list of tensors)

        Returns:
            logits : predicting scores
        '''
    
        item_emb, rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask, user_emb = self.parse_input_test(input_data)

        rec_his_emb, src_his_emb = self.his2feats(rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask)

        pos_neg_mask = self.interest_contrast(rec_his_emb, src_his_emb, rec_his_mask, src_his_mask, return_loss=False)

        mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg = pos_neg_mask

        user_feats = [rec_his_emb, rec_his_mask, src_his_emb, src_his_mask, user_emb,\
                        mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg] 

        logits = self.inter_pred(user_feats, item_emb)# batch

        return logits




class infoNCE(nn.Module):
    
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp_init)
        
        self.weight_matrix = nn.Parameter(torch.randn((hdim,hdim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.tanh = nn.Tanh()

    def calculate_loss(self, query, item, neg_item):

        positive_logit = torch.sum( (query @ self.weight_matrix) * item, dim=1, keepdim=True)
        negative_logits = (query @ self.weight_matrix) @ neg_item.transpose(-2, -1)

        positive_logit, negative_logits = self.tanh(positive_logit), self.tanh(negative_logits)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits / self.temp, labels, reduction='mean')

     
    def forward(self, query, click_item, neg_item, neg_query):
        '''
        Args:
            query: matrix (#item, dim) 
            click item: matrix (#item, dim)
            neg item: matrix (#neg, dim)
            neg query: matrix (#neg, dim)

        Returns: loss
            infoNCE loss: (,)
        '''


        query_loss = self.calculate_loss(query, click_item, neg_item)
        item_loss = self.calculate_loss(click_item, query, neg_query)

        return 0.5 * (query_loss + item_loss)


class feature_align(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.infoNCE_loss = infoNCE(temp_init, hdim)

    def filter_user_src_his(self, qry_his_emb, click_item_mask, click_item_emb):
        '''process data to construct query-clicking item pairs.
            For issued queries, expand query embeddings to all their clicking items and filter out paddings.
            For clicked items, filter out paddings
        
        Args:
            qry_his_emb: (B, seq_len, dim)
            click_item_mask: (B, seq_len, max_click_item_num)
            click_item_emb: (B, seq_len, max_click_item_num, dim)
        
        Returns:
            src_his_query_emb, src_his_click_item_emb: (#item, dim)
        '''
        
        qry_his_emb = qry_his_emb.unsqueeze(2).expand(-1, -1, click_item_mask.size(2), -1)

        src_his_query_emb = torch.masked_select(qry_his_emb, click_item_mask.unsqueeze(-1)).reshape(-1, qry_his_emb.size(-1))
        src_his_click_item_emb = torch.masked_select(click_item_emb, click_item_mask.unsqueeze(-1))\
                                                                        .reshape(-1, click_item_emb.size(-1))

        return src_his_query_emb, src_his_click_item_emb

    def forward(self, align_loss_input, query_emb, click_item_mask, q_click_item_emb):
        neg_item_emb, neg_query_emb = align_loss_input 
        src_his_query_emb, src_his_click_item_emb = self.filter_user_src_his(query_emb, click_item_mask, q_click_item_emb)

        align_loss = self.infoNCE_loss(src_his_query_emb, src_his_click_item_emb, neg_item_emb, neg_query_emb)

        return align_loss



class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = nn.LeakyReLU()

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)


    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim = hidden_size, num_heads = n_heads, batch_first = True
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor,  attention_mask):
        attention_output, _ = self.multi_head_attention(
            query = input_tensor, key = input_tensor, value = input_tensor,
            key_padding_mask = attention_mask, #ignore padded places with True
            need_weights=False
        )

        attention_output = self.dropout(attention_output)
        return self.LayerNorm(input_tensor + attention_output)


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps
        ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 1
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.2
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=1,
        n_heads=2,
        hidden_size=60,
        inner_size=64,
        hidden_dropout_prob=0.2,
        layer_norm_eps=1e-8
    ):

        super().__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask):
        """
        Args
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
        """
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


class CoAttention(nn.Module):
    def __init__(self, embed_dim=100):
        super().__init__()

        self.embed_dim = embed_dim

        self.W1 = nn.parameter.Parameter( torch.rand((self.embed_dim, self.embed_dim)) )
        self.Wq = nn.parameter.Parameter( torch.randn((1, self.embed_dim)) )
        self.Wd = nn.parameter.Parameter( torch.randn((1, self.embed_dim)) )
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.Wd)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

     
    def forward(self, query, doc, query_mask=None, doc_mask=None): 
        '''calculating co-attention scores which indicate the similarity of the common interests and the corresponding element

        Args:
            query, doc: (B, seq_len, dim)
            query_mask, doc_mask: (B, seq_len)

        Return: attention scores
            Aq, Ad: (B, 1, seq_len)
        '''
        query_trans = query.transpose(2, 1)
        doc_trans = doc.transpose(2, 1)
        L = self.tanh(torch.matmul(torch.matmul(query, self.W1), doc_trans)) # batch, max_s_query, max_s_doc
        L_trans = L.transpose(2, 1) # DWQ_T  batch, max_s_doc, max_s_query

        score_d = torch.matmul(torch.matmul(self.Wq, query_trans), L) #batch, 1, max_s_doc
        score_q = torch.matmul(torch.matmul(self.Wd, doc_trans), L_trans) #batch, 1, max_s_query

        score_d = score_d.masked_fill(doc_mask.unsqueeze(1), torch.tensor(-1e12))
        score_q = score_q.masked_fill(query_mask.unsqueeze(1), torch.tensor(-1e12))

        Aq = self.softmax(score_q) # [batchsize, 1, max_s_query]
        Ad = self.softmax(score_d) # [batchsize, 1, max_s_doc]

        return Aq, Ad


class Interest_Contrast(nn.Module):
    def __init__(self, hid_dim, margin = 1.0):
        super().__init__()

        self.co_att = CoAttention(hid_dim)
        self.sigmoid = nn.Sigmoid()

        self.__init_Contrastive_Loss_(margin)

    def __init_Contrastive_Loss_(self, margin):

        self.trip_loss = nn.TripletMarginWithDistanceLoss(
            # default distance function is Eucdu
            margin = margin
        )
    
    def pooling(self, ele_set, mask):
        '''mean pooling for positives and negatives
        
        Args:
            ele_set: (batch, seq_len, dim)
            mask: (batch, seq_len)
        
        Returns:
            vector: (Batch, dim)
        '''
        mask = mask[:,None]
        len = mask.sum(-1)
        res = (ele_set * mask.transpose(2,1)).sum(1) / torch.max(len, torch.ones_like(len)) 
        return res #batch, dim
 
    def filter_pos_neg(self, w_rec, w_src, rec_mask, src_mask):
        '''select weights with gate 1/n. 
            weights >= 1/n -> positive
            weights < 1/n -> negative
        
        Args:
            w_rec, w_src: (B, 1, seq_len). attention scores
            rec_mask, src_mask: (B, seq_len), mask for padding values, "True" means padding

        Returns:
            mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg: (B, seq_len). positive and negative position masks for reco and src
        '''
        # batch, max_s_sequence          #batch, 1, max_s_seq
        w_rec, w_src = w_rec.squeeze(1), w_src.squeeze(1)
        rec_len, src_len = (~rec_mask).sum(-1, keepdim=True), (~src_mask).sum(-1, keepdim=True)
        rec_len, src_len = rec_len.expand(-1, w_rec.size(1)), src_len.expand(-1, w_src.size(1))

        large_gate_rec, large_gate_src = w_rec >= (1 / rec_len), w_src >= (1 / src_len)
        less_gate_rec, less_gate_src = w_rec < (1 / rec_len), w_src < (1 / src_len)

        # batch, max_s_sequence
        mask_rec_neg = less_gate_rec.masked_fill(rec_mask, 0.)
        mask_src_neg = less_gate_src.masked_fill(src_mask, 0.)

        mask_rec_pos = large_gate_rec.masked_fill(rec_mask, 0.)
        mask_src_pos = large_gate_src.masked_fill(src_mask, 0.)

        # Make sequences have at least one element for postive (negativeçc) samples
        # lenght == 1 or all weights equal 1/len may lead to all values False
        mask_rec_neg = torch.where(~mask_rec_neg.sum(-1, keepdim=True).bool(), ~rec_mask, mask_rec_neg)
        mask_rec_pos = torch.where(~mask_rec_pos.sum(-1, keepdim=True).bool(), ~rec_mask, mask_rec_pos)
        mask_src_neg = torch.where(~mask_src_neg.sum(-1, keepdim=True).bool(), ~src_mask, mask_src_neg)
        mask_src_pos = torch.where(~mask_src_pos.sum(-1, keepdim=True).bool(), ~src_mask, mask_src_pos)


        return mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg

    def forward(self, rec_seq, src_seq, rec_mask, src_mask, return_loss=True):
        '''
        Args:
            src_seq, rec_seq: (batch, seq, dim)
            src_mask,rec_mask: (batch, seq), True means padding
            return_loss(bool): whether return loss
        '''

        #get co-attention weights
        w_rec, w_src = self.co_att(rec_seq, src_seq, rec_mask, src_mask) #B, 1, max_s_rec.  B, 1, max_s_src

        #note: these masks set valuable elements with True, different with rec_mask&src_mask
        mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg =\
             self.filter_pos_neg(w_rec, w_src, rec_mask, src_mask) 

        if return_loss:

            anchor_rec, anchor_src = w_rec @ rec_seq, w_src @ src_seq      #B, 1, dim
            
            rec_loss = self.trip_loss(  anchor_rec.squeeze(1), 
                                        self.pooling(rec_seq, mask_rec_pos),
                                        self.pooling(rec_seq, mask_rec_neg)  )
            
            src_loss = self.trip_loss(  anchor_src.squeeze(1), 
                                        self.pooling(src_seq, mask_src_pos),
                                        self.pooling(src_seq, mask_src_neg)  )

            return rec_loss + src_loss, [mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg]
        else:

            return [mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg]
        

class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

     
    def forward(self, seq_emb, target, mask, pos_mask, neg_mask):
        '''
        Args:
            seq_emb: batch, seq_length, dim1
            target: batch, dim2
            mask: batch, seq_length. True means padding
            pos_mask: batch, seq_length. True means postive elements
            neg_mask: batch, seq_length. True means negative elements
        '''

        score = torch.matmul(seq_emb, self.W) #batch, seq, dim2
        score = torch.matmul(score, target.unsqueeze(-1)) #batch, seq, 1
        
        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1)) #batch,1,seq
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1) #batch, dim1

        pos_score = score.masked_fill((~pos_mask).unsqueeze(-1), torch.tensor(-1e16))
        pos_weight = self.softmax(pos_score.transpose(-2, -1)) #batch,1,seq
        pos_vec = torch.matmul(pos_weight, seq_emb).squeeze(1) #batch, dim1

        neg_score = score.masked_fill((~neg_mask).unsqueeze(-1), torch.tensor(-1e16))
        neg_weight = self.softmax(neg_score.transpose(-2, -1)) #batch,1,seq
        neg_vec = torch.matmul(neg_weight, seq_emb).squeeze(1) #batch, dim1

        res = torch.cat([all_vec, pos_vec, neg_vec], dim=-1)

        return res
