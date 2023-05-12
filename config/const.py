#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error


def init_dataset_setting_commercial():

    global load_path, ckpt, user_vocab, item_vocab, src_session_vocab, train_file, valid_file, test_file,\
        JSR_train_file, item_id_num, item_id_dim, item_type1_num, item_type1_dim, item_cate_num, item_cate_dim,\
            user_id_num, user_id_dim, user_gender_num, user_gender_dim, user_age_num, user_age_dim, user_src_level_num, user_src_level_dim,\
                query_id_num, query_id_dim, query_search_source_num, query_search_source_dim, query_word_segs_num, query_word_segs_dim,\
                    max_rec_his_len, max_words_of_query, max_src_his_len, max_src_click_item, item2query_vocab

    """data files info"""

    load_path = './data/commercial'
    ckpt = 'ckpt'

    user_vocab = 'vocab/user_vocab.pickle'
    item_vocab = 'vocab/item_vocab.pickle'
    src_session_vocab = 'vocab/s_session_vocab.pickle'


    train_file = 'dataset/train_inter.tsv'
    valid_file = 'dataset/valid_inter.tsv'
    test_file = 'dataset/test_inter.tsv'

    """item/user/query feature"""

    item_id_num = 822832 + 1 #zero for padding
    item_id_dim = 32 
    item_type1_num = 38
    item_type1_dim = 8
    item_cate_num = 37
    item_cate_dim = 8

    user_id_num = 35721
    user_id_dim = 16
    user_gender_num = 3
    user_gender_dim = 4
    user_age_num = 8
    user_age_dim = 4
    user_src_level_num = 4
    user_src_level_dim = 4

    query_id_num = 398924 + 1 #zero for padding
    query_id_dim = 32
    query_search_source_num = 4
    query_search_source_dim = 48
    query_word_segs_num = 116569 + 1 #zero for padding
    query_word_segs_dim = 32


    """experiment config"""
    max_rec_his_len = 150 
    max_words_of_query = 6 # maximum number of words in a query 
    max_src_his_len = 25
    max_src_click_item = 5 # maximum number of clicked items for a query in one session


def init_dataset_setting_amazon():
    
    global load_path, ckpt, user_vocab, item_vocab, src_session_vocab, train_file, valid_file, test_file,\
        JSR_train_file, item_id_num, item_id_dim, item_type1_num, item_type1_dim, item_cate_num, item_cate_dim,\
            user_id_num, user_id_dim, user_gender_num, user_gender_dim, user_age_num, user_age_dim, user_src_level_num, user_src_level_dim,\
                query_id_num, query_id_dim, query_search_source_num, query_search_source_dim, query_word_segs_num, query_word_segs_dim,\
                    max_rec_his_len, max_words_of_item, max_words_of_query, max_src_his_len, max_src_click_item, item2query_vocab


    """data files info"""

    load_path = './data/amazon'
    ckpt = 'ckpt'

    user_vocab = 'vocab/user_vocab.pickle'
    item_vocab = 'vocab/item_vocab.pickle'
    src_session_vocab = 'vocab/s_session_vocab.pickle'


    train_file = 'dataset/train_inter.tsv'
    valid_file = 'dataset/valid_inter.tsv'
    test_file = 'dataset/test_inter.tsv'

    """item/user/query feature"""

    item_id_num = 61934 + 1 #zero for padding
    item_id_dim = 32 

    user_id_num = 68223
    user_id_dim = 32

    query_id_num = 4298 + 1 #zero for padding
    query_id_dim = 16
    query_search_source_num = 1 #since there isn't search source for amazon dataset. this can be seen as a bias vector for all query embedding
    query_search_source_dim = 32
    query_word_segs_num = 1846 + 1 #zero for padding
    query_word_segs_dim = 16

    """experiment config"""
    max_rec_his_len = 15 
    max_words_of_query = 15 # maximum number of words in a query 
    max_src_his_len = 15
    max_src_click_item = 1