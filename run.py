# eocding: utf-8

import time
import torch
import argparse
import numpy as np

# from importlib      import import_module
# from utils          import build_dataset
# from utils          import build_iterator
# from utils          import get_time_dif
# from utils_fasttext import build_dataset_fasttext
# from utils_fasttext import build_iterator_fasttext
# from train_eval     import train
# from train_eval     import init_network

import fasttext

from data  import load_dataset
from data  import build_dataset
from data  import build_iterator
from model import Model



def parse_embedding(args, embedding='embedding_SougouNews.npz'):

    """
    Parameters
    ----------
    embedding: str
        embedding_SougouNews.npz, 搜狗新闻
        embedding_Tencent.npz, 腾讯
        random, 随机初始化
    """
    
    if 'random' == args.embedding or 'FastText' == args.model:
        embedding = 'random'

    return embedding



def parse_module(args):

    return import_module('models.' + args.model)



def init_config(module, dataset, embedding):

    return module.Config(dataset, embedding)



def random_seed(seed=1):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def load_dataset(args, config):

    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter   = build_iterator(dev_data, config)
    test_iter  = build_iterator(test_data, config)

    return vocab, train_iter, dev_iter, test_iter



def fit(args):

    """ Model
    """
    # module    = parse_module(args)
    module    = fasttext
    embedding = parse_embedding(args)
    config    = init_config(module, args.dataset, embedding)

    random_seed()

    vocab, train_iter, dev_iter, test_iter = load_dataset(args, config)

    config.vocab_num = len(vocab)
    # model = module.Model(config).to(config.device)
    ft  = module.FastText(config).to(config.device)
    mdl = Model()

    if 'Transformer' != args.model:
        # init_network(model)
        mdl.init_network(ft)

    # print(model.parameters)
    # train(config, model, train_iter, dev_iter, test_iter)

    print(ft.parameters)
    mdl.fit_predict(config, ft, train_iter, dev_iter, test_iter)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model'    , type=str , required= True         , help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    parser.add_argument('--embedding', type=str , default = 'pre_trained', help='random or pre_trained')
    parser.add_argument('--dataset'  , type=str , default = 'THUCNews')
    parser.add_argument('--word'     , type=bool, default = False        , help='True for word, False for char')
    args = parser.parse_args()

    fit(args)
