# encoding: utf-8

import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np


class Config(object):

    def __init__(self, path_dataset, embedding):
        self.name = 'FastText'

        """ path
        """
        self.path_train  = path_dataset + '/data/train.txt'
        self.path_valid  = path_dataset + '/data/dev.txt'
        self.path_test   = path_dataset + '/data/test.txt'
        self.path_labels = path_dataset + '/data/class.txt'
        self.path_vocab  = path_dataset + '/data/vocab.pkl'

        self.path_pretrained_embedding = path_dataset + '/data/'       + embedding
        self.path_model                = path_dataset + '/saved_dict/' + self.name + '.ckpt'
        self.path_log                  = path_dataset + '/log/'        + self.name

        """ basic info
        """

        self.labels = self.load_labels()
        self.pretrained_embedding = self.load_pretrained_embedding() if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """ parameters
        """
        self.labels_num     = len(self.labels)
        self.vocab_num      = 0
        self.vocab_gram_num = 250499
        self.pretrained_embedding_dim = self.pretrained_embedding.size(1) if self.pretrained_embedding is not None else 300

        self.epoches_num   = 20
        self.pad_size      = 32
        self.batch_size    = 128
        self.hidden_size   = 256
        self.dropout       = 0.5
        self.learning_rate = 1e-3

        self.batch_num_reqimp = 1000


    def load_labels(self):
        return [l.strip() for l in open(self.path_labels, encoding='utf-8').readlines()]


    def load_pretrained_embedding(self):
        word_embeddings = np.load(self.path_pretrained_embedding)["word_embeddings"].astype('float32')
        return torch.tensor(word_embeddings)




class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        """ embedding
        """
        # TODO:
        # nn.Embedding
        if config.pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrained_embedding, freeze=False)
        else:
            # TODO:
            # padding_idx
            self.embedding = nn.Embedding(config.vocab_num, config.pretrained_embedding_dim, padding_idx=config.vocab_num-1)

        self.embedding_ngram2 = nn.Embedding(config.vocab_gram_num, config.pretrained_embedding_dim)
        self.embedding_ngram3 = nn.Embedding(config.vocab_gram_num, config.pretrained_embedding_dim)

        self.dropout = nn.Dropout(config.dropout)

        # TODO:
        # nn.Linear
        # dim * 3
        self.hidden = nn.Linear(config.pretrained_embedding_dim * 3, config.hidden_size)
        self.output = nn.Linear(config.hidden_size                 , config.labels_num)

    def forward(self, X):
        # TODO:
        # X

        embedding         = self.embedding(X[0])
        embedding_bigram  = self.embedding_ngram2(X[2])
        embedding_trigram = self.embedding_ngram3(X[3])

        # TODO:
        # torch.cat
        embeddings = torch.cat((embedding, embedding_bigram, embedding_trigram), -1)

        embeddings_mean = embeddings.mean(dim=1)
        embeddings_mean = self.dropout(embeddings_mean)

        hidden = self.hidden(embeddings_mean)
        hidden = F.relu(hidden)

        return self.output(hidden)
