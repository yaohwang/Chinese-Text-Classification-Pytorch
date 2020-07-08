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




class FastText(nn.Module):

    def __init__(self, config):
        super(FastText, self).__init__()

        """ nn.Embedding

        Parameters
        ----------
        freeze: bool 
            if true, the tensor does not get updated in the learning process

        padding_idx: int, optional
            with padding_idx set, the embedding vector at padding_idx is initialized to all zeros.
            The gradient for this vector from Embedding is always zero.
        """
        if config.pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrained_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_num, config.pretrained_embedding_dim, padding_idx=config.vocab_num-1)

        self.embedding_ngram2 = nn.Embedding(config.vocab_gram_num, config.pretrained_embedding_dim)
        self.embedding_ngram3 = nn.Embedding(config.vocab_gram_num, config.pretrained_embedding_dim)

        self.dropout = nn.Dropout(config.dropout)

        """ nn.Linear

        y = XW + b

        Parameters
        ----------
        in_features:
             size of each input sample

        out_features:
             size of each output sample

        bias:
             If set to False, the layer will not learn an additive bias. Default: True

        Tips
        ----
        config.pretrained_embedding_dim * 3 = words/chars embedding + bigram embedding + trigram embedding
        """
        self.hidden = nn.Linear(config.pretrained_embedding_dim * 3, config.hidden_size)
        self.output = nn.Linear(config.hidden_size                 , config.labels_num)



    def forward(self, X):
        # TODO:
        """
        Parameters
        ----------
        X: (words_idx, num_valid_words, bigram, trigram)

            words_idx --> list words, index, pad_size (padding or cutoff)
            bigram    --> bi-gram   , index, pad_size
            trigram   --> tri-gram  , index, pad_size

        N: number of samples

        P: pad size

        words_idx, (N, P)
        bigram   , (N, P)
        trigram  , (N, P)

        E: dim of Embedding

        then after Embedding

        words_idx, (N, P, E)
        bigram   , (N, P, E)
        trigram  , (N, P, E)

        """

        embedding_vector         = self.embedding(X[0])
        embedding_vector_bigram  = self.embedding_ngram2(X[2])
        embedding_vector_trigram = self.embedding_ngram3(X[3])

        """
        torch.cat

        embeddings_vector, (N, P, E+E+E)

        Parameters
        ----------
        dim: int
            0, row
            1, column
           -1, last dim, here means dim E
        """
        embeddings_vector = torch.cat((embedding_vector, embedding_vector_bigram, embedding_vector_trigram), -1)

        """
        torch.Tensor.mean

        embeddings_vector_mean, (N, E+E+E)

        fix sample, such as 0 in N
        fix Embedding, such as 5 in E+E+E

        then, mean(embeddings_vector_mean[0, :, 5])
        sample 0 words/chars mean over Embedding dim 5

        Parameters
        ----------
        dim: int
            0: mean(column), row index is changing
            1: mean(row), column index is changing

            so the other way to understand it is dim specify which dim index is changing
        """
        embeddings_vector_mean = embeddings_vector.mean(dim=1)

        """
        nn.Dropout(p)

        during training
        * randomly zeroes some of the elements of the input tensor with probability p
        * the outputs are scaled by a factor of 1/(1-p), during training

        during evaluation
        * the module simply computes an identity function.
          which means nop/pass.
        """
        # TODO: wheather stop backward
        embeddings_vector_mean = self.dropout(embeddings_vector_mean)

        hidden = self.hidden(embeddings_vector_mean)
        hidden = F.relu(hidden)

        return self.output(hidden)
