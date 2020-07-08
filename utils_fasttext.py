# encoding: utf-8

import os
import time
import torch
import numpy  as np
import pickle as pkl

from tqdm     import tqdm
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
UNK            = '<UNK>'
PAD            = '<PAD>'



def tokenizer(towords):

    if towords
        return lambda sentence: sentence.split(' ')
    else:
        return lambda sentence: list(sentence)



def split(line, tokenizer):
    l = line.strip()
    if not l:
        return
    
    text, label = l.split('\t')
    label = int(label)
    words = tokenizer(text)

    return words, label



def build_vocab(file_path, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1):
    """ Create Vocabulary

    {word:index}, with unknown and pad
    

    Tips
    ----
    tdqm, show progress
    """

    vocab = {}

    with open(file_path, 'r', encoding='UTF-8') as f:

        # count
        for line in tqdm(f):
            l = line.strip()
            if not l:
                continue

            # content \t label
            content = l.split('\t')[0]

            # TODO:
            # change into collections.Counter
            for word in tokenizer(content):
                vocab[word] = vocab.get(word, 0) + 1

        vocab = filter(lambda wc: wc[1]>=min_freq, vocab.items()) # filter out < min frequency
        vocab = sorted(vocab, key=lambda wc: wc[1], reverse=True) # sort descending
        vocab = vocab[:max_size]                                  # cutoff by max size limit
        vocab = {w:i for i, w, _ in enumerate(vocab)}             # {word:count} --> {word:index}
        vocab.update({UNK:len(vocab), PAD:len(vocab)+1})          # add unknown and pad to vocab

    return vocab



def bog_vocab(config, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1):

    if os.path.exists(config.path_vocab):
        vocab = pkl.load(open(config.path_vocab, 'rb'))
    else:
        vocab = build_vocab(config.path_train, tokenizer, max_size, min_freq)
        pkl.dump(vocab, open(config.path_vocab, 'wb'))

    print(f"vocab size: {len(vocab)}")



def hash2(words_idx, i, num_vocab_ngram):
    t1 = words_idx[i-1] if i-1 >= 0 else 0
    return (t1 * 14918087) % num_vocab_ngram



def hash3(words_idx, i, num_vocab_ngram):
    # TODO:
    # why 14918087 and 18408749
    t1 = words_idx[i-1] if i-1 >= 0 else 0
    t2 = words_idx[i-2] if i-2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % num_vocab_ngram



def hash_bigram(words_idx, size, num_vocab_ngram):
    return [hash2(words_idx, i, num_vocab_ngram) for i in range(size)]



def hash_trigram(words_idx, size, num_vocab_ngram):
    return [hash3(words_idx, i, num_vocab_ngram) for i in range(size)]



def padding(words, size):
    length_words    = len(words)
    num_valid_words = min(length_words, size)

    if size > length_words:
        words.extend([PAD] * (size-length_words))
    else:
        words = words[:size]

    return words, num_valid_words



def words_to_index(words, vocab):
    return [vocab.get(word, vocab.get(UNK)) for word in words]



def load_dataset(path, vocab, pad_size=32):
    dataset = []

    with open(path, 'r', encoding='UTF-8') as f:

        for line in tqdm(f):
            words, label = split(line, tokenizer)                              # words
            words, num_valid_words = padding(words, pad_size)                  # padding
            words_idx = words_to_index(words, vocab)                           # words to index
            bigram  = hash_bigram(words_idx , pad_size, config.vocab_gram_num) # bi-gram
            trigram = hash_trigram(words_idx, pad_size, config.vocab_gram_num) # tri-gram

            dataset.append((words_idx, label, num_valid_words, bigram, trigram))

    return dataset



def build_dataset_fasttext(config, towords):
    """ Build Dataset
    """

    tokenizer = tokenizer(towords)
    vocab     = bog_vocab(config, tokenizer)

    train = load_dataset(config.path_train, vocab, config.pad_size)
    valid = load_dataset(config.path_valid  , vocab, config.pad_size)
    test  = load_dataset(config.path_test , vocab, config.pad_size)

    return vocab, train, valid, test



class DatasetIterater(object):

    def __init__(self, dataset, batch_size, device):
        self.dataset    = dataset

        self.batch_size     = batch_size
        self.batch_num      = len(dataset) // batch_size
        self.batch_residual = False if 0 == len(dataset) % self.batch_num else True

        self.device = device
        self.index  = 0


    def to_tensor(self, batch):
        words_idx       = torch.LongTensor([_[0] for _ in batch]).to(self.device)
        label           = torch.LongTensor([_[1] for _ in batch]).to(self.device)
        num_valid_words = torch.LongTensor([_[2] for _ in batch]).to(self.device)
        bigram          = torch.LongTensor([_[3] for _ in batch]).to(self.device)
        trigram         = torch.LongTensor([_[4] for _ in batch]).to(self.device)

        return (words_idx, num_valid_words, bigram, trigram), label


    def __next__(self):
        if self.batch_residual and self.index == self.batch_num:
            # last batch
            batch = self.dataset[self.index * self.batch_size: len(self.dataset)]
            batch = self.to_tensor(batch)
            self.index += 1
            return batch

        elif self.index >= self.batch_num:
            # out index
            self.index = 0
            raise StopIteration

        else:
            # normal batch
            batch = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self.to_tensor(batch)
            self.index += 1
            return batch


    def __iter__(self):
        return self


    def __len__(self):
        return self.batch_num+1 if self.batch_residual else self.batch_num



def build_iterator_fasttext(dataset, config):
    return DatasetIterater(dataset, config.batch_size, config.device)



if __name__ == "__main__":

    # take pretrained word vector

    path_vocab           = "./THUCNews/data/vocab.pkl"
    path_vocab_embedding = "./THUCNews/data/vocab.embedding.sougou"
    path_pretrained      = "./THUCNews/data/sgns.sogou.char"
    embedding_dim        = 300

    vocab = pkl.load(open(path_vocab, 'rb'))
    word_embeddings = np.random.rand(len(vocab), embedding_dim)

    with open(path_pretrained, "r", encoding='UTF-8') as f:
        for i, line in enumerate(f.readlines()):
            words = line.strip().split(" ")
            if words[0] in vocab:
                word_idx = vocab[words[0]]
                word_embeddings[word_idx] = np.asarray([float(x) for x in words[1:301]], dtype='float32')

    np.savez_compressed(path_vocab_embedding, word_embeddings=word_embeddings)
