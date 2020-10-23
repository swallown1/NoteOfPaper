import codecs
import regex
import numpy as np
import torch.utils.data as data
# from torch.utils.data import Dataset
from parse import parse_args
import torch
arg = parse_args()


class Data_source():
    def __init__(self):

        self.load_de_vocab()
        self.load_en_vocab()

    def load_de_vocab(self):
        vocab = [line.split()[0] for line in codecs.open('data/de.vocab.tsv', 'r', 'utf-8').read().splitlines()
                 if int(line.split()[1]) >= arg.min_cnt]
        self.de2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2de = {idx: word for idx, word in enumerate(vocab)}
        self.num_words_de = len(self.idx2de)

    def load_en_vocab(self):
        vocab = [line.split()[0] for line in codecs.open('data/en.vocab.tsv', 'r', 'utf-8').read().splitlines()
                 if int(line.split()[1]) >= arg.min_cnt]

        self.en2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2en = {idx: word for idx, word in enumerate(vocab)}
        self.num_words_en = len(self.idx2en)

    def create_data(self,source_sents, target_sents):
        x_list, y_list, Sources, Targets = [], [], [], []
        for source_sent, target_sent in zip(source_sents, target_sents):
            x = [self.de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
            y = [self.en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]

            if max(len(x), len(y)) <= arg.maxlen:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                Sources.append(source_sent)
                Targets.append(target_sent)

        # Pad
        X = np.zeros([len(x_list), arg.maxlen], np.int32)
        Y = np.zeros([len(y_list), arg.maxlen], np.int32)

        for i, (x, y) in enumerate(zip(x_list, y_list)):
            X[i] = np.lib.pad(x, [0, arg.maxlen - len(x)], 'constant', constant_values=(0, 0))
            Y[i] = np.lib.pad(y, [0, arg.maxlen - len(y)], 'constant', constant_values=(0, 0))
        return X, Y, Sources, Targets

    def load_train_data(self):
        def _refine(line):
            line = regex.sub("[^\s\p{Latin}']", "", line)
            return line.strip()

        de_sents = [_refine(line) for line in codecs.open(arg.source_train, 'r', 'utf-8').read().split('\n') if
                    line and line[0] != "<"]
        en_sents = [_refine(line) for line in codecs.open(arg.target_train, 'r', 'utf-8').read().split('\n') if
                    line and line[0] != '<']

        X, Y, Sources, Targets = self.create_data(de_sents, en_sents)
        return X, Y

    def load_test_data(self):
        def _refine(line):
            line = regex.sub("<[^>]+>", "", line)
            line = regex.sub("[^\s\p{Latin}']", "", line)
            return line.strip()

        de_sents = [_refine(line) for line in codecs.open(arg.source_test, 'r', 'utf-8').read().split('\n') if
                    line and line[:4] == "<seg"]
        en_sents = [_refine(line) for line in codecs.open(arg.target_test, 'r', 'utf-8').read().split('\n') if
                    line and line[:4] == '<seg']

        X, Y, Sources, Targets = self.create_data(de_sents, en_sents)
        return X, Sources, Targets


class My_dataset(data.Dataset):
    def __init__(self,data_source,train=True):
        self.data_source = data_source
        if train:
            self.X,self.Y = self.data_source.load_train_data()
        # else:self.X,self.Y = load_test_data()

    def __getitem__(self, item):
        return self.X[item],self.Y[item]

    def __len__(self):
        return len(self.X)

# if __name__ == '__main__':
#     train_data = My_dataset()
#     train_loader = data.DataLoader(train_data,batch_size=4,shuffle=False)
#     for x, y in train_loader:
#         print(torch.ones_like(y[:,:1])*2 ,y[:,:-1])
#         print(torch.cat((torch.ones_like(y[:,:1])*2 ,y[:,:-1]),-1))
#         break