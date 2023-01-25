import collections
import string
import torch
import torchtext
from torch.nn.utils import rnn as torch_rnn
from torch.utils.data import Dataset

TOKEN_LEN_WARNING_THRES = 10000


class CmdLineTokenDataset(Dataset):
    def __init__(self, data_csv_path):
        alphabet_str = string.ascii_letters + string.digits + string.punctuation
        alphabet_counter = collections.Counter(list(alphabet_str))
        self.vocab = torchtext.vocab.Vocab(alphabet_counter)
        self.__load_data__(data_csv_path)

    def __len__(self):
        return len(self._data_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self._data_[idx]
        label = int(row[0])
        token_tensor = self.__str_to_tensor__(row[1])
        if token_tensor.shape[0] > TOKEN_LEN_WARNING_THRES:
            print('Warning: token too long', token_tensor.shape[0], len(idx), idx)
        return label, token_tensor

    def __str_to_tensor__(self, s):
        tensor = torch.zeros(len(s), len(self.vocab))
        for ci, c in enumerate(s):
            tensor[ci, self.vocab[c]] = 1  # one hot encoding
        return tensor

    def __load_data__(self, data_csv_path):
        self._data_ = []
        with open(data_csv_path) as data_file:
            for line in data_file:
                comma_idx = line.strip().find(',')
                label = line[0:comma_idx]
                token = line[comma_idx+1:]
                self._data_.append((label, token))

    def size(self):
        return len(self._data_)


class CmdLineTokenCollator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, tokens = zip(*batch)
        yy = torch.FloatTensor(labels).reshape(len(labels), 1)
        xx = torch_rnn.pad_sequence(tokens, padding_value=self.pad_idx)
        lengths = torch.LongTensor([len(x) for x in tokens])
        return yy, xx, lengths
