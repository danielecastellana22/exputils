from abc import abstractmethod

from torch.utils.data import Dataset


class ListDataset(Dataset):

    def __init__(self, list):
        self.data = list

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class ConstValues:
    UNK = 0
    NO_ELEMENT = -1


class CollateFun:

    def __init__(self, device, **kwargs):
        self.device = device

    @abstractmethod
    def __call__(self, tuple_data):
        raise NotImplementedError('Must be implemented in subclasses')