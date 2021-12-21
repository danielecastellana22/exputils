import os
from abc import abstractmethod, ABC
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from exputils.serialisation import from_pkl_file


class ConstValues:
    UNK = 0
    NO_ELEMENT = -1


class ListDataset(Dataset):

    def __init__(self, list):
        self.data = list

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class BaseDatasetLoader:

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def get_training_data_loader(self):
        raise NotImplementedError('Must be implemented in subclasses')

    @abstractmethod
    def get_validation_data_loader(self):
        raise NotImplementedError('Must be implemented in subclasses')

    @abstractmethod
    def get_test_data_loader_dict(self):
        raise NotImplementedError('Must be implemented in subclasses')

    @abstractmethod
    def __collate_fun__(self, data):
        raise NotImplementedError('Must be implemented in subclasses')

    def __get_data_loader__(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.__collate_fun__, shuffle=shuffle, num_workers=0)


class FileDatasetLoader(BaseDatasetLoader, ABC):

    def __init__(self, data_dir, max_tr_elements=None, **kwargs):
        super(FileDatasetLoader, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.max_tr_elements = max_tr_elements

    def get_training_data_loader(self):

        train_set = ConcatDataset(self.__load_list_of_datasets__('train').values())

        if self.max_tr_elements:
            train_set = train_set[:self.max_tr_elements]

        return self.__get_data_loader__(train_set, shuffle=True)

    def get_validation_data_loader(self):
        val_set = ConcatDataset(self.__load_list_of_datasets__('validation').values())
        return self.__get_data_loader__(val_set, shuffle=False)

    def get_test_data_loader_dict(self):
        return {k: self.__get_data_loader__(v, shuffle=False) for k, v in self.__load_list_of_datasets__('test').items()}

    def __load_list_of_datasets__(self, tag):
        out_dict = {}
        for f in os.listdir(self.data_dir):
            if tag in f:
                out_dict[f] = ListDataset(from_pkl_file(os.path.join(self.data_dir, f)))
        return out_dict

# TODO: implement OGB loader

