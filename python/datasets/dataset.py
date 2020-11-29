from abc import ABCMeta, abstractmethod
from os import path as osp


class Dataset(metaclass=ABCMeta):

    CLASSES = []

    def  __init__(self,
                  dataset_path,
                  ann_file):
        self.img_prefix = dataset_path
        self.cat2label = dict()

        self.data_infos = self.load_annotations(osp.join(dataset_path, ann_file))

        self.curent_num = 0

    def __len__(self):
        return len(self.data_infos)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curent_num < len(self):
            self.curent_num += 1
            return self.data_infos[self.curent_num - 1]
        else:
            raise StopIteration

    # @abstractmethod
    def get_ann_info(self, idx):
        pass

    @abstractmethod
    def load_annotations(self, ann_file):
        pass

    def get_img_width(self):
        pass

    def get_img_heights(self):
        pass

    def get_img_area(self):
        pass

