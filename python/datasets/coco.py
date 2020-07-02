from pycocotools.coco import COCO
import numpy as np

from .dataset import Dataset


class Coco(Dataset):
    def __init__(self, dataset_path, ann_info):
        super().__init__(dataset_path, ann_info)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            bbox = []
            label = []
            for k in ann_info:
                bbox.append(k['bbox'])
                label.append(k['category_id'])
            ann = {'bboxes': np.array(bbox, dtype=np.float),
                   'labels': np.array(label, dtype=np.int)}
            info['filename'] = info['file_name']
            info['ann'] = ann
            data_infos.append(info)
        self.CLASSES = [c['name'] for c in self.coco.dataset['categories']]
        return data_infos
