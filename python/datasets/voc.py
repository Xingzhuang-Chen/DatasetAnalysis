import os.path as osp
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

from .dataset import Dataset
from utils import list_from_file


class Voc(Dataset):
    def  __init__(self,
                  dataset_path,
                  ann_file):
        super().__init__(dataset_path, ann_file)

    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            ann = self.get_ann_info(root)
            ann['bboxes'][:, 2] = ann['bboxes'][:, 2] - ann['bboxes'][:, 0]
            ann['bboxes'][:, 3] = ann['bboxes'][:, 3] - ann['bboxes'][:, 1]
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height, ann = ann))

        return data_infos

    def get_ann_info(self, root):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is None:
                continue
            name = name.text
            if name not in self.CLASSES:
                self.CLASSES.append(name)
                self.cat2label[name] = len(self.cat2label)
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
