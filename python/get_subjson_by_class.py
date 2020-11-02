"""
将一个coco json文件按照存在的目标对象拆分为多个json文件。
得到的json文件保存在dataset_path/annotations/class/ 目录下
文件以 class.json命名
拆分后的每个文件中的image和annotations的id都独立重新编号

为了加快保存速度，不保存annotation信息
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt
import os
from os import path as osp
import re
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='analysis a dataset.')
    parser.add_argument('dataset_path', help='dataset path')
    parser.add_argument('ann_file', help='COCO: ann file. VOC: img list file')
    args = parser.parse_args()

    return args


def get_sub_class_json(dataset):
    img_id2idx = dict()
    for i, img in enumerate(dataset['images']):
        img_id2idx[img['id']] = i
    ann_id2idx = dict()
    for i, ann in enumerate(dataset['annotations']):
        ann_id2idx[ann['id']] = i

    cate = dataset['categories']
    images = dataset['images']
    annos = dataset['annotations']

    class_json = [{
        'categories': cate,
        'images': [],
        'annotations': []
        # 'annotations': annos
    } for _ in cate]
    img_exist_ids = [[] for _ in cate]

    for ann in tqdm(annos):
        anno = ann.copy()
        cate_idx = anno['category_id'] - 1
        img_id = anno['image_id']
        img_idx = img_id2idx[img_id]
        if img_id not in img_exist_ids[cate_idx]:
            img = images[img_idx]
            class_json[cate_idx]['images'].append(img)
            img_exist_ids[cate_idx].append(img_id)

    return class_json


if __name__=='__main__':
    arg = parse_args()

    # load json
    f = open(osp.join(arg.dataset_path, arg.ann_file), 'r')
    dataset = json.load(f)
    f.close()

    # get sub class json
    class_json = get_sub_class_json(dataset)
    ann_path = osp.join(arg.dataset_path, 'annotations')
    class_json_path = osp.join(ann_path, 'classes')
    if not osp.exists(class_json_path):
        os.makedirs(class_json_path)
    count = 0
    for i, js in enumerate(class_json):
        class_name = dataset['categories'][i]['name']
        class_name = re.sub('[^\w]', '_', class_name)
        filename = '{:06d}_'.format(len(js['images'])) + class_name+'.json'
        with open(osp.join(class_json_path, filename), 'w') as fw:
            fw.writelines(json.dumps(js, ensure_ascii=False, indent=2))
        count += len(js['images'])
    print(count)
