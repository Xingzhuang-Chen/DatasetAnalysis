"""
用于将多个coco ann文件合并成一个。
只合并images字段，ann字段和category字段由main json文件提供

by natsusou
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
    parser.add_argument('main_ann_file', help='COCO: ann file. VOC: img list file')
    parser.add_argument('out_ann_file', help='COCO: ann file. VOC: img list file')
    parser.add_argument(
        '--ann-file-list',
        type=str,
        nargs='+',
        help='2个参数为一组，分别为ann文件相对路径和需要拷贝的次数，可同时提供多组参数')
    args = parser.parse_args()

    return args


def load_json(path):
    print(f'load {path}')
    f = open(path, 'r')
    dataset = json.load(f)
    f.close()
    return dataset


def merge_json(main_set, sub_set, count, imgid2annids):
    # for _ in range(count):
    # main_set['images'].extend(sub_set['images']*int(count))
    max_img_id = max([img['id'] for img in main_set['images']])
    max_ann_id = max([ann['id'] for ann in main_set['annotations']])
    for im in sub_set['images']:
        for _ in range(int(count)):
            img = im.copy()
            img_id = img['id']
            ann_ids = imgid2annids[img_id]
            ann_idxs = [ann_id2idx[i] for i in ann_ids]
            anns = [main_set['annotations'][i].copy() for i in ann_idxs]
            max_img_id += 1
            img['id'] = max_img_id
            for ann in anns:
                max_ann_id += 1
                ann['id'] = max_ann_id
                ann['image_id'] = max_img_id

            main_set['images'].append(img)
            main_set['annotations'].extend(anns)

    return main_set


if __name__=='__main__':
    arg = parse_args()
    ann_file_list = arg.ann_file_list[0::2]
    ann_count = arg.ann_file_list[1::2]

    # load json
    main_set = load_json(osp.join(arg.dataset_path, arg.main_ann_file))

    imgid2annids = dict()
    ann_id2idx = dict()
    for i, ann in enumerate(main_set['annotations']):
        if ann['image_id'] in imgid2annids.keys():
            imgid2annids[ann['image_id']].append(ann['id'])
        else:
            imgid2annids[ann['image_id']] = [ann['id']]
        ann_id2idx[ann['id']] = i

    for file, count in zip(ann_file_list, ann_count):
        sub_set = load_json(osp.join(arg.dataset_path, file))
        main_set = merge_json(main_set, sub_set, count, imgid2annids)

    with open(osp.join(arg.dataset_path, arg.out_ann_file), 'w') as fw:
        fw.writelines(json.dumps(main_set, ensure_ascii=False, indent=2))
