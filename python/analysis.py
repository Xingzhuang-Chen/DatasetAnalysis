import argparse
import numpy as np
from matplotlib import pyplot as plt
import os
from os import path as osp
import re

from datasets import Voc, Coco


# 图像resize后等效矩形边长
original_wh = 1024
# box大小划分范围
areaRng = np.array([[32 ** 2, 96 ** 2]], dtype=np.float)


def parse_args():
    parser = argparse.ArgumentParser(description='analysis a dataset.')
    parser.add_argument('dataset_path', help='dataset path')
    parser.add_argument(
        'dataset_type',
        choices=['VOC', 'COCO'],
        help='dataset type')
    parser.add_argument('ann_file', help='COCO: ann file. VOC: img list file')
    args = parser.parse_args()

    return args


def analysis(dataset):
    width = []
    height = []
    area = []
    class_list = np.ndarray((0, 6), dtype=np.float)
    box_label = np.ndarray((0,), dtype=np.int)
    for info in dataset:
        # img
        img_area = info['width'] * info['height']
        width.append(info['width'])
        height.append(info['height'])
        area.append(img_area)

        # class
        box_width = info['ann']['bboxes'][:, 2].reshape(-1, 1)
        box_height = info['ann']['bboxes'][:, 3].reshape(-1, 1)
        box_ratio = box_width/box_height
        box_area = box_width*box_height
        box_scale = box_area/img_area

        mask_area = info['ann']['area'].reshape(-1, 1)
        box_area_class = ((mask_area - areaRng)>1).sum(1, keepdims=True)

        class_list = np.vstack([class_list,np.hstack([box_width, box_height, box_ratio, box_area, box_scale, box_area_class])])
        box_label = np.hstack([box_label, info['ann']['labels']])

    info = {'width': width,
            'height': height,
            'area': area,
            'boxes': class_list,
            'label': box_label,
            'class': dataset.CLASSES}
    return info


def show_result(info, result_path, result_path_suffix):
    result_path = osp.join(result_path, 'analysis_result_'+result_path_suffix)
    if not osp.exists(result_path):
        os.makedirs(result_path)
    class_result_path = osp.join(result_path, 'class')
    if not osp.exists(class_result_path):
        os.makedirs(class_result_path)

    def show_img_info(info):
        plt.figure()
        plt.subplot(2,2,1)
        plt.hist(x=info['width'], bins=100, edgecolor='black',  alpha=0.6)
        plt.xlabel('width')
        plt.ylabel('number')
        plt.title('img width')

        plt.subplot(2,2,2)
        plt.hist(x=info['height'], bins=100, edgecolor='black',  alpha=0.6)
        plt.xlabel('height')
        plt.ylabel('number')
        plt.title('img height')

        plt.subplot(2,1,2)
        plt.hist(x=info['area'], bins=100, edgecolor='black',  alpha=0.6)
        plt.xlabel('area')
        plt.ylabel('number')
        plt.title('img area')

        plt.savefig(osp.join(result_path, 'image_size.jpg'))

    def show_class_info(info):
        boxes = info['boxes']
        label = info['label']
        le = len(boxes)
        for i in range(len(info['class'])):
            class_box = boxes[label==i+1, :]
            class_name = info['class'][i]
            f = plt.figure(class_name)
            plt.subplot(121)
            plt.hist(x=class_box[:, 2], bins=100, edgecolor='black', alpha=0.6)
            plt.title('box ratio')
            plt.subplot(122)
            plt.hist(x=class_box[:, 4], bins=100, edgecolor='black', alpha=0.6)
            plt.title('box area/img area')
            plt.savefig(osp.join(class_result_path, re.sub('[^\w]', '_', class_name)+'.jpg'))
            plt.close(f)
        plt.figure('All class')
        plt.subplot(121)
        # 拉伸96%
        ratio =boxes[:, 2].copy()
        ratio.sort()
        plt.hist(x=ratio[int(le*0.02):int(le*0.98)], bins=100, edgecolor='black', alpha=0.6)
        plt.title('box ratio')
        plt.subplot(122)
        plt.hist(x=boxes[:, 4], bins=100, edgecolor='black', alpha=0.6)
        plt.title('box area/img area')
        plt.savefig(osp.join(result_path, 'All_class.jpg'))

    def show_anchor_scale(info):
        box_scale = info['boxes'][:, 4]*original_wh**2
        box_wh = np.sqrt(box_scale).reshape(-1,1)
        anchor_scale = np.dot(box_wh, 1/np.array([[4, 8, 16, 32, 64]], dtype=np.float))

        fig_name = 'All stage anchor scale'
        plt.figure(fig_name)
        plt.hist(x=anchor_scale[:, 4], bins=100, alpha=0.6)
        plt.hist(x=anchor_scale[:, 0], bins=100, alpha=0.6)
        plt.hist(x=anchor_scale[:, 1], bins=100, alpha=0.6)
        plt.hist(x=anchor_scale[:, 2], bins=100, alpha=0.6)
        plt.hist(x=anchor_scale[:, 3], bins=100, alpha=0.6)
        # plt.hist(x=anchor_scale[:, 4], bins=100, alpha=0.6)
        plt.title(fig_name)
        plt.savefig(osp.join(result_path, fig_name+'.jpg'))

        print('original box scale(h,w) mean= %f'%np.mean(box_wh))
        print('stage 1 box scale(h,w) mean= %f'%np.mean(anchor_scale[:, 0]))
        print('stage 2 box scale(h,w) mean= %f'%np.mean(anchor_scale[:, 1]))
        print('stage 3 box scale(h,w) mean= %f'%np.mean(anchor_scale[:, 2]))
        print('stage 4 box scale(h,w) mean= %f'%np.mean(anchor_scale[:, 3]))
        print('stage 5 box scale(h,w) mean= %f'%np.mean(anchor_scale[:, 4]))

        pass

    def show_class_count(info):
        CLASSES = info['class']
        label = info['label'].tolist()
        count = []
        for c in range(len(CLASSES)):
            count.append(label.count(c+1))
        fig_name = 'Class count'
        plt.figure(fig_name)
        # 柱子总数
        N = len(CLASSES)
        # 包含每个柱子对应值的序列
        values = np.array(count)
        sort_idx = values.argsort()[::-1]
        values = values[sort_idx]
        # 包含每个柱子下标的序列
        index = [CLASSES[j] + f'\n{values[i]}' for i, j in enumerate(sort_idx)]
        # 柱子的宽度
        width = 0.45
        # 绘制柱状图, 每根柱子的颜色为紫罗兰色
        p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
        plt.ylabel('number of bbox', )
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.savefig(osp.join(result_path, fig_name+'.jpg'))
        print(sum(values))

    def show_area_class(info):
        area_class = info['boxes'][:, 5]
        count = [(area_class==c).sum() for c in range(3)]
        index = [f'small({count[0]})', f'medium({count[1]})', f'large({count[2]})']

        fig_name = 'area_class'
        plt.figure(fig_name)
        # 柱子的宽度
        width = 0.45
        # 绘制柱状图, 每根柱子的颜色为紫罗兰色
        p2 = plt.bar(index, count, width, label="num", color="#87CEFA")
        plt.savefig(osp.join(result_path, fig_name+'.jpg'))

    # plt.ion()
    show_class_count(info)
    show_img_info(info)
    show_class_info(info)
    show_anchor_scale(info)
    show_area_class(info)
    # plt.ioff()
    plt.show()


if __name__=='__main__':
    arg = parse_args()

    if arg.dataset_type == 'VOC':
        dataset = Voc(arg.dataset_path, arg.ann_file)
    elif arg.dataset_type == 'COCO':
        dataset = Coco(arg.dataset_path, arg.ann_file)
    else:
        dataset = None
        exit(0)

    info = analysis(dataset)
    _, filepath = osp.split(arg.ann_file)
    filepath, _ = osp.splitext(filepath)
    show_result(info, arg.dataset_path, filepath)
