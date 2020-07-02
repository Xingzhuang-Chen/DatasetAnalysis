# 目标检测数据集统计
用于统计分析目标检测数据集中目标的大小和比例

## 使用

```shell script
python analysis.py <dataset_path> <dataset_type> <annotations_file_relative_path>
# eg.COCO
python analysis.py /media/dl/Samsung/IJCAI2020/COCO_format COCO annotations/train_set_coco_v2.json
# eg.VOC
python analysis.py /media/dl/TOSHIBA/002.DataSet/SecurityX VOC ImageSets/Main/all.txt
```

## 结果
统计结果保存在路径 `result_path = <dataset_path>/analysis_result_<annotations_file_name>/`
结果包括文件
><result_path> 
>
>|- image_size.jpg  所有图像长、宽、面积分布
>
>|- All stage anchor scale.jpg  所有box在backbone各个stage特征图上的边长分布
>
>|- All_class.jpg   所有box的长宽比和面积分布
>
>|_ class/  各个类别box的长宽比和面积分布

在统计程序运行时会plot显示出前3个统计结果图像，便于放大查看细节。
`class/`中的图像，为了避免打开过多窗口，程序运行时不显示，只在结果中保存。
