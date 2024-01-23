# 数据集标注说明

使用coco数据集格式对遥感影像进行标注，生成coco数据集格式的标注文件，方便后续训练使用。


coco的标注文件分为5个部分：
```json
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```
## info

info部分记录了整个数据集的名称、URL、版本、时间等基本信息。不包含具体的数据信息

## licenses
licenses部分记录了数据集中的数据遵循了哪些license。不包含具体的数据信息。

## images
images部分记录了数据集中的所有图片的信息，包括图片的id、文件名、高度、宽度、拍摄日期、拍摄设备等信息。不包含具体的数据信息。

```json
annotations['images'] = [
	{
		'license': 1,
		'file_name': '000000397133.jpg',
		'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
		'height': 427,
		'width': 640,
		'date_captured': '2013-11-14 17:02:52',
		'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
		'id': 397133
	},
	...
]

```

## annotations
annotations部分记录了标注信息，包括segmentation、bbox、面积（area）、
是否为群体（iscrowd）、image_id、类别（category_id）、instance id（id）。
注意，这里的’image_id’对应的是前面images部分的’id’，而这里的’id’指的是
每个instance的’id’。
每个dict记录的是一个instance的标注，而不是整幅图像。

```json
annotation['annotations'] = [
	{
		'segmentation': []
		'area': 702.1057
		'iscrowd': 0
		'image_id': 289343
		'bbox': [473.07, 395.93, 38.65, 28.67]
		'category_id': 18
		'id': 1768
	},
	{
		'segmentation':
		'area':
		'iscrowd':
		'image_id':
		'bbox':
		'category_id':
		'id':
	},
	...
]

```
`segmentation`
该部分标注的是分割任务所需的mask。
注意，一个segmentation中可能包含不止一个mask，原因是一个instance可能被遮挡，
从而对应多个mask（例如大象被树干挡住，从而变成了两部分）。
mask有两种标注方式。一种方式是多边形标注（polygon），其中每一个list都对应一个多边形，
另一种方式是run-length encoding（RLE），通常用于标注iscrowd为1的instance


`bbox`
bbox标注的是instance的包围框（bounding box），其格式是：[x, y, width, height]

`area`
mask覆盖的面积（像素总数），可用来筛选不同面积的instance。

`image_id`
对应于前面’images’部分的’id’，用来唯一标识一幅图像。

`id`
用来唯一标识一个instance。

`category_id`
对应于后面’categories’部分的’id’，用来唯一标识一个类别。


## categories

categories部分记录了数据集中所有类别的信息，包括类别的id、name、

```json
annotation['categories'] = [
	{
		'supercategory': 'vehicle',
		'id': 2,
		'name': 'bicycle'
	},
	{
		'supercategory': 'animal',
		'id': 22,
		'name': 'elephant'
	}
	...
]

```