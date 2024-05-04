# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.

import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
from pycocotools.coco import COCO
from deprecated import deprecated

from .utils import close_contour, binary_mask_to_polygon, singleMask2rle
from ...config import DISASTER_TYPE


@deprecated(version='0.0.1', reason="This class is deprecated, use NewFarmAnnotation instead")
class FarmAnnotation:
    """
    This class is used for get annotations on a farm tif, and save these annotations as like-coco
    format json file.

    :param block_size_x: block_size_x
    :param block_size_y: block_size_y
    :param repetition_rate: repetition_rate
    :param save_dir: img data save_dir
    :return: None
    """

    def __init__(self,
                 block_size_x: int,
                 block_size_y: int,
                 repetition_rate: float,
                 img_save_dir: str = None,
                 anno_save_dir: str = None,
                 save_dir: str = None):
        if repetition_rate < 0 or repetition_rate > 1:
            raise Exception(f"repetition_rate: {repetition_rate} not in [0,1]")

        self.img_id = 1
        self.annotaion_id = 1
        self.reg_id = 1
        self.CATEGORY_DICT = {}
        self.REG_DICT = {}

        self.block_size_x = block_size_x
        self.block_size_y = block_size_y
        self.repetition_rate = repetition_rate
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"{save_dir} not exists, create it")

        self.coco = self._init()

    def _init(self):
        """
        init coco dataset structure
        """
        coco = COCO()
        # 初始化coco数据集结构
        info = {
            "description": "Yield Prediction Dataset",
            "url": "",
            "version": "v1.0",
            "year": 2024,
            "contributor": "Shenzhou Liu",
            "date_created": "2024-01-09"
        }
        coco.dataset = {
            "info": info,
            "images": [],
            "annotations": [],
            "categories": [],
            "regressions": [],
            "reg_categories": [],
        }
        return coco

    def export_label(self, coco_fp):
        """
        export coco label to json file
        Args:
            coco_fp:

        Returns:

        """
        # save coco
        # Use COCO's method to encode the dataset as a JSON string
        coco_json_data = self.coco.dataset
        coco_json_data_str = json.dumps(coco_json_data, indent=4)

        # Write the JSON string to a file
        with open(coco_fp, 'w') as json_file:
            json_file.write(coco_json_data_str)

    def add_one_farm_annos(self,
                           tif_im,
                           yield_im=None,
                           cluster_yield_im=None,
                           disaster_im=None,
                           disaster_dict=None,
                           crop_type_im=None,
                           crop_type_dict=None,
                           export_tif=True
                           ):
        """
        add one farm annotations to built-in dataset

        :param tif_im: RSImg  tif image
        :param yield_im: RSImg  yield image
        :param cluster_yield_im: RSImg  cluster yield image
        :param disaster_im: RSImg  disaster image
        :param disaster_dict: dict  disaster dict used for present raster value to semantic label
        :param crop_type_im: RSImg  crop type image
        :param crop_type_dict: dict  crop type dict used for present raster value to instance label
        :param export_tif: bool  if export tif to save_dir
        """
        anno_dict = {
            'data': tif_im,
            'label': {
                'disaster': {
                    'annotype': 'semantic',
                    'dict': disaster_dict,
                    'im': disaster_im,
                    'gen': False
                },
                'yield': {
                    'annotype': 'regression',
                    'dict': None,
                    'im': yield_im,
                    'gen': False
                },
                'cluster_yield': {
                    'annotype': 'regression',
                    'dict': None,
                    'im': cluster_yield_im,
                    'gen': False
                },
                'croptype': {
                    'annotype': 'instance',
                    'dict': crop_type_dict,
                    'im': crop_type_im,
                    'gen': False
                },
            }
        }

        tif = anno_dict['data']
        annos = anno_dict['label']

        # ==================================check==============================
        for anno_content, anno_config in annos.items():
            annotype = anno_config['annotype']
            _dict = anno_config['dict']
            im = anno_config['im']
            if annotype != 'regression':
                if im is not None and _dict is not None:
                    anno_dict['label'][anno_content]['gen'] = True
                elif im is None and _dict is None:
                    pass
                else:
                    raise Exception(f"{anno_content}: im and dict must be both None or not None")

            else:
                if im is not None:
                    anno_dict['label'][anno_content]['gen'] = True

            # check img size if the same
            if im is not None:
                if im.WIDTH != tif.WIDTH or im.HEIGHT != tif.HEIGHT:
                    raise Exception(f"im: {im.name} and tif size not the same")
                else:
                    pass

            print(f"{anno_content}:")
            for key, value in anno_config.items():
                print(f"{key}: {value}")
            print("====================================")

        # =============================================================================

        # ======================== read categories from dicts =========================
        for anno_content, anno_config in annos.items():
            annotype = anno_config['annotype']
            _dict = anno_config['dict']
            ifgen = anno_config['gen']
            if ifgen:
                if annotype in ['semantic', 'instance']:
                    dict_value_unique = list(set(_dict.values()))
                    for value in dict_value_unique:
                        self.coco.dataset['categories'].append({
                            "id": len(self.coco.dataset['categories']) + 1,
                            "name": value,
                            "supercategory": "crop"
                        })
                        self.CATEGORY_DICT[value] = len(self.coco.dataset['categories'])
                elif annotype == 'regression':
                    self.coco.dataset['reg_categories'].append({
                        "id": len(self.coco.dataset['reg_categories']) + 1,
                        "name": anno_content,
                    })
                    self.REG_DICT[anno_content] = len(self.coco.dataset['reg_categories'])
                else:
                    pass

        # =============================================================================

        self.gen_anno_in_one_tif(
            anno_dict=anno_dict,
            block_size_x=self.block_size_x,
            block_size_y=self.block_size_y,
            repetition_rate=self.repetition_rate,
            img_save_dir=self.save_dir,
            export_tif=export_tif
        )

    def nan_array_to_json(self, anno_array):
        """
        Convert a 2D array with NaNs to JSON data
        """
        # Find the indices and values of non-nan elements
        non_nan_mask = ~np.isnan(anno_array)
        row_idx, col_idx = np.where(non_nan_mask)
        values = anno_array[row_idx, col_idx]

        # Optionally, quantize the non-nan values to reduce the number of unique data points
        quantization_step = 0.001  # or any other step that makes sense for your data
        quantized_values = np.around(values / quantization_step).astype(int)
        unique_values, inverse_indices = np.unique(quantized_values, return_inverse=True)

        # Create a COO sparse matrix with the quantized values
        sparse_matrix = coo_matrix((quantized_values, (row_idx, col_idx)), shape=anno_array.shape)

        # Create the JSON data object including the unique values and their run-length-encoded counts
        json_data = {
            "num_rows": sparse_matrix.shape[0],
            "num_cols": sparse_matrix.shape[1],
            "rows": row_idx.tolist(),
            "cols": col_idx.tolist(),
            "unique_values": unique_values.tolist(),
            "value_indices": inverse_indices.tolist(),
            "quantization_step": quantization_step
        }
        return json_data

    def json_to_array(self, json_data, nodata_value=np.nan):
        """
        Convert JSON data to a 2D array
        """
        num_rows = json_data['num_rows']
        num_cols = json_data['num_cols']
        rows = json_data['rows']
        cols = json_data['cols']
        unique_values = np.array(json_data['unique_values'])
        value_indices = json_data['value_indices']
        quantization_step = json_data['quantization_step']

        # Dequantize the values
        values = unique_values[value_indices] * quantization_step

        # Create a dense array with NaNs where data is invalid
        dense_array = np.full((num_rows, num_cols), nodata_value, dtype=np.float32)

        # Assign quantized values to their corresponding positions
        dense_array[rows, cols] = values

        # No need to substitute 0 for NaN, as we initialized the array with NaNs
        return dense_array

    def gen_anno_in_one_tif(self, anno_dict, block_size_x: int, block_size_y: int, repetition_rate: float,
                            img_save_dir, export_tif=True):
        """
        generate annotations in one tif

        :param anno_dict: dict  anno_dict
        :param block_size_x: int  block_size_x
        :param block_size_y: int  block_size_y
        :param repetition_rate: float  repetition_rate
        :param img_save_dir: str  img_save_dir
        :param export_tif: bool  if export tif to save_dir
        """
        tif = anno_dict['data']
        label = anno_dict['label']

        WIDTH = tif.WIDTH
        HEIGHT = tif.HEIGHT

        # 计算重复率对应的像素数
        step_size_x = int(block_size_x * (1 - repetition_rate))
        step_size_y = int(block_size_y * (1 - repetition_rate))

        row_col_list = []
        for row in range(0, HEIGHT - block_size_y + 1, step_size_y):
            for col in range(0, WIDTH - block_size_x + 1, step_size_x):
                row_col_list.append((row, col))

        for row, col in tqdm(row_col_list, total=len(row_col_list)):
            tif_block = tif.crop(col, row, col + block_size_x, row + block_size_y)
            GEN_SENTINEL2 = False
            tif_prefix = tif.name
            tif_save_name = f"{tif_prefix}_{row}_{col}.tif"
            tif_save_path = os.path.join(img_save_dir, tif_save_name)
            image_info = {
                "id": self.img_id,
                "file_name": tif_save_name,
                "height": tif_block.HEIGHT,
                "width": tif_block.WIDTH
            }
            for anno_content, anno_config in label.items():
                if not anno_config['gen']:
                    continue
                else:
                    annotype = anno_config['annotype']
                    _dict = anno_config['dict']
                    anno_im = anno_config['im']
                    anno_block = anno_im.crop(col, row, col + block_size_x, row + block_size_y)
                    anno_valid_mask = anno_block.valid_mask

                    if anno_valid_mask.sum() == 0:
                        continue
                    else:
                        GEN_SENTINEL2 = True
                        if annotype == 'regression':
                            anno_array = anno_block.ds.ReadAsArray()
                            anno_nodata_value = anno_block.nodatavalue
                            anno_array = np.where(anno_array == anno_nodata_value, np.nan, anno_array)
                            json_data = self.nan_array_to_json(anno_array)

                            # Add the JSON data object to the COCO dataset
                            self.coco.dataset['regressions'].append({
                                'id': self.reg_id,
                                "image_id": self.img_id,
                                "reg_id": self.REG_DICT[anno_content],
                                "array": json_data
                            })
                            self.reg_id += 1
                            # print(f"annotype: {annotype}, anno_content: {anno_content}, reg_id: {self.REG_DICT[anno_content]}")
                        elif annotype == 'instance':
                            instance_dict = anno_config['dict']
                            # find value set except nodata value
                            nodatavalue = anno_block.nodatavalue
                            value_set = list(set(anno_block.ds.ReadAsArray().flatten().tolist()))
                            value_set.remove(nodatavalue)
                            # print(f"value_set: {value_set}")
                            for value in value_set:
                                instance_name = instance_dict[value]
                                value_pos = np.where(anno_block.ds.ReadAsArray() == value)
                                min_x = int(np.min(value_pos[1]))
                                min_y = int(np.min(value_pos[0]))
                                max_x = int(np.max(value_pos[1]))
                                max_y = int(np.max(value_pos[0]))
                                bbox = [min_x, min_y, max_x, max_y]
                                # tran to coco bbox
                                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                                value_pos = (anno_block.ds.ReadAsArray() == value).astype(int)
                                polygon = singleMask2rle(value_pos)
                                attributes = instance_name
                                attributes_id = self.CATEGORY_DICT[attributes]

                                annotation_info = {
                                    "id": self.annotaion_id,
                                    "image_id": self.img_id,
                                    "category_id": attributes_id,
                                    "iscrowd": 1,
                                    "bbox": bbox,
                                    "segmentation": polygon,
                                }
                                self.annotaion_id += 1
                                self.coco.dataset['annotations'].append(annotation_info)
                        elif annotype == 'semantic':
                            semantic_dict = anno_config['dict']
                            # find value set except nodata value
                            nodatavalue = anno_block.nodatavalue
                            value_set = list(set(anno_block.ds.ReadAsArray().flatten().tolist()))
                            value_set.remove(nodatavalue)
                            for value in value_set:
                                semantic_name = semantic_dict[value]
                                value_pos = (anno_block.ds.ReadAsArray() == value).astype(int)
                                polygon = singleMask2rle(value_pos)
                                attributes = semantic_name
                                attributes_id = self.CATEGORY_DICT[attributes]
                                annotation_info = {
                                    "id": self.annotaion_id,
                                    "image_id": self.img_id,
                                    "category_id": attributes_id,
                                    "iscrowd": 1,
                                    "segmentation": polygon,
                                }
                                self.coco.dataset['annotations'].append(annotation_info)
                                self.annotaion_id += 1
                        else:
                            raise Exception(f"anno_tool_type: {annotype} not support")
            if GEN_SENTINEL2:
                if export_tif:
                    tif_block.to_tif(tif_save_path)
                self.coco.dataset['images'].append(image_info)
                self.img_id += 1

    @staticmethod
    def array2txt(array):
        txt = array.tolist()
        return txt

    @staticmethod
    def array2file(array):
        # np.save, np.savez, np.savez_compressed
        np.save(array)


class NewFarmAnnotation:
    """
    This class is used for get annotations on a farm tif, and save these annotations as like-coco
    format json file.

    :param block_size_x: block_size_x
    :param block_size_y: block_size_y
    :param repetition_rate: repetition_rate
    :param save_dir: img data save_dir
    :return: None
    """

    def __init__(self,
                 block_size_x: int,
                 block_size_y: int,
                 repetition_rate: float,
                 img_save_dir: str = None,
                 anno_save_dir: str = None,
                 # save_dir: str = None
                 ):
        if repetition_rate < 0 or repetition_rate > 1:
            raise Exception(f"repetition_rate: {repetition_rate} not in [0,1]")

        self.img_id = 1
        self.annotaion_id = 1
        self.reg_id = 1
        self.CATEGORY_DICT = {}
        self.REG_DICT = {}

        self.block_size_x = block_size_x
        self.block_size_y = block_size_y
        self.repetition_rate = repetition_rate
        self.img_save_dir = img_save_dir
        self.anno_save_dir = anno_save_dir

        for _dir in [self.img_save_dir, self.anno_save_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
                print(f"{_dir} not exists, create it")

        self.coco = self._init()

    def _init(self):
        """
        init coco dataset structure
        """
        coco = COCO()
        # 初始化coco数据集结构
        info = {
            "description": "Yield Prediction Dataset",
            "url": "",
            "version": "v1.0",
            "year": 2024,
            "contributor": "Shenzhou Liu",
            "date_created": "2024-01-09"
        }
        coco.dataset = {
            "info": info,
            "images": [],
            "annotations": [],
            "categories": [],
            "regressions": [],
            "reg_categories": [],
        }
        return coco

    def export_label(self, coco_fp):
        """
        export coco label to json file
        Args:
            coco_fp:

        Returns:

        """
        # save coco
        # Use COCO's method to encode the dataset as a JSON string
        coco_json_data = self.coco.dataset
        coco_json_data_str = json.dumps(coco_json_data, indent=4)

        # Write the JSON string to a file
        with open(coco_fp, 'w') as json_file:
            json_file.write(coco_json_data_str)

    def add_one_farm_annos(self,
                           anno_dict,
                           export_tif=True
                           ):
        """
        anno_dict stores the tif and annotations, e.g.:
        anno_dict = {
                    'data': sentinel2_im,
                    'label': {
                        'yield': {
                            'annotype': 'regression',
                            'dict': None,
                            'im': yield_im,
                        },
                        'croptype': {
                            'annotype': 'instance',
                            'dict': farm_crop_dict,
                            'im': croptype_img,
                        },
                    }
                }
        The data is the tif image, and the label is the annotations
        Args:
            anno_dict:
            export_tif:

        Returns:

        """
        tif = anno_dict['data']
        annos = anno_dict['label']
        # ==================================check==============================
        for anno_content, anno_config in annos.items():
            annotype = anno_config['annotype']
            _dict = anno_config['dict']
            im = anno_config['im']
            # check img size if the same
            if im is not None:
                if im.width != tif.width or im.height != tif.height:
                    raise Exception(f"im: {im.name} and tif size not the same")
            if annotype != 'regression':
                if im is not None and _dict is not None:
                    pass
                elif im is None and _dict is None:
                    pass
                else:
                    raise Exception(f"{anno_content}: im and dict must be both None or not None")
        # =============================================================================

        # ======================== read categories from dicts =========================
        for anno_content, anno_config in annos.items():
            annotype = anno_config['annotype']
            _dict = anno_config['dict']
            if annotype in ['semantic', 'instance']:
                dict_value_unique = list(set(_dict.values()))
                for value in dict_value_unique:
                    if value not in self.CATEGORY_DICT:
                        value_id = len(self.coco.dataset['categories']) + 1
                        self.CATEGORY_DICT[value] = value_id
                        self.coco.dataset['categories'].append({
                            "id": value_id,
                            "name": value,
                            "supercategory": "crop"
                        })
                    else:
                        value_id = self.CATEGORY_DICT[value]
            elif annotype == 'regression':
                if anno_content not in self.REG_DICT:
                    reg_id = len(self.coco.dataset['reg_categories']) + 1
                    self.REG_DICT[anno_content] = reg_id
                    self.coco.dataset['reg_categories'].append({
                        "id": reg_id,
                        "name": anno_content,
                    })
                else:
                    reg_id = self.REG_DICT[anno_content]
            else:
                raise Exception(f"anno_tool_type: {annotype} not support")

        # =============================================================================
        self.gen_anno_in_one_tif(
            anno_dict=anno_dict,
            block_size_x=self.block_size_x,
            block_size_y=self.block_size_y,
            repetition_rate=self.repetition_rate,
            img_save_dir=self.img_save_dir,
            anno_save_dir=self.anno_save_dir,
            export_tif=export_tif
        )

    def gen_anno_in_one_tif(self, anno_dict,
                            block_size_x: int,
                            block_size_y: int,
                            repetition_rate: float,
                            img_save_dir,
                            anno_save_dir,
                            export_tif=True):
        """
        generate annotations in one tif
        Args:
            anno_dict:
            block_size_x:
            block_size_y:
            repetition_rate:
            img_save_dir:
            anno_save_dir:
            export_tif:

        Returns:

        """
        assert os.path.exists(img_save_dir), f"{img_save_dir} not exists"
        assert os.path.exists(anno_save_dir), f"{anno_save_dir} not exists"

        tif = anno_dict['data']
        label = anno_dict['label']

        """
        label = 'disaster': {
                    'annotype': 'semantic',
                    'dict': disaster_dict,
                    'im': disaster_im,
                },
                'yield': {
                    'annotype': 'regression',
                    'dict': None,
                    'im': yield_im,
                },
                'croptype': {
                    'annotype': 'instance',
                    'dict': crop_type_dict,
                    'im': crop_type_im,
                },
        """

        WIDTH = tif.width
        HEIGHT = tif.height

        # 计算重复率对应的像素数
        step_size_x = int(block_size_x * (1 - repetition_rate))
        step_size_y = int(block_size_y * (1 - repetition_rate))

        row_col_list = []
        for row in range(0, HEIGHT - block_size_y + 1, step_size_y):
            for col in range(0, WIDTH - block_size_x + 1, step_size_x):
                row_col_list.append((row, col))

        pbar = tqdm(row_col_list,
                    total=len(row_col_list),
                    desc=f"Processing {tif.name} block annotations",  # 进度条描述
                    ascii=False,
                    )

        for row, col in pbar:
            pbar.set_postfix_str(f"Processing Row: {row}, Col: {col}")
            GEN_SENTINEL2 = False
            # check if need gen data img
            for anno_content, anno_config in label.items():
                """
                'yield': {
                            'annotype': 'regression',
                            'dict': None,
                            'im': yield_im,
                        },  
                """
                annotype = anno_config['annotype']
                _dict = anno_config['dict']
                anno_im = anno_config['im']
                anno_block = anno_im.crop(col, row, col + block_size_x, row + block_size_y)
                anno_valid_mask = anno_block.valid_mask
                # if anno img is null and GEN_SENTINEL2 is False, gen sentinel2 img
                anno_is_null = anno_valid_mask.sum() == 0
                if not anno_is_null and not GEN_SENTINEL2:
                    GEN_SENTINEL2 = True
                    tif_block = tif.crop(col, row, col + block_size_x, row + block_size_y)
                    tif_prefix = tif.name
                    tif_save_name = f"{tif_prefix}_{row}_{col}.tif"
                    tif_save_path = os.path.join(img_save_dir, tif_save_name)
                    image_info = {
                        "id": self.img_id,
                        "file_name": tif_save_name,
                        "height": tif_block.height,
                        "width": tif_block.width
                    }
                    if export_tif:
                        tif_block.to_tif(tif_save_path)
                    self.coco.dataset['images'].append(image_info)

                elif anno_is_null:
                    continue
                else:
                    pass

                if annotype == 'regression':
                    anno_array = anno_block.array.squeeze(0)
                    npy_file_name = f"{tif_save_name}_{anno_content}.npy"
                    npy_file_path = os.path.join(anno_save_dir, npy_file_name)
                    np.save(npy_file_path, anno_array)
                    self.coco.dataset['regressions'].append({
                        'id': self.reg_id,
                        "image_id": self.img_id,
                        "reg_id": self.REG_DICT[anno_content],
                        "array": npy_file_name
                    })
                    self.reg_id += 1
                elif annotype == 'instance':
                    instance_dict = anno_config['dict']
                    # find value set except nodata value
                    nodatavalue = anno_block.nodatavalue
                    # 过滤掉 NaN 值
                    filtered_arr = anno_block.array[~np.isnan(anno_block.array)]
                    # 将过滤后的数组转换为集合
                    value_set = set(filtered_arr)
                    try:
                        value_set.remove(nodatavalue)
                    except:
                        if not np.isnan(nodatavalue):
                            print(f"nodatavalue: {nodatavalue} not in value_set:{value_set}")
                    # print(f"value_set: {value_set}")
                    for value in value_set:
                        instance_name = instance_dict[value]
                        value_pos = np.where(anno_block.array == value)
                        min_x = int(np.min(value_pos[1]))
                        min_y = int(np.min(value_pos[0]))
                        max_x = int(np.max(value_pos[1]))
                        max_y = int(np.max(value_pos[0]))
                        bbox = [min_x, min_y, max_x, max_y]
                        # tran to coco bbox
                        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        value_pos = (anno_block.array == value).astype(int).squeeze(0)
                        polygon = singleMask2rle(value_pos)
                        attributes = instance_name
                        attributes_id = self.CATEGORY_DICT[attributes]

                        annotation_info = {
                            "id": self.annotaion_id,
                            "image_id": self.img_id,
                            "category_id": attributes_id,
                            "iscrowd": 1,
                            "bbox": bbox,
                            "segmentation": polygon,
                        }
                        self.annotaion_id += 1
                        self.coco.dataset['annotations'].append(annotation_info)

                elif annotype == 'semantic':
                    semantic_dict = anno_config['dict']
                    # find value set except nodata value
                    nodatavalue = anno_block.nodatavalue
                    value_set = list(set(anno_block.array.flatten().tolist()))
                    value_set.remove(nodatavalue)
                    for value in value_set:
                        semantic_name = semantic_dict[value]
                        value_pos = (anno_block.array == value).astype(int)
                        polygon = singleMask2rle(value_pos)
                        attributes = semantic_name
                        attributes_id = self.CATEGORY_DICT[attributes]
                        annotation_info = {
                            "id": self.annotaion_id,
                            "image_id": self.img_id,
                            "category_id": attributes_id,
                            "iscrowd": 1,
                            "segmentation": polygon,
                        }
                        self.coco.dataset['annotations'].append(annotation_info)
                        self.annotaion_id += 1

                else:
                    raise Exception(f"anno_tool_type: {annotype} not support")

            if GEN_SENTINEL2:
                self.img_id += 1
