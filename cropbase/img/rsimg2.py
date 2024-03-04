# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import os
import uuid
import datetime
import rasterio
import pandas as pd
import numpy as np
import rasterio
from collections import *

def _is_value_in_dtype_range(value, dtype):
    if np.issubdtype(dtype, np.integer):
        # 整数类型
        info = np.iinfo(dtype)
        return info.min <= value <= info.max
    elif np.issubdtype(dtype, np.floating):
        # 浮点数类型
        info = np.finfo(dtype)
        return info.min <= value <= info.max
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _get_array_shape(arr: np.ndarray):
    """

    """
    if arr.ndim == 2:
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:
        height, width, channels = arr.shape
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return height, width, channels

class RsImg:

    def __init__(self, ds, *args, **kwargs) -> None:
        self.ds = ds
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def read_from_tif(cls, tif_path, *args, **kwargs):
        with rasterio.open(tif_path) as ds:
            return cls(ds=ds, *args, **kwargs)

    @classmethod
    def read_from_array(cls, array: np.ndarray, nodatavalue, projection, geoTransform: tuple, *args, **kwargs):
        # check
        assert len(array.shape) == 2 or len(array.shape) == 3, "array shape must be 2 or 3, but got {}".format(
            array.shape)
        # nodatavalue can't over the range of dtype
        array_dtype = array.dtype
        assert _is_value_in_dtype_range(nodatavalue, array_dtype), "nodatavalue can't over the range of dtype"


        height, width, channels = _get_array_shape(array)
        # 创建数据集配置字典
        dataset_meta = {
            'count': channels,
            'dtype': array_dtype,
            'width': width,
            'height': height,
            'nodata': nodatavalue,
            'transform': geoTransform,
            'crs': projection
        }

        memfile = rasterio.io.MemoryFile()
        dataset = memfile.open(**dataset_meta)
        dataset.write(array, indexes=[i + 1 for i in range(channels)])
        memfile.close()
        return cls(ds=dataset, *args, **kwargs)

    def to_tif(self, tif_path):
        with rasterio.open(tif_path, 'w', **self.ds.meta) as dst:
            dst.write(self.ds.read())
    def to_array(self):
        return self.ds.read()

    def __del__(self):
        self.ds.close()
        del self.ds
