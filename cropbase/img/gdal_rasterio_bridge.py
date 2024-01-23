# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.

from osgeo import gdal
import rasterio
from rasterio.io import MemoryFile



from .gdalmemfilemanager import GDALVirtualFileManager
def is_mem_dataset(dataset):
    """
    判断 GDAL 数据集是否由 MEM 创建
    """
    description = dataset.GetDescription()
    return description == ''

def tran_ds2tif_path(ds: gdal.Dataset):
    # 获取 GDAL 数据集的文件路径
    IF_MEM = is_mem_dataset(ds)
    if IF_MEM:
        # 将 GDAL 数据集写入虚拟文件
        vsi_mem_file = GDALVirtualFileManager.create_virtual_file(ds=ds)
        return vsi_mem_file
    else:
        return ds.GetDescription()