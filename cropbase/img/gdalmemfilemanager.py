# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.


import uuid
from osgeo import gdal

class GDALVirtualFileManager:
    file_list = []

    @classmethod
    def create_virtual_file(cls, ds: gdal.Dataset):
        vsi_mem_file_path = f'/vsimem/{uuid.uuid4().hex}.tif'
        gdal.GetDriverByName('GTiff').CreateCopy(vsi_mem_file_path, ds)
        cls.file_list.append(vsi_mem_file_path)
        print(gdal.VSIStatL(vsi_mem_file_path))
        return vsi_mem_file_path

    @classmethod
    def clear(cls, file_name=None):
        if file_name is None:
            for file_name in GDALVirtualFileManager.file_list:
                gdal.Unlink(file_name)
            cls.file_list.clear()
        else:
            gdal.Unlink(file_name)
            cls.file_list.remove(file_name)