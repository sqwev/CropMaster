import os
import warnings
import numpy as np
import pandas as pd
import fiona
import rtree
import json
import time
import rasterio
from rasterio.mask import raster_geometry_mask
import pandas as pd
import numpy as np
from osgeo import gdal, gdal_array, ogr, gdal, osr
from tqdm import tqdm
from shapely.geometry import Point, Polygon, shape, mapping

class FieldImg:
    def __init__(self, array, nodatavalue, projection=None, geoTransform=None) -> None:
        print(f"*********Init FieldImg*********")
        self.array = array
        self.nodatavalue = nodatavalue
        self.dim = len(array.shape)
        if self.dim != 2 and self.dim != 3:
            raise Exception(f"Field array dimemsion: {self.dim} not supported")

        if self.dim == 2:
            self.BANDS = 1
            self.HEIGHT = array.shape[0]
            self.WIDTH = array.shape[1]
        else:
            self.BANDS = array.shape[0]
            self.HEIGHT = array.shape[1]
            self.WIDTH = array.shape[2]

        # 不是nodatavalue的地方是有效值，生成一个二维数组，1为有效值，0为无效值
        if nodatavalue is None:
            self.valid_mask = np.ones((self.HEIGHT, self.WIDTH))
        elif np.isnan(nodatavalue):
            if self.dim == 2:
                self.valid_mask = np.where(np.isnan(array), 0, 1)
            else:
                template = self.array[0]
                self.valid_mask = np.where(np.isnan(template), 0, 1)
        else:
            if self.dim == 2:
                self.valid_mask = np.where(array == nodatavalue, 0, 1)
            else:
                template = self.array[0]
                self.valid_mask = np.where(template == nodatavalue, 0, 1)

        self.projection = projection
        self.geoTransform = geoTransform

        if self.geoTransform is not None:
            self.x_min, self.x_res, self.rotation, self.y_min, self.rotation, self.y_res = self.geoTransform
            self.x_max = self.x_min + self.x_res * self.WIDTH
            self.y_max = self.y_min + self.y_res * self.HEIGHT

        print(f"Field shape: {self.array.shape}, valid_mask shape: {self.valid_mask.shape},\
nodatavalue: {self.nodatavalue}")

    # 设置属性
    def set_projection(self, projection):
        self.projection = projection

    def set_geoTransform(self, geoTransform):
        self.geoTransform = geoTransform
        x_min, pixel_width, rotation, y_min, rotation, pixel_height = geoTransform
        self.x_min = x_min
        self.y_min = y_min
        self.x_res = pixel_width
        self.y_res = pixel_height
        self.x_max = x_min + pixel_width * self.WIDTH
        self.y_max = y_min + pixel_height * self.HEIGHT
        print(f"x_min:{x_min}, y_min:{y_min}, x_res:{pixel_width}, y_res:{pixel_height}, x_max:{self.x_max}, y_max:{self.y_max}")


    # 打印属性
    def print_geoInfo(self):
        print(f"projection: {self.projection}")
        print(f"geoTransform: {self.geoTransform}")

    # 保存影像
    def gen_tif(self, savepath):
        # 必须有projection和geoTransform
        if self.projection is None or self.geoTransform is None:
            raise Exception(f"projection or geoTransform is None")

        # 如果有nan而nodatavalue不是nan，将nan转为nodatavalue
        if np.isnan(self.nodatavalue):
            self.array[np.isnan(self.array)] = self.nodatavalue

        print(f"\nGenerate_tif, savepath: {savepath}...")
        # 只支持二维数组和三维数组
        
        def gdal_array_type(np_datatype):
            np_datatype = str(np_datatype)

            dtype_to_gdal = {
                'uint8': gdal.GDT_Byte,
                'uint16': gdal.GDT_UInt16,
                'int16': gdal.GDT_Int16,
                'uint32': gdal.GDT_UInt32,
                'int32': gdal.GDT_Int32,
                'float32': gdal.GDT_Float32,
                'float64': gdal.GDT_Float64
            }
            supported_dtypes = list(dtype_to_gdal.keys())

            assert np_datatype in supported_dtypes, f"np_datatype:{np_datatype} not supported"
            return dtype_to_gdal[np_datatype]
        nptype = self.array.dtype
        gdaltype = gdal_array_type(nptype)
        
        print(f"WIDTH:{self.WIDTH}, HEIGHT:{self.HEIGHT}, array_dim:{self.dim},filled_array.shape:{self.array.shape}")
        print(f"nptype:{nptype}, gdaltype:")

        
        if self.dim == 2:
            bands_num = 1         
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(savepath, self.WIDTH, self.HEIGHT, bands_num, gdaltype)
            out_ds.SetProjection(self.projection)
            out_ds.SetGeoTransform(self.geoTransform)
            out_ds.GetRasterBand(1).WriteArray(self.array)
            out_ds.GetRasterBand(1).SetNoDataValue(self.nodatavalue)
            out_ds.FlushCache()
            out_ds = None

        if self.dim == 3:
            bands_num = self.array.shape[0]
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(savepath, self.WIDTH, self.HEIGHT, bands_num, gdaltype)
            out_ds.SetProjection(self.projection)
            out_ds.SetGeoTransform(self.geoTransform)
            for i in range(bands_num):
                out_ds.GetRasterBand(i+1).WriteArray(self.array[i])
                out_ds.GetRasterBand(i+1).SetNoDataValue(self.nodatavalue)
            
            out_ds.FlushCache()
            out_ds = None



