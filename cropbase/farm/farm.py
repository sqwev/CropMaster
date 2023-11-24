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


# private
from ..img import RSImg, Sentinel2Tif
from ..shp import Farmshp


class FarmWithOneImg:
    img = None
    features = None
    def __init__(self, tif_path, farm_shp_path,year = None):
        print(f"*********Init Farm*********")
        self.tif_path = tif_path
        self.farm_shp_path = farm_shp_path

        self.img = Sentinel2Tif(path=self.tif_path, level="L2A")
        self.shp = Farmshp(farm_shp_path=self.farm_shp_path)
        self.year = year

        # 输入的shp必须没有multipolygon
        IFMULTIPOLYGON = self.shp.IFMULTIPOLYGON
        if IFMULTIPOLYGON:
            self.farm_shp_path = self.shp.split_shp()
            self.shp = Farmshp(farm_shp_path=self.farm_shp_path)
            IFMULTIPOLYGON = self.shp.IFMULTIPOLYGON

        # tif内容
        self.ds = gdal.Open(self.tif_path)
        self.projection = self.ds.GetProjection()
        self.geoTransform = self.ds.GetGeoTransform()
        # shp内容
        self.features = fiona.open(self.farm_shp_path)

        # 检查坐标系是否一致
        tif_espg = self.img.get_espg()
        shp_espg = self.shp.get_espg()
        if tif_espg != shp_espg:
            raise Exception(f"tif_espg: {tif_espg}, shp_espg: {shp_espg}, they are not equal")
        else:
            print(f"tif_espg: {tif_espg}, shp_espg: {shp_espg}, they are equal")
        print(f"tif path: {self.tif_path}\nshp path: {self.farm_shp_path}")
        print(f"*********Farm init finished*********")

    def process_by_field(self, features, process_fun, nodatavalue, force_data_type=None):
        """
        features是fiona.open读取的shp文件，是一个可迭代对象
        process_fun是一个函数，输入的是array和properties，输出是array,输入和输出的array中的nodatavalue都是np.nan
        nodatavalue是输出的tif中的nodatavalue
        force_data_type是输出的tif中的数据类型，如果为None，则输出的tif中的数据类型一般为float32

        这个程序用于处理一个tif中的所有田块，每个田块的处理结果是一个array，对每个田块的array进行处理，
        返回一个与田块大小相同的array，将所有田块的array填充到这个array中，最后返回一个RSImg对象
        """
        with rasterio.open(self.tif_path) as ref_src:
            field_meta = ref_src.meta.copy()
            ref_width, ref_height = ref_src.width, ref_src.height


            # 先使用一个field测试返回的数组的维度，长宽是否正确
            testfield = features[0]
            ploygon = testfield['geometry']
            properties = testfield["properties"]
            testfield_image, testfield_transform = rasterio.mask.mask(ref_src,[ploygon],crop=True)
            print(f"testfield_shape:{testfield_image.shape}")
            testfield_dim = len(testfield_image.shape)
            if testfield_dim == 2:
                testfield_height, testfield_width  = testfield_image.shape
                # print(f"testfield_shape:{testfield_image.shape}")
            elif testfield_dim == 3:
                testfield_bands, testfield_height, testfield_width = testfield_image.shape
                # print(f"testfield_bands:{testfield_bands},testfield_width:{testfield_width},testfield_height:{testfield_height}")
            output = process_fun(field_image=testfield_image, properties=properties)
            print(f"output shape:{output.shape}")
            # 检查维度
            output_dim = len(output.shape)
            if output_dim == 2:
                output_height, output_width  = output.shape
            elif output_dim == 3:
                output_bands, output_height, output_width = output.shape
            else:
                raise Exception(f"output dim: {output_dim} not supported")
            # 检查长宽是否正确
            if output_height != testfield_height or output_width != testfield_width:
                raise Exception(f"output height: {output_height}, output width: {output_width} not equal to testfield height: {testfield_height}, testfield width: {testfield_width}")
            else:
                print(f"output height: {output_height}, output width: {output_width} equal to testfield height: {testfield_height}, testfield width: {testfield_width}")
            
            
            
            # 开始处理

            if output_dim == 2:
                field_template_array = np.zeros((self.img.HEIGHT, self.img.WIDTH))
                # 全部填充为nan
                field_template_array[:] = np.nan

            elif output_dim == 3:
                field_template_array = np.zeros((output_bands, self.img.HEIGHT, self.img.WIDTH))
                # 全部填充为nan
                field_template_array[:] = np.nan
            
            for i, field in tqdm(enumerate(features)):
                ploygon = field['geometry']
                properties = field["properties"]
                field_image, field_transform = rasterio.mask.mask(ref_src,[ploygon],crop=True)
                field_dim = len(field_image.shape)
                if field_dim == 2:
                    field_height, field_width  = field_image.shape
                    # print(f"field_width:{field_width},field_height:{field_height}")
                elif field_dim == 3:
                    field_bands, field_height, field_width = field_image.shape
                    # print(f"field_bands:{field_bands},field_width:{field_width},field_height:{field_height}")
                # field_image中的nodatavalue替换为nan
                field_image = np.where(field_image == 0, np.nan, field_image)

                output = process_fun(field_image=field_image, properties=properties)
                # print(f"output shape:{output.shape}")
                field_meta.update({"height": field_height,
                                    "width": field_width,
                                    "transform": field_transform})
                x_res, y_res = field_meta['transform'].a, field_meta['transform'].e
                x_min, y_min = field_meta['transform'].c, field_meta['transform'].f

                x_max = x_min + x_res * field_width
                y_max = y_min + y_res * field_height
                # print(f"x_res:{x_res},y_res:{y_res}, x_min:{x_min}, y_min:{y_min}, x_max:{x_max}, y_max:{y_max}")
                
                # 计算在田块中的位置
                def calculate_index(value, transform, index):
                    return int(round((value - transform[index]) / transform[index+1]))
                # 计算 col_min 等值，并四舍五入到整数
                col_min = calculate_index(x_min, self.geoTransform, 0)
                row_max = calculate_index(y_max, self.geoTransform, 3)
                col_max = calculate_index(x_max, self.geoTransform, 0)
                row_min = calculate_index(y_min, self.geoTransform, 3)
                # print(f"col_min:{col_min},col_max:{col_max},row_min:{row_min},row_max:{row_max}")

                # 将田块内的值填充到filled_array中
                if output_dim == 2:
                    field_template_array[row_min:row_max,col_min:col_max] = \
                    np.where(np.isnan(field_template_array[row_min:row_max,col_min:col_max]) & np.isnan(output),
                    np.nan,
                    np.nan_to_num(field_template_array[row_min:row_max,col_min:col_max], nan=0) +
                    np.nan_to_num(output, nan=0))
                elif output_dim == 3:
                    field_template_array[:,row_min:row_max,col_min:col_max] = \
                    np.where(np.isnan(field_template_array[:,row_min:row_max,col_min:col_max]) & np.isnan(output),
                    np.nan,
                    np.nan_to_num(field_template_array[:,row_min:row_max,col_min:col_max], nan=0) +
                    np.nan_to_num(output, nan=0))


            if force_data_type is not None:
                # nan to nodatavalue
                field_template_array[np.isnan(field_template_array)] = nodatavalue
                field_template_array = field_template_array.astype(force_data_type)

            farm_res = RSImg(array=field_template_array, nodatavalue=nodatavalue, projection=self.projection, geoTransform=self.geoTransform)
            # farm_res.gen_tif(savepath=f"farm_{self.year}.tif")
            return farm_res
