# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import os
import fiona
import json
import uuid
import rasterio
from rasterio.mask import raster_geometry_mask
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon, shape, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon

# private
from ..img import RsImg
from ..field import Field
from ..config import DISASTER_TYPE








class Farm:
    """
    Farm is a class to store a farm's information, including fields, farm name, farm shp path, farm tif path, etc.

    - fields_df: gpd.GeoDataFrame, fields_df is a GeoDataFrame, each row is a field, and each field has a geometry.
    - fields: list, each element is a Field object
    - name: str, farm name
    - crs: crs, coordinate reference system


    :param file: shp or geojson
    :param convert_dict: convert cols name in shp to standard cols name
    :param name: str Farm name
    :param gis_index_type: arcgis or qgis  This is used for init the index of field index,
    for Arcgis start from 0, and Qgis start from 1.

    """

    def __init__(self, *args, **kwargs):  # shp or geojson
        self.name = kwargs.get("name", str(uuid.uuid1()))
        if not isinstance(self.name, str):
            raise Exception(f"name: {self.name} must be str")
        file = kwargs.get("file", None)
        if file is None:
            raise Exception(f"file: {file} is None, must be shp or geojson")
        self.fields_df = gpd.read_file(file)
        self.crs = self.fields_df.crs

        # convert part
        # convert_dict = {
        #     "col_name": {},
        #     "col_value": {},
        # }

        convert_dict = kwargs.get("convert_dict", None)  # convert cols name in shp to standard cols name
        if convert_dict is not None:
            col_name_convert_func = convert_dict["col_name"]
            col_value_convert_func = convert_dict["col_value"]
            if col_value_convert_func is not None and len(col_value_convert_func) > 0:
                for col, func in col_value_convert_func.items():
                    self.fields_df[col] = self.fields_df[col].apply(func)
            if col_name_convert_func is not None and len(col_name_convert_func) > 0:
                self.fields_df.rename(columns=col_name_convert_func, inplace=True)

        # 遍历每个地块
        self.fields = self.get_fields(self.fields_df, self.crs)
        self.gis_index_type = kwargs.get("gis_index_type", "arcgis")

    def __getitem__(self, item):
        return self.fields[item]

    def __len__(self):
        return len(self.fields)

    def __str__(self):
        return f"Farm : {self.name}, {len(self.fields)} fields"

    def set_name(self, name):
        """
        Set farm name
        :param name: str Farm name
        """
        self.name = name

    def update_fields(self, fields_df):
        """
        Update fields from fields_df
        """
        self.fields_df = fields_df
        self.fields = self.get_fields(self.fields_df, self.crs)

    def get_geoDataFrame(self):
        """
        Collect all fields' public properties to a GeoDataFrame

        :return: gpd.GeoDataFrame
        """
        farm_geoDataFrame_l = [i.to_geodataframe() for i in self.fields]
        farm_geoDataFrame = pd.concat(farm_geoDataFrame_l)
        return farm_geoDataFrame

    def to_file(self, save_path):
        """
        Save farm to file, support shp and geojson
        """
        self.fields_df.to_file(save_path)

    @staticmethod
    def get_fields(df: gpd.GeoDataFrame, crs) -> list:
        """
        convert GeoDataFrame to list of Field object

        :param df: gpd.GeoDataFrame
        :return: list, Field
        """
        fields = []
        for i in range(len(df)):
            row = df.iloc[i]
            field_geometry = row.geometry
            field = Field(pdseries=row, crs=crs, geometry=field_geometry)
            fields.append(field)
        return fields

    def split_multipolygon_fields(self):
        """
        if there are multipolygon in fields_df, split it to single polygon

        :return: None
        """
        df = self.get_geoDataFrame()
        df = self.split_multipolygon(df)
        # reset index
        df.reset_index(drop=True, inplace=True)
        self.fields_df = df
        self.fields = self.get_fields(df, self.crs)

    def plot(self):
        """
        plot farm use matplotlib

        :return: None
        """
        # self.fields_df.plot()
        # 绘制地图
        farm_geoDataFrame = self.fields_df
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # 设置图形大小
        farm_geoDataFrame.plot(ax=ax, color='lightblue', edgecolor='gray')  # 设置填充颜色和边界颜色
        ax.set_xlabel('Longitude', fontsize=20)
        ax.set_ylabel('Latitude', fontsize=20)
        plt.show()

    def split_multipolygon(self, x):
        """
        split multipolygon object, support GeoDataFrame and fiona.Collection

        :param x: GeoDataFrame or fiona.Collection
        """
        if isinstance(x, gpd.GeoDataFrame):
            x = self.split_multipolygon_in_geoDataFrame(x)
        elif isinstance(x, fiona.Collection):
            x = self.split_multipolygon_in_fiona_features(x)
        return x

    @staticmethod
    def split_multipolygon_in_geoDataFrame(geo_df: gpd.GeoDataFrame):
        """
        split multipolygon object in GeoDataFrame

        :param geo_df: GeoDataFrame
        :return: GeoDataFrame
        """
        # 创建一个空的列表来存放拆分后的 polygons 和对应的属性数据
        exploded_geo = []
        # 遍历 GeoDataFrame
        for _, row in geo_df.iterrows():
            # 选取几何体数据
            geom = row.geometry
            # 如果是 MultiPolygon 类型，拆分
            if isinstance(geom, MultiPolygon):
                # 遍历 MultiPolygon 中的每个 Polygon
                for poly in geom:
                    # 保持其他信息不变，只改变 geometry 列
                    new_row = row.drop('geometry').copy()
                    new_row['geometry'] = poly
                    # 将这行新增到列表中
                    exploded_geo.append(new_row)
            else:
                # 如果是 Polygon 或其他类型的几何体，直接增加到列表中
                exploded_geo.append(row)

        # 利用列表创建一个新的 GeoDataFrame
        return gpd.GeoDataFrame(exploded_geo, crs=geo_df.crs)

    @staticmethod
    def split_multipolygon_in_fiona_features(features: fiona.Collection):
        """
        split multipolygon object in fiona.Collection

        :param features: fiona.Collection
        :return: list
        """
        output_features = []
        for feature in features:
            geom = shape(feature['geometry'])
            properties = feature['properties'].copy()
            if feature['geometry']['type'] == 'Polygon':
                output_features.append({
                    'geometry': mapping(geom),
                    'properties': properties
                })
            if feature['geometry']['type'] == 'MultiPolygon':
                properties = feature['properties']
                for polygon in geom:
                    output_features.append({
                        'geometry': mapping(polygon),
                        'properties': properties
                    })
        return output_features

    def find_points_in_which_field(self, df: gpd.GeoDataFrame, split_multipolygon: bool):
        """
        input a point GeoDataFrame, find which field each point in.
        Warning: This method only use polygon to find which field each point in,

        :param points_df : GeoDataFrame, points GeoDataFrame
        :return: GeoDataFrame add field_id column in input GeoDataFrame, if the point not in any polygon,
        field_id is nan.
        """
        # check input
        # geometry should be Point
        if not self.is_point_gdf(df):
            raise Exception(f"geometry dtype: {df.geometry.dtype} not supported, must be Point")
        if 'field_id' in df.columns:
            raise Exception(f"field_id in df.columns, please rename it")

        if split_multipolygon:
            fields_df = self.split_multipolygon(self.fields_df)
            self.update_fields(fields_df)
        else:
            fields_df = self.fields_df

        fields_df['field_id'] = fields_df.index  # the same id as FID in arcgis
        result = gpd.sjoin(df, fields_df, how="left", op="within")
        df['field_id'] = result['field_id'].tolist()
        return df

    def is_point_gdf(self, gdf):
        """
        check GeoDataFrame is point or not

        :param gdf: GeoDataFrame
        :return: bool
        """
        # 检查是否存在geometry列
        if 'geometry' not in gdf.columns:
            return False

        # 检查是否所有几何对象的类型都是Point
        return all(geom.type == 'Point' for geom in gdf['geometry'])


class FarmWithOneImg:
    def __init__(self, img, farm):
        """
        A farm object with one image

        :param img: RSImg object
        :param farm: Farm object
        """
        if isinstance(img, str):
            img = RsImg.from_tif(img)
        if isinstance(farm, str):
            farm = Farm(file=farm)

        # unify the crs in img and farm
        img_crs = img.crs
        farm_crs = farm.crs
        if img_crs != farm_crs:
            raise Exception(f"img_crs: {img_crs} != farm_crs: {farm_crs}")

        self.img = img
        self.farm = farm

    def select_field(self, field_id):
        """
        select field by field_id

        :param field_id: int
        :return: Field object
        """
        field = self.farm.fields[field_id]
        field_geometry = field.geometry
        field_img = self.img.cut_by_geometry(field_geometry)
        field.register_img(field_img, field_img.name)
        return field

    def __len__(self):
        return len(self.farm.fields)

    def __getitem__(self, item):
        field = self.farm[item]
        field_geometry = field.geometry
        field_img = self.img.cut_by_geometry(field_geometry)
        field.register_img(field_img, field_img.name)
        return field

    # @staticmethod
    # def find_field_pos_in_farm(field_geoTransform: tuple, farm_geoTransform: tuple,
    #                            field_width_pixel: int, field_height_pixel: int):
    #     """
    #     Find the position of a field in a farm given their geoTransform information and pixel dimensions.
    #
    #     Parameters:
    #     - field_geoTransform (tuple): GeoTransform information of the field (x_min, x_res, _, y_min, _, y_res).
    #     - farm_geoTransform (tuple): GeoTransform information of the farm.
    #     - field_width_pixel (int): Width of the field in pixels.
    #     - field_height_pixel (int): Height of the field in pixels.
    #
    #     Returns:
    #     Tuple (col_min, row_max, col_max, row_min): The position of the field in the farm, expressed in pixel coordinates.
    #     - col_min (int): Minimum column index.
    #     - row_max (int): Maximum row index.
    #     - col_max (int): Maximum column index.
    #     - row_min (int): Minimum row index.
    #     """
    #
    #     x_min, x_res, _, y_min, _, y_res = field_geoTransform
    #     x_max = x_min + x_res * field_width_pixel
    #     y_max = y_min + y_res * field_height_pixel
    #     col_min = int(round((x_min - farm_geoTransform[0]) / farm_geoTransform[1], 0))
    #     row_max = int(round((y_max - farm_geoTransform[3]) / farm_geoTransform[5], 0))
    #     col_max = int(round((x_max - farm_geoTransform[0]) / farm_geoTransform[1], 0))
    #     row_min = int(round((y_min - farm_geoTransform[3]) / farm_geoTransform[5], 0))
    #     return col_min, row_max, col_max, row_min

    def yield_mask(self,
                   lossrate_df,
                   if_aug: bool,
                   gen_tif: bool,
                   save_path: str,
                   ):
        """
        my personal function

        """
        lossrate_df_copy = lossrate_df.copy()
        lossrate_df_copy = self.farm.find_points_in_which_field(lossrate_df_copy, True)
        # del nan
        lossrate_df_copy.dropna(inplace=True)
        field_id_list = list(lossrate_df_copy['field_id'].unique())

        # apply get yield algorithm
        yield_img_list = []
        for field_id in field_id_list:
            field = self[field_id]
            field_yield_img = gen_augyield_by_field(field=field)

            yield_img_list.append(field_yield_img)

        farm_yield_img = self.img.merge(yield_img_list)
        if gen_tif:
            farm_yield_img.to_tif(save_path=save_path)
            return farm_yield_img
        else:
            return farm_yield_img

    def croptype_mask(self):
        """
        my personal function

        """

        farm_crop_dict = {}
        # 将shp转换为tif，一个田块一个值，同时，将每个值对应的作物的种类也转化成字典， 这些都在内存中完成
        field_crop_type_img_list = []
        for i in range(len(self.farm)):
            field = self[i]
            field_crop_type_img = gen_crop_type_mask(field=field)
            field_crop_type_img_list.append(field_crop_type_img)

        croptype_img = self.img.merge(field_crop_type_img_list)
        # croptype_img = self.process_by_field_img(process_fun=gen_crop_type_mask,
        #                                          output_nodatavalue=65535,
        #                                          force_data_type=np.uint16)

        # 这里有一个问题，就是有可能有的田块相距过近，有的像素值在累加的时候会叠加。
        max_value = max(farm_crop_dict.keys())
        arr = croptype_img.ds.ReadAsArray()
        arr[arr > max_value] = 65535
        croptype_img.ds.WriteArray(arr)

        return croptype_img, farm_crop_dict

    def disaster_mask(self):
        """
        my personal function

        """
        raise NotImplementedError
