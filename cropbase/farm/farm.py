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


    def process_by_field_img(self, process_fun, output_nodatavalue=0, force_data_type=None, field_list=None):
        # """
        # Use rasterio.open(tif_path).meta to get the geoTransform information of the field.
        #
        # :param meta: rasterio.open(tif_path).meta
        # :return: geoTransform
        # """
        """
        Mapper the process function to each field in the farm, and return the processed result as a RSImg object.
        It's hard for me to compatiate gdal and rasterio.

        :param process_fun: The processing function applied to the field's image data. The param into the function \
        is using keyword param, and the param name is field. (field = kwargs["field"]).
        :param output_nodatavalue: The nodata value used in the output.
        :param force_data_type: Optional parameter to specify the desired data type for the processed output.
        :param field_list: Optional parameter to specify the desired field list for the processed output. If None,
        the field list is the farm's all field list.
        :return: RSImg object
        """
        # 1. test output shape
        if field_list is None:
            field_list = self.farm.fields
        # os.environ['CPL_DEBUG'] = 'ON'
        test_field = field_list[0]

        if self.img.name is not None:
            img_name = self.img.name
        else:
            img_name = 'default'
            self.img.set_name(img_name)


        tif_path = self.img.ds.GetDescription()

        # 使用rasterio.open()打开GDAL dataset对象
        with rasterio.open(tif_path) as ref_src:
            test_output = self.process_one_field_img(process_fun=process_fun,
                                                     field=test_field,
                                                     ref_src=ref_src,
                                                     output_nodatavalue=output_nodatavalue,
                                                     img_name=img_name,
                                                     force_data_type=force_data_type)
            test_field_output, test_field_geoTransform, test_field_height, test_field_width = test_output
            # 检查维度
            OUTPUT_DIM = len(test_field_output.shape)
            if OUTPUT_DIM == 2:
                output_height, output_width = test_field_output.shape
            elif OUTPUT_DIM == 3:
                output_bands, output_height, output_width = test_field_output.shape
            else:
                raise Exception(f"output dim: {OUTPUT_DIM} not supported")
            # 检查长宽是否正确
            if output_height != test_field_height or output_width != test_field_width:
                raise Exception(
                    f"output height: {output_height}, output width: {output_width} not equal " +
                    f"to testfield height: {test_field_height}, testfield width: {test_field_width}")
            else:
                pass
                # print(
                #     f"output height: {output_height}, output width: {output_width} equal " +
                #     f"to testfield height: {testfield_height}, testfield width: {testfield_width}")

            # 2. process
            if OUTPUT_DIM == 2:
                field_template_array = np.zeros((self.img.HEIGHT, self.img.WIDTH))
                # 全部填充为nan
                field_template_array[:] = np.nan
            elif OUTPUT_DIM == 3:
                field_template_array = np.zeros((output_bands, self.img.HEIGHT, self.img.WIDTH))
                # 全部填充为nan
                field_template_array[:] = np.nan
            else:
                raise Exception(f"output dim: {OUTPUT_DIM} not supported")

            farm_geoTransform = self.img.geoTransform

            for i, field in tqdm(enumerate(field_list)):
                output = self.process_one_field_img(process_fun=process_fun,
                                                    field=field,
                                                    ref_src=ref_src,
                                                    output_nodatavalue=output_nodatavalue,
                                                    img_name=img_name,
                                                    force_data_type=force_data_type)

                field_output, field_geoTransform, field_height, field_width = output

                col_min, row_max, col_max, row_min = self.find_field_pos_in_farm(field_geoTransform,
                                                                                 farm_geoTransform,
                                                                                 field_width,
                                                                                 field_height)

                if OUTPUT_DIM == 2:
                    field_template_array[row_min:row_max, col_min:col_max] = \
                        np.where(
                            np.isnan(field_template_array[row_min:row_max, col_min:col_max]) & np.isnan(
                                field_output),
                            np.nan,
                            np.nan_to_num(field_template_array[row_min:row_max, col_min:col_max], nan=0) +
                            np.nan_to_num(field_output, nan=0))
                elif OUTPUT_DIM == 3:
                    field_template_array[:, row_min:row_max, col_min:col_max] = \
                        np.where(
                            np.isnan(field_template_array[:, row_min:row_max, col_min:col_max]) & np.isnan(
                                field_output),
                            np.nan,
                            np.nan_to_num(field_template_array[:, row_min:row_max, col_min:col_max], nan=0) +
                            np.nan_to_num(field_output, nan=0))

            if force_data_type is not None:
                # nan to nodatavalue
                field_template_array[np.isnan(field_template_array)] = output_nodatavalue
                field_template_array = field_template_array.astype(force_data_type)

            farm_res = RSImg.from_array(array=field_template_array, nodatavalue=output_nodatavalue,
                                        projection=self.img.projection, geoTransform=self.img.geoTransform)

        # delete tif
        # if IS_MEM:
        #     time.sleep(1)
        #     os.remove(tif_path)
        #     print(f"Delete temp tif {tif_path}")
        return farm_res





    def process_one_field_img(self,
                              process_fun,
                              field: Field,
                              ref_src,
                              output_nodatavalue,
                              img_name: str,
                              force_data_type=None
                              ):
        """
        Process the image data of a single field using a specified processing function.

        :params process_fun (callable): The processing function applied to the field's image data.
        :params field (Field): The field object containing geometry information.
        :params ref_src: The reference source (rasterio dataset) used for masking the field's image.
        :params output_nodatavalue: The nodata value used in the output.
        :params img_name (str): The name associated with the field's image.
        :params force_data_type (type): Optional parameter to specify the desired data type for the processed output.
        :return: Tuple (output, field_transform, field_height, field_width):\
            - output: The processed output obtained from the specified processing function.\
            - field_transform: The transformation matrix for the field's image data.\
            - field_height (int): The height (number of rows) of the field's image data.\
            - field_width (int): The width (number of columns) of the field's image data.
        """
        field_meta = ref_src.meta.copy()
        field_projection = self.img.projection
        field_polygon = shape(field.geometry)
        field_array, field_transform = rasterio.mask.mask(ref_src, [field_polygon], crop=True)
        field_height, field_width = field_array.shape[-2:]
        field_meta.update({"height": field_height,
                           "width": field_width,
                           "transform": field_transform})
        field_geoTransform = self.field_meta2geoTransform(field_meta)

        # here should be use the same class as self.img
        img_class = self.img.__class__
        field_img = img_class.from_array(array=field_array, nodatavalue=self.img.nodatavalue,
                                         projection=field_projection, geoTransform=field_geoTransform)
        field.register_img(field_img, img_name)

        output = process_fun(field=field)

        field.deregister_img(img_name)
        return output, field_geoTransform, field_height, field_width


    @staticmethod
    def find_field_pos_in_farm(field_geoTransform: tuple, farm_geoTransform: tuple,
                               field_width_pixel: int, field_height_pixel: int):
        """
        Find the position of a field in a farm given their geoTransform information and pixel dimensions.

        Parameters:
        - field_geoTransform (tuple): GeoTransform information of the field (x_min, x_res, _, y_min, _, y_res).
        - farm_geoTransform (tuple): GeoTransform information of the farm.
        - field_width_pixel (int): Width of the field in pixels.
        - field_height_pixel (int): Height of the field in pixels.

        Returns:
        Tuple (col_min, row_max, col_max, row_min): The position of the field in the farm, expressed in pixel coordinates.
        - col_min (int): Minimum column index.
        - row_max (int): Maximum row index.
        - col_max (int): Maximum column index.
        - row_min (int): Minimum row index.
        """

        x_min, x_res, _, y_min, _, y_res = field_geoTransform
        x_max = x_min + x_res * field_width_pixel
        y_max = y_min + y_res * field_height_pixel
        col_min = int(round((x_min - farm_geoTransform[0]) / farm_geoTransform[1], 0))
        row_max = int(round((y_max - farm_geoTransform[3]) / farm_geoTransform[5], 0))
        col_max = int(round((x_max - farm_geoTransform[0]) / farm_geoTransform[1], 0))
        row_min = int(round((y_min - farm_geoTransform[3]) / farm_geoTransform[5], 0))
        return col_min, row_max, col_max, row_min

    def yield_mask(self,
                   lossrate_df,
                   if_aug: bool,
                   gentif: bool,
                   save_path: str,
                   add_pos=True,
                   method='kmeans',
                   cluster_number=15,
                   select_bands=None
                   ):
        """
        my personal function

        """
        lossratedf = lossrate_df.copy()
        lossratedf = self.farm.find_points_in_which_field(lossratedf, True)
        # del nan
        lossratedf.dropna(inplace=True)

        lossrate_df = lossratedf
        field_id_list = list(lossratedf['field_id'].unique())
        # 根据结果遍历每个田块
        field_list = [self.farm.fields[int(i)] for i in field_id_list]

        lossrate_name = 'lossrate'

        def gen_augyield_by_field(**kwargs):
            field = kwargs["field"]
            field_id = field.index
            field_sample_yield = lossrate_df[lossrate_df['field_id'] == field_id]
            img = field.get_img(self.img.name)
            point_df = field.filter_point_value_df(field_sample_yield, img)
            if len(point_df) == 0:
                yield_mask = np.zeros((img.HEIGHT, img.WIDTH))
                yield_mask[:] = np.nan
                return yield_mask

            cluster_mask = img.cluster(cluster_number=cluster_number, if_add_position_encoding=add_pos,
                                       method=method,select_bands=select_bands).ds.ReadAsArray()
            yield_mask = field.aug_mask(cluster_mask, location_df=point_df, type='mean',
                                        value_col_name=lossrate_name)
            return yield_mask

        def gen_yield_by_field(**kwargs):

            field = kwargs["field"]
            field_id = field.index
            field_sample_yield = lossrate_df[lossrate_df['field_id'] == field_id]
            img = field.get_img(self.img.name)
            point_df = field.filter_point_value_df(field_sample_yield, img)
            if len(point_df) == 0:
                yield_mask = np.zeros((img.HEIGHT, img.WIDTH))
                yield_mask[:] = np.nan
                return yield_mask

            # 直接将点映射到tif中
            yield_mask = np.zeros((img.HEIGHT, img.WIDTH))
            yield_mask[:] = np.nan

            for i in range(len(point_df)):
                col_idx = point_df.iloc[i]['col_idx']
                row_idx = point_df.iloc[i]['row_idx']
                col_idx = int(col_idx)
                row_idx = int(row_idx)
                yield_mask[row_idx, col_idx] = point_df.iloc[i][lossrate_name]

            return yield_mask

        if if_aug:
            yield_img = self.process_by_field_img(process_fun=gen_augyield_by_field,
                                                  output_nodatavalue=-1,
                                                  force_data_type=np.float32,
                                                  field_list=field_list)
        else:
            yield_img = self.process_by_field_img(process_fun=gen_yield_by_field,
                                                  output_nodatavalue=-1,
                                                  force_data_type=np.float32,
                                                  field_list=field_list)

        if gentif:
            yield_img.to_tif(save_path=save_path)
            return yield_img
        else:
            return yield_img

    def croptype_mask(self):
        """
        my personal function

        """

        crop_dict2023 = {
            "1": "rice",
            "2": "maize",
            "3": "soybean",
            "4": "other",
            "5": "wheat"
        }

        farm_crop_dict = {}

        # 将shp转换为tif，一个田块一个值，同时，将每个值对应的作物的种类也转化成字典， 这些都在内存中完成
        def gen_crop_type_mask(**kwargs):
            field = kwargs["field"]
            field_id = field.index
            img = field.get_img(self.img.name)
            img_valid_mask = img.valid_mask.copy()
            img_valid_mask = img_valid_mask.astype(np.float32)
            img_valid_mask[img_valid_mask == 0] = np.nan
            img_valid_mask[img_valid_mask == 1] = int(field_id)

            farm_crop_dict[field_id] = crop_dict2023[field.public_properties['CROP_ID']]
            return img_valid_mask

        croptype_img = self.process_by_field_img(process_fun=gen_crop_type_mask,
                                                 output_nodatavalue=65535,
                                                 force_data_type=np.uint16)

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
