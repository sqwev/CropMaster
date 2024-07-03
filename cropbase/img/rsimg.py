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
from osgeo import gdal
from osgeo.gdal import Dataset as gdalDataset

from .vegetation_vis import VegetationVIs
from ..utils import gdal_array_type

class RestrictedDict:
    def __init__(self, allowed_type):
        self.allowed_type = allowed_type
        self.internal_dict = {}

    def __setitem__(self, key, value):
        if isinstance(value, self.allowed_type):
            self.internal_dict[key] = value
        else:
            raise ValueError(f"Value must be of type {self.allowed_type}")

    def __getitem__(self, key):
        return self.internal_dict[key]

    def __delitem__(self, key):
        del self.internal_dict[key]

    def __repr__(self):
        return repr(self.internal_dict)

    def __len__(self):
        return len(self.internal_dict)


class RSImg:
    """
    provide 2 ways to create RSImg; 1. from_array; 2. from_tif
    We recommend to create RsImg object with property `name` and `date`


    if `RsImg` object is created without `name` and `date`, the `name` will be set to `uuid.uuid1()`,
    the date will be set to `datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`.

    the useful property of `RsImg` object is:

    - `ds`: gdal dataset, campatible with gdal API
    - `name`: the name of the image
    - `date`: the date of the image
    - `nodatavalue`: the nodatavalue of the image
    - `geotransform`: the geotransform of the image
    - `projection`: the projection of the image
    - `WIDTH`: the width of the image
    - `HEIGHT`: the height of the image
    - `BANDS`: the bands of the image
    - `dim`: the dimension of the image
    - `x_min, x_max, y_min, y_max`: the boundary of the image

    suggest property init in __init__ function
    name: str
    date: str
    """

    # if geoTransform is not None:
    #     self.x_min, self.x_res, self.rotation, self.y_min, self.rotation, self.y_res = self.geoTransform
    #     self.x_max = self.x_min + self.x_res * self.WIDTH
    #     self.y_max = self.y_min + self.y_res * self.HEIGHT

    def __init__(self, ds: gdalDataset, *args, **kwargs) -> None:
        self.ds = ds
        self.name = kwargs.get("name", uuid.uuid1())
        self.date = kwargs.get("date", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not isinstance(self.date, str):
            raise Exception(f"date: {self.date} must be str type")

        self.projection = ds.GetProjection()
        self.geoTransform = ds.GetGeoTransform()
        self.WIDTH = ds.RasterXSize
        self.HEIGHT = ds.RasterYSize
        self.BANDS = ds.RasterCount
        self.dim = 2 if self.BANDS == 1 else 3
        self.x_min, self.x_res, self.rotation, self.y_min, self.rotation, self.y_res = self.geoTransform
        self.x_max = self.x_min + self.x_res * self.WIDTH
        self.y_max = self.y_min + self.y_res * self.HEIGHT
        self.nodatavalue = ds.GetRasterBand(1).GetNoDataValue()

    @classmethod
    def from_array(cls, array, nodatavalue, projection, geoTransform, *args, **kwargs):
        """
        Create RsImg object from array

        :param array: np.array  The array of the image
        :param nodatavalue: The nodatavalue of the image
        :param projection: The projection of the image
        :param geoTransform: The geoTransform of the image
        """
        gdal_type = gdal_array_type(array.dtype)
        dim = len(array.shape)
        if dim != 2 and dim != 3:
            raise Exception(f"Field array dimemsion: {dim} not supported, only support 2 or 3 dim array")

        if dim == 2:
            BANDS = 1
            HEIGHT = array.shape[0]
            WIDTH = array.shape[1]
        else:
            BANDS = array.shape[0]
            HEIGHT = array.shape[1]
            WIDTH = array.shape[2]

        mem_driver = gdal.GetDriverByName('MEM')
        # get source epsg
        ds = mem_driver.Create('', WIDTH, HEIGHT, BANDS, gdal_type)
        ds.SetProjection(projection)
        ds.SetGeoTransform(geoTransform)
        if BANDS == 1:
            ds.GetRasterBand(1).WriteArray(array)
            ds.GetRasterBand(1).SetNoDataValue(nodatavalue)
        else:
            for i in range(BANDS):
                ds.GetRasterBand(i + 1).WriteArray(array[i])
                ds.GetRasterBand(i + 1).SetNoDataValue(nodatavalue)
        return cls(ds, *args, **kwargs)

    @classmethod
    def from_tif(cls, tif_path: str, *args, **kwargs):
        """
        Create RsImg object from tif file

        :param tif_path: str   The path of the tif file
        """
        return cls(gdal.Open(tif_path), *args, **kwargs)

    # 析构函数
    def __del__(self):
        self.ds = None
        self.projection = None
        self.geoTransform = None
        self.x_min = None
        self.x_res = None
        self.rotation = None
        self.y_min = None
        self.rotation = None
        self.y_res = None
        self.x_max = None
        self.y_max = None
        self.WIDTH = None
        self.HEIGHT = None
        self.BANDS = None
        self.array = None
        self.nodatavalue = None
        self.dim = None
        self.name = None

    @property
    def valid_mask(self):
        """
        Generate a 2d array, 1 for valid, 0 for invalid
        """
        nodatavalue = self.nodatavalue
        # read 1 band
        array = self.ds.GetRasterBand(1).ReadAsArray()

        # 不是nodatavalue的地方是有效值，生成一个二维数组，1为有效值，0为无效值
        if nodatavalue is None:
            valid_mask = np.ones((self.HEIGHT, self.WIDTH))
        else:
            valid_mask = np.where(array == nodatavalue, 0, 1)
        return valid_mask

    @property
    def espg(self):
        """
        Get the epsg of the image
        """
        assert self.projection is not None, "projection is None"
        return int(self.projection.split('"')[-2])

    @property
    def border(self):
        """
        Get the border of the image, but has not been Implemented
        """
        from skimage import measure
        from shapely.geometry import Polygon
        import numpy as np

        def boolean_array_to_polygons(boolean_array):
            # 寻找所有轮廓在 0.5 的水平高度值
            contours = measure.find_contours(boolean_array, 0.5)
            polygons = [Polygon(contour) for contour in contours if len(contour) > 2]  # 忽略小于3个点的轮廓
            return polygons

        # 创建一个布尔数组示例
        bool_array = np.array(
            [[False, False, False, False, False],
             [False, True, True, True, False],
             [False, True, False, True, False],
             [False, True, True, True, False],
             [False, False, False, False, False]])

        # 转换布尔数组为多边形
        polygons = boolean_array_to_polygons(bool_array)

        for polygon in polygons:
            print(polygon)


        raise NotImplementedError


    def register_band(self, band: dict):
        """
        register band, not supported now
        default_band = {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3
        }
        """
        # 如果有4个波段，默认为blue, green, red, nir
        default_band = {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3
        }
        if band is None:
            self.band_register = default_band
        else:
            self.band_register = band

    def set_name(self, name: str):
        """
        Set the name of the image

        :param name: str   The name of the image
        """
        self.name = name


    # 保存影像
    def to_tif(self, save_path):
        """
        Save RsImg object to tif file

        :param save_path: str   The path of save tif file
        """
        # 必须有projection和geoTransform
        if self.projection is None or self.geoTransform is None:
            raise Exception(f"projection or geoTransform is None")

        driver = gdal.GetDriverByName('GTiff')
        creation_options = [
            'TILED=YES',  # 设置平铺以改善性能
            'COMPRESS=LZW',  # 应用LZW压缩来减少文件大小
            # 'PREDICTOR=2',        # 对于带有浮点型数据的图像，使用预测器
            'BIGTIFF=IF_SAFER',  # 如果数据大小需要，使用BIGTIFF
            # 'BLOCKXSIZE=256',     # 设置块大小X
            # 'BLOCKYSIZE=256'      # 设置块大小Y
        ]
        output_dataset = driver.CreateCopy(save_path, self.ds, options=creation_options)
        output_dataset.FlushCache()
        output_dataset = None

    def crop(self, left, top, right, bottom):
        """
        Crop the image by pixel serial number

        :param left: int   The left pixel serial number
        :param top: int   The top pixel serial number
        :param right: int   The right pixel serial number
        :param bottom: int   The bottom pixel serial number
        """
        if self.dim == 2:
            crop_array = self.ds.ReadAsArray(left, top, right - left, bottom - top)
        else:
            crop_array = self.ds.ReadAsArray(left, top, right - left, bottom - top)

        # 根据偏移量，重新计算geoTransform
        origin_x, pixel_width, rotation, origin_y, rotation, pixel_height = self.geoTransform
        new_x_min = origin_x + pixel_width * left
        new_y_min = origin_y + pixel_height * top
        new_geoTransform = (new_x_min, pixel_width, rotation, new_y_min, rotation, pixel_height)

        return RSImg.from_array(crop_array, nodatavalue=self.nodatavalue, projection=self.projection,
                                geoTransform=new_geoTransform)

    def select_bands(self, band_list):
        """
        Select bands from RsImg object

        :param band_list: list or None   The band list to select
        """

        if self.dim == 2:
            raise Exception(f"array dim: {self.dim} not supported")
        else:
            if isinstance(band_list, list):

                # 通过循环读取所选波段数据为 NumPy 数组
                selected_bands_data = []
                for band_index in band_list:
                    band = self.ds.GetRasterBand(band_index)
                    band_data = band.ReadAsArray()
                    selected_bands_data.append(band_data)

                # 将所选波段数据转换为 NumPy 数组
                selected_bands_array = np.array(selected_bands_data)
                return RSImg.from_array(selected_bands_array, nodatavalue=self.nodatavalue, projection=self.projection,
                                        geoTransform=self.geoTransform)
            else:
                return self

    # 切片
    def sliding_window_crop(self, save_dir, block_size: int, repetition_rate: float, nodatavalue):
        """
        Sliding window crop the image

        :param save_dir: str   The path to save the crop image
        :param block_size: int   The size of the crop image
        :param repetition_rate: float   The repetition rate of the crop image
        :param nodatavalue:    The nodatavalue of the image
        """
        # self.tif_path is not None
        # assert self.tif_path is not None, "self.tif_path is None, only support tif format now"
        # tif_path = self.tif_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if repetition_rate < 0 or repetition_rate > 1:
            raise Exception(f"repetition_rate: {repetition_rate} not in [0,1]")

        src_ds = self.ds
        geo_transform = src_ds.GetGeoTransform()
        nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()
        x_size = src_ds.RasterXSize
        y_size = src_ds.RasterYSize

        # 计算重复率对应的像素数
        step_size = int(block_size * (1 - repetition_rate))

        for row in range(0, y_size - block_size + 1, step_size):
            for col in range(0, x_size - block_size + 1, step_size):
                # print(f"row: {row}, col: {col}")
                window = (col, row, block_size, block_size)

                band = src_ds.GetRasterBand(1)
                crop = band.ReadAsArray(col, row, block_size, block_size)

                # If crop contains only nodata values, don't save
                if np.all(crop == nodatavalue):
                    print(f"crop: all nodatavalue")
                    continue

                # Generate the path for the cropped file
                filename = f"crop_{row}_{col}.tif"
                filename = os.path.join(save_dir, filename)

                driver = gdal.GetDriverByName('GTiff')
                dst_ds = driver.Create(filename, block_size, block_size, 1, band.DataType)
                dst_ds.SetGeoTransform((geo_transform[0] + col * geo_transform[1], geo_transform[1], 0,
                                        geo_transform[3] + row * geo_transform[5], 0, geo_transform[5]))
                dst_ds.SetProjection(src_ds.GetProjection())
                dst_ds.GetRasterBand(1).WriteArray(crop)

                # Close the cropped file
                dst_ds = None

        src_ds = None
        print("裁剪完成！")

        # with rasterio.open(tif_path) as src:
        #     image = src.read()
        #     height, width = src.shape
        #
        #     # 计算重复率对应的像素数
        #     step_size = int(block_size * (1 - repetition_rate))
        #
        #     for row in range(0, height - block_size + 1, step_size):
        #
        #         for col in range(0, width - block_size + 1, step_size):
        #             print(f"row: {row}, col: {col}")
        #             window = Window(col, row, block_size, block_size)
        #             crop = src.read(window=window)
        #
        #             # 如果crop中全是有nodata值，就不保存
        #
        #             arr_dim = len(crop.shape)
        #             if arr_dim == 2:
        #                 if np.all(crop == nodatavalue):
        #                     print(f"crop: all nodatavalue")
        #                     continue
        #             elif arr_dim == 3:
        #                 if np.all(crop[0] == nodatavalue):
        #                     print(f"crop: all nodatavalue")
        #                     continue
        #
        #             # 生成裁剪后文件的路径
        #
        #             filename = f"crop_{row}_{col}.tif"
        #             filename = os.path.join(save_dir, filename)
        #
        #             # 将裁剪后的数据写入输出影像
        #             profile = src.profile
        #             profile['width'], profile['height'] = block_size, block_size
        #             profile['transform'] = rasterio.windows.transform(window, src.transform)
        #             with rasterio.open(filename, 'w', **profile) as dst:
        #                 dst.write(crop)
        # print("裁剪完成！")

    # 计算

    def cluster(self, cluster_number,
                if_add_position_encoding=True,
                method='kmeans',
                select_bands=None,
                filter_function=None):
        """
        Cluster the image by different method

        :param cluster_number: int   The number of cluster
        :param if_add_position_encoding: bool   If add position encoding
        :param method: str   The method of cluster
        :param select_bands: list   The bands to select
        :param filter_function: function   The function to filter the data
        """
        # 根据select_bands选择波段
        if select_bands is None:
            select_array = self.ds.ReadAsArray()
            nodata_value = self.nodatavalue
        else:
            select_ds = self.select_bands(select_bands)
            select_array = select_ds.ds.ReadAsArray()
            nodata_value = select_ds.nodatavalue

        out_image = select_array

        # print(f"out_image.shape: {out_image.shape}")

        def field_filter(df, nodatavalue):
            df = df[~df['B1'].isin([nodatavalue])]
            return df

        def cluster_preprocess(array, if_add_position_encoding, select_bands: list = None):
            """
            在原来的基础上扩展两个波段，分别是x，y的行列号
            """
            if select_bands is not None:
                if self.BANDS == 1:
                    return array
                else:
                    array = array[select_bands, :, :]

            # 重新计算arr的shape
            dim = len(array.shape)
            if dim != 2 and dim != 3:
                raise Exception(f"Field array dimemsion: {dim} not supported")
            if dim == 2:
                BANDS = 1
            else:
                BANDS = array.shape[0]

            # 在原来的基础上扩展两个波段，分别是x，y
            if if_add_position_encoding:
                extended_array = np.zeros((BANDS + 2, self.HEIGHT, self.WIDTH), dtype=array.dtype)
                x, y = np.meshgrid(np.arange(self.WIDTH), np.arange(self.HEIGHT))
                # 将x y 旋转
                # x = x.transpose(1, 0)
                # y = y.transpose(1, 0)
                extended_array[:BANDS, :, :] = array
                extended_array[BANDS, :, :] = x
                extended_array[BANDS + 1, :, :] = y
                return extended_array, BANDS
            else:
                return array, BANDS

        def builtin_cluster(df, cluster_number, method='kmeans'):
            """
            对df进行聚类，返回聚类结果，并根据聚类结果对等级进行排序，分为1到n级
            """
            if method == "kmeans":
                from sklearn.cluster import KMeans
                clustering = KMeans(n_clusters=cluster_number).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "dbscan":
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.3, min_samples=10).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "meanshift":
                from sklearn.cluster import MeanShift
                clustering = MeanShift().fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "spectral":
                from sklearn.cluster import SpectralClustering
                clustering = SpectralClustering(n_clusters=cluster_number).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=cluster_number).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "gaussian":
                from sklearn.mixture import GaussianMixture
                clustering = GaussianMixture(n_components=cluster_number).fit(df)
                labels = clustering.predict(df)
                centers = clustering.cluster_centers_
            elif method == "hierarchical":
                from sklearn.cluster import Birch
                clustering = Birch(n_clusters=cluster_number).fit(df)
                labels = clustering.predict(df)
                centers = clustering.cluster_centers_
            else:
                raise Exception(f"method: {method} not supported")

            # print(f"centers: {centers}")
            # 计算每个center对应的ndvi
            ndvi_list = []

            def cal_ndvi(red, nir):
                return (nir - red) / (nir + red)

            for center in centers:
                ndvi = cal_ndvi(center[2], center[3])
                ndvi_list.append(ndvi)
            # print(f"ndvi_list: {ndvi_list}")

            # 根据聚类中心的ndvi值对等级进行排序
            ndvi_list = np.array(ndvi_list)
            sort_index = np.argsort(ndvi_list)
            labels_dict = {sort_index[i]: i for i in range(len(sort_index))}
            labels = [labels_dict[i] + 1 for i in labels]
            # sort_index = np.argsort(ndvi_list)
            # # print(f"sort_index: {sort_index}")
            # labels_dict = {}
            # for i in range(len(sort_index)):
            #     labels_dict[sort_index[i]] = i
            #
            # labels = [labels_dict[i] + 1 for i in labels]
            # # print(f"labels: {labels}")
            df["label"] = labels
            return df

        array, BANDS = cluster_preprocess(out_image, if_add_position_encoding=if_add_position_encoding)
        reshape_array = array.transpose(2, 1, 0)
        reshape_array = reshape_array.reshape(
            -1, reshape_array.shape[-1], order='C')

        if if_add_position_encoding:
            columns = [f"B{i+1}" for i in range(BANDS)] + ["x", "y"]
        else:
            columns = [f"B{i+1}" for i in range(BANDS)]
        rawdf = pd.DataFrame(reshape_array)
        rawdf.columns = columns

        # print(f"rawdf: {rawdf}")

        # 过滤掉nodatavalue
        filter_df = field_filter(rawdf, nodata_value)
        if filter_function is not None:
            filter_df = filter_function(filter_df)

        if len(filter_df) != 0 and len(rawdf) == 0:
            raise Exception("can't be use cluster, because all data in field is cloud")

        # uniform filtered_df each column to 0-1
        filtered_df_copy = filter_df.copy()
        for col in filtered_df_copy.columns:
            filtered_df_copy[col] = (filtered_df_copy[col] - filtered_df_copy[col].min()) / (
                    filtered_df_copy[col].max() - filtered_df_copy[col].min())

        # -----------------聚类-----------------
        clustered_df = builtin_cluster(filtered_df_copy, cluster_number)
        cluster_label = np.full(self.WIDTH * self.HEIGHT, -1)  # 创建一个填充了 -1 的数组

        indices = clustered_df.index.values
        mask = np.isin(np.arange(self.WIDTH * self.HEIGHT), indices)  # 创建一个布尔掩码来检查索引是否存在于 clustered_df 中
        cluster_label[mask] = clustered_df.loc[indices, "label"].values  # 使用布尔掩码来更新 cluster_label

        save_array = cluster_label.reshape(self.WIDTH, self.HEIGHT)
        save_array = save_array.transpose(1, 0)

        return RSImg.from_array(save_array, nodatavalue=-1, projection=self.projection, geoTransform=self.geoTransform)


class NewRSImg:
    """
    provide 2 ways to create RSImg; 1. from_array; 2. from_tif
    We recommend to create RsImg object with property `name` and `date`


    if `RsImg` object is created without `name` and `date`, the `name` will be set to `uuid.uuid1()`,
    the date will be set to `datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`.

    the useful property of `RsImg` object is:

    - `ds`: rasterio dataset, campatible with rasterio API
    - `name`: the name of the image
    - `date`: the date of the image
    - `nodatavalue`: the nodatavalue of the image
    - `geotransform`: the geotransform of the image
    - `projection`: the projection of the image
    - `WIDTH`: the width of the image
    - `HEIGHT`: the height of the image
    - `BANDS`: the bands of the image
    - `dim`: the dimension of the image
    - `x_min, x_max, y_min, y_max`: the boundary of the image

    suggest property init in __init__ function
    name: str
    date: str
    """

    # if geoTransform is not None:
    #     self.x_min, self.x_res, self.rotation, self.y_min, self.rotation, self.y_res = self.geoTransform
    #     self.x_max = self.x_min + self.x_res * self.WIDTH
    #     self.y_max = self.y_min + self.y_res * self.HEIGHT

    def __init__(self, ds, *args, **kwargs) -> None:
        self.name = kwargs.get("name", uuid.uuid1())
        self.date = kwargs.get("date", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not isinstance(self.date, str):
            raise Exception(f"date: {self.date} must be str type")

        self.ds = ds
        self.projection = ds.crs
        self.geoTransform = ds.transform
        self.width = ds.width
        self.height = ds.height
        self.BANDS = ds.count
        self.dim = 2 if self.BANDS == 1 else 3
        self.x_min, self.x_res, self.rotation, self.y_min, self.rotation, self.y_res = self.geoTransform
        self.x_max = self.x_min + self.x_res * self.width
        self.y_max = self.y_min + self.y_res * self.height


    @classmethod
    def from_array(cls, array, nodatavalue, projection, geoTransform, *args, **kwargs):
        """
        Create RsImg object from array

        :param array: np.array  The array of the image
        :param nodatavalue: The nodatavalue of the image
        :param projection: The projection of the image
        :param geoTransform: The geoTransform of the image
        """
        gdal_type = gdal_array_type(array.dtype)
        dim = len(array.shape)
        if dim != 2 and dim != 3:
            raise Exception(f"Field array dimemsion: {dim} not supported, only support 2 or 3 dim array")

        if dim == 2:
            BANDS = 1
            HEIGHT = array.shape[0]
            WIDTH = array.shape[1]
        else:
            BANDS = array.shape[0]
            HEIGHT = array.shape[1]
            WIDTH = array.shape[2]

        mem_driver = gdal.GetDriverByName('MEM')
        # get source epsg
        ds = mem_driver.Create('', WIDTH, HEIGHT, BANDS, gdal_type)
        ds.SetProjection(projection)
        ds.SetGeoTransform(geoTransform)
        if BANDS == 1:
            ds.GetRasterBand(1).WriteArray(array)
            ds.GetRasterBand(1).SetNoDataValue(nodatavalue)
        else:
            for i in range(BANDS):
                ds.GetRasterBand(i + 1).WriteArray(array[i])
                ds.GetRasterBand(i + 1).SetNoDataValue(nodatavalue)
        return cls(ds, *args, **kwargs)

    @classmethod
    def from_tif(cls, tif_path: str, *args, **kwargs):
        """
        Create RsImg object from tif file

        :param tif_path: str   The path of the tif file
        """
        return cls(rasterio.open(tif_path), *args, **kwargs)

    # 析构函数
    def __del__(self):
        self.ds.close()
        self.projection = None
        self.geoTransform = None
        self.x_min = None
        self.x_res = None
        self.rotation = None
        self.y_min = None
        self.rotation = None
        self.y_res = None
        self.x_max = None
        self.y_max = None
        self.WIDTH = None
        self.HEIGHT = None
        self.BANDS = None
        self.array = None
        self.nodatavalue = None
        self.dim = None
        self.name = None

    @property
    def valid_mask(self):
        """
        Generate a 2d array, 1 for valid, 0 for invalid
        """
        nodatavalue = self.nodatavalue
        # read 1 band
        array = self.ds.GetRasterBand(1).ReadAsArray()

        # 不是nodatavalue的地方是有效值，生成一个二维数组，1为有效值，0为无效值
        if nodatavalue is None:
            valid_mask = np.ones((self.HEIGHT, self.WIDTH))
        else:
            valid_mask = np.where(array == nodatavalue, 0, 1)
        return valid_mask

    @property
    def espg(self):
        """
        Get the epsg of the image
        """
        assert self.projection is not None, "projection is None"
        return int(self.projection.split('"')[-2])

    @property
    def border(self):
        """
        Get the border of the image, but has not been Implemented
        """
        from skimage import measure
        from shapely.geometry import Polygon
        import numpy as np

        def boolean_array_to_polygons(boolean_array):
            # 寻找所有轮廓在 0.5 的水平高度值
            contours = measure.find_contours(boolean_array, 0.5)
            polygons = [Polygon(contour) for contour in contours if len(contour) > 2]  # 忽略小于3个点的轮廓
            return polygons

        # 创建一个布尔数组示例
        bool_array = np.array(
            [[False, False, False, False, False],
             [False, True, True, True, False],
             [False, True, False, True, False],
             [False, True, True, True, False],
             [False, False, False, False, False]])

        # 转换布尔数组为多边形
        polygons = boolean_array_to_polygons(bool_array)

        for polygon in polygons:
            print(polygon)


        raise NotImplementedError


    def register_band(self, band: dict):
        """
        register band, not supported now
        default_band = {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3
        }
        """
        # 如果有4个波段，默认为blue, green, red, nir
        default_band = {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3
        }
        if band is None:
            self.band_register = default_band
        else:
            self.band_register = band

    def set_name(self, name: str):
        """
        Set the name of the image

        :param name: str   The name of the image
        """
        self.name = name


    # 保存影像
    def to_tif(self, save_path):
        """
        Save RsImg object to tif file

        :param save_path: str   The path of save tif file
        """
        # 必须有projection和geoTransform
        if self.projection is None or self.geoTransform is None:
            raise Exception(f"projection or geoTransform is None")

        driver = gdal.GetDriverByName('GTiff')
        creation_options = [
            'TILED=YES',  # 设置平铺以改善性能
            'COMPRESS=LZW',  # 应用LZW压缩来减少文件大小
            # 'PREDICTOR=2',        # 对于带有浮点型数据的图像，使用预测器
            'BIGTIFF=IF_SAFER',  # 如果数据大小需要，使用BIGTIFF
            # 'BLOCKXSIZE=256',     # 设置块大小X
            # 'BLOCKYSIZE=256'      # 设置块大小Y
        ]
        output_dataset = driver.CreateCopy(save_path, self.ds, options=creation_options)
        output_dataset.FlushCache()
        output_dataset = None

    def crop(self, left, top, right, bottom):
        """
        Crop the image by pixel serial number

        :param left: int   The left pixel serial number
        :param top: int   The top pixel serial number
        :param right: int   The right pixel serial number
        :param bottom: int   The bottom pixel serial number
        """
        if self.dim == 2:
            crop_array = self.ds.ReadAsArray(left, top, right - left, bottom - top)
        else:
            crop_array = self.ds.ReadAsArray(left, top, right - left, bottom - top)

        # 根据偏移量，重新计算geoTransform
        origin_x, pixel_width, rotation, origin_y, rotation, pixel_height = self.geoTransform
        new_x_min = origin_x + pixel_width * left
        new_y_min = origin_y + pixel_height * top
        new_geoTransform = (new_x_min, pixel_width, rotation, new_y_min, rotation, pixel_height)

        return RSImg.from_array(crop_array, nodatavalue=self.nodatavalue, projection=self.projection,
                                geoTransform=new_geoTransform)

    def select_bands(self, band_list):
        """
        Select bands from RsImg object

        :param band_list: list or None   The band list to select
        """

        if self.dim == 2:
            raise Exception(f"array dim: {self.dim} not supported")
        else:
            if isinstance(band_list, list):

                # 通过循环读取所选波段数据为 NumPy 数组
                selected_bands_data = []
                for band_index in band_list:
                    band = self.ds.GetRasterBand(band_index)
                    band_data = band.ReadAsArray()
                    selected_bands_data.append(band_data)

                # 将所选波段数据转换为 NumPy 数组
                selected_bands_array = np.array(selected_bands_data)
                return RSImg.from_array(selected_bands_array, nodatavalue=self.nodatavalue, projection=self.projection,
                                        geoTransform=self.geoTransform)
            else:
                return self

    # 切片
    def sliding_window_crop(self, save_dir, block_size: int, repetition_rate: float, nodatavalue):
        """
        Sliding window crop the image

        :param save_dir: str   The path to save the crop image
        :param block_size: int   The size of the crop image
        :param repetition_rate: float   The repetition rate of the crop image
        :param nodatavalue:    The nodatavalue of the image
        """
        # self.tif_path is not None
        # assert self.tif_path is not None, "self.tif_path is None, only support tif format now"
        # tif_path = self.tif_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if repetition_rate < 0 or repetition_rate > 1:
            raise Exception(f"repetition_rate: {repetition_rate} not in [0,1]")

        src_ds = self.ds
        geo_transform = src_ds.GetGeoTransform()
        nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()
        x_size = src_ds.RasterXSize
        y_size = src_ds.RasterYSize

        # 计算重复率对应的像素数
        step_size = int(block_size * (1 - repetition_rate))

        for row in range(0, y_size - block_size + 1, step_size):
            for col in range(0, x_size - block_size + 1, step_size):
                # print(f"row: {row}, col: {col}")
                window = (col, row, block_size, block_size)

                band = src_ds.GetRasterBand(1)
                crop = band.ReadAsArray(col, row, block_size, block_size)

                # If crop contains only nodata values, don't save
                if np.all(crop == nodatavalue):
                    print(f"crop: all nodatavalue")
                    continue

                # Generate the path for the cropped file
                filename = f"crop_{row}_{col}.tif"
                filename = os.path.join(save_dir, filename)

                driver = gdal.GetDriverByName('GTiff')
                dst_ds = driver.Create(filename, block_size, block_size, 1, band.DataType)
                dst_ds.SetGeoTransform((geo_transform[0] + col * geo_transform[1], geo_transform[1], 0,
                                        geo_transform[3] + row * geo_transform[5], 0, geo_transform[5]))
                dst_ds.SetProjection(src_ds.GetProjection())
                dst_ds.GetRasterBand(1).WriteArray(crop)

                # Close the cropped file
                dst_ds = None

        src_ds = None
        print("裁剪完成！")

        # with rasterio.open(tif_path) as src:
        #     image = src.read()
        #     height, width = src.shape
        #
        #     # 计算重复率对应的像素数
        #     step_size = int(block_size * (1 - repetition_rate))
        #
        #     for row in range(0, height - block_size + 1, step_size):
        #
        #         for col in range(0, width - block_size + 1, step_size):
        #             print(f"row: {row}, col: {col}")
        #             window = Window(col, row, block_size, block_size)
        #             crop = src.read(window=window)
        #
        #             # 如果crop中全是有nodata值，就不保存
        #
        #             arr_dim = len(crop.shape)
        #             if arr_dim == 2:
        #                 if np.all(crop == nodatavalue):
        #                     print(f"crop: all nodatavalue")
        #                     continue
        #             elif arr_dim == 3:
        #                 if np.all(crop[0] == nodatavalue):
        #                     print(f"crop: all nodatavalue")
        #                     continue
        #
        #             # 生成裁剪后文件的路径
        #
        #             filename = f"crop_{row}_{col}.tif"
        #             filename = os.path.join(save_dir, filename)
        #
        #             # 将裁剪后的数据写入输出影像
        #             profile = src.profile
        #             profile['width'], profile['height'] = block_size, block_size
        #             profile['transform'] = rasterio.windows.transform(window, src.transform)
        #             with rasterio.open(filename, 'w', **profile) as dst:
        #                 dst.write(crop)
        # print("裁剪完成！")

    # 计算

    def cluster(self, cluster_number, if_add_position_encoding=True, method='kmeans', select_bands=None,
                filter_function=None):
        """
        Cluster the image by different method

        :param cluster_number: int   The number of cluster
        :param if_add_position_encoding: bool   If add position encoding
        :param method: str   The method of cluster
        :param select_bands: list   The bands to select
        :param filter_function: function   The function to filter the data
        """
        # 根据select_bands选择波段
        if select_bands is None:
            select_array = self.ds.ReadAsArray()
            nodata_value = self.nodatavalue
        else:
            select_ds = self.select_bands(select_bands)
            select_array = select_ds.ds.ReadAsArray()
            nodata_value = select_ds.nodatavalue

        out_image = select_array

        # print(f"out_image.shape: {out_image.shape}")

        def field_filter(df, nodatavalue):
            df = df[~df['B1'].isin([nodatavalue])]
            return df

        def cluster_preprocess(array, if_add_position_encoding, select_bands: list = None):
            """
            在原来的基础上扩展两个波段，分别是x，y的行列号
            """
            if select_bands is not None:
                if self.BANDS == 1:
                    return array
                else:
                    array = array[select_bands, :, :]

            # 重新计算arr的shape
            dim = len(array.shape)
            if dim != 2 and dim != 3:
                raise Exception(f"Field array dimemsion: {dim} not supported")
            if dim == 2:
                BANDS = 1
            else:
                BANDS = array.shape[0]

            # 在原来的基础上扩展两个波段，分别是x，y
            if if_add_position_encoding:
                extended_array = np.zeros((BANDS + 2, self.HEIGHT, self.WIDTH), dtype=array.dtype)
                x, y = np.meshgrid(np.arange(self.WIDTH), np.arange(self.HEIGHT))
                # 将x y 旋转
                # x = x.transpose(1, 0)
                # y = y.transpose(1, 0)
                extended_array[:BANDS, :, :] = array
                extended_array[BANDS, :, :] = x
                extended_array[BANDS + 1, :, :] = y
                return extended_array, BANDS
            else:
                return array, BANDS

        def builtin_cluster(df, cluster_number, method='kmeans'):
            """
            对df进行聚类，返回聚类结果，并根据聚类结果对等级进行排序，分为1到n级
            """
            if method == "kmeans":
                from sklearn.cluster import KMeans
                clustering = KMeans(n_clusters=cluster_number).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "dbscan":
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.3, min_samples=10).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "meanshift":
                from sklearn.cluster import MeanShift
                clustering = MeanShift().fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "spectral":
                from sklearn.cluster import SpectralClustering
                clustering = SpectralClustering(n_clusters=cluster_number).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=cluster_number).fit(df)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
            elif method == "gaussian":
                from sklearn.mixture import GaussianMixture
                clustering = GaussianMixture(n_components=cluster_number).fit(df)
                labels = clustering.predict(df)
                centers = clustering.cluster_centers_
            elif method == "hierarchical":
                from sklearn.cluster import Birch
                clustering = Birch(n_clusters=cluster_number).fit(df)
                labels = clustering.predict(df)
                centers = clustering.cluster_centers_
            else:
                raise Exception(f"method: {method} not supported")

            # print(f"centers: {centers}")
            # 计算每个center对应的ndvi
            ndvi_list = []

            def cal_ndvi(red, nir):
                return (nir - red) / (nir + red)

            for center in centers:
                ndvi = cal_ndvi(center[2], center[3])
                ndvi_list.append(ndvi)
            # print(f"ndvi_list: {ndvi_list}")

            # 根据聚类中心的ndvi值对等级进行排序
            ndvi_list = np.array(ndvi_list)
            sort_index = np.argsort(ndvi_list)
            labels_dict = {sort_index[i]: i for i in range(len(sort_index))}
            labels = [labels_dict[i] + 1 for i in labels]
            # sort_index = np.argsort(ndvi_list)
            # # print(f"sort_index: {sort_index}")
            # labels_dict = {}
            # for i in range(len(sort_index)):
            #     labels_dict[sort_index[i]] = i
            #
            # labels = [labels_dict[i] + 1 for i in labels]
            # # print(f"labels: {labels}")
            df["label"] = labels
            return df

        array, BANDS = cluster_preprocess(out_image, if_add_position_encoding=if_add_position_encoding)
        reshape_array = array.transpose(2, 1, 0)
        reshape_array = reshape_array.reshape(
            -1, reshape_array.shape[-1], order='C')

        if if_add_position_encoding:
            columns = [f"B{i+1}" for i in range(BANDS)] + ["x", "y"]
        else:
            columns = [f"B{i+1}" for i in range(BANDS)]
        rawdf = pd.DataFrame(reshape_array)
        rawdf.columns = columns

        # print(f"rawdf: {rawdf}")

        # 过滤掉nodatavalue
        filter_df = field_filter(rawdf, nodata_value)
        if filter_function is not None:
            filter_df = filter_function(filter_df)

        if len(filter_df) != 0 and len(rawdf) == 0:
            raise Exception("can't be use cluster, because all data in field is cloud")

        # -----------------聚类-----------------
        clustered_df = builtin_cluster(filter_df, cluster_number)
        cluster_label = np.full(self.WIDTH * self.HEIGHT, -1)  # 创建一个填充了 -1 的数组

        indices = clustered_df.index.values
        mask = np.isin(np.arange(self.WIDTH * self.HEIGHT), indices)  # 创建一个布尔掩码来检查索引是否存在于 clustered_df 中
        cluster_label[mask] = clustered_df.loc[indices, "label"].values  # 使用布尔掩码来更新 cluster_label

        save_array = cluster_label.reshape(self.WIDTH, self.HEIGHT)
        save_array = save_array.transpose(1, 0)

        return RSImg.from_array(save_array, nodatavalue=-1, projection=self.projection, geoTransform=self.geoTransform)



class Sentinel2RSImg(RSImg):
    def __init__(self, ds: gdalDataset, *args, **kwargs) -> None:
        super().__init__(ds)

    @classmethod
    def from_tif(cls, tif_path: str, *args, **kwargs):
        return super().from_tif(tif_path, *args, **kwargs)

    @classmethod
    def from_array(cls, array, nodatavalue, projection, geoTransform, *args, **kwargs):
        return super().from_array(array, nodatavalue, projection, geoTransform, *args, **kwargs)

    def renderRGB(self):
        def normalizedArray2RGB(array):
            normalizedArray = np.clip(array, 0, 3000) / 3000 * 255
            return normalizedArray.astype(np.uint8)

        rgbarray = normalizedArray2RGB(self.array[[3, 2, 1]])
        return RSImg.from_array(rgbarray, nodatavalue=self.nodatavalue, projection=self.projection,
                                geoTransform=self.geoTransform)

    def cluster(self, cluster_number, if_add_position_encoding=True, method='kmeans', select_bands=[1, 2, 3, 7, -1],
                filter_function=None):
        def sentinel2_filter(df):
            # 删除cloud_mask为2，4，删除B2 == 0
            df = df[~df['B5'].isin([2, 4])]
            df = df[~df['B2'].isin([0])]
            return df

        return super().cluster(cluster_number, if_add_position_encoding, method, select_bands=select_bands
                               , filter_function=sentinel2_filter)

    def get_cloudage(self):
        """
        1为水体，2为云阴影，3为雪，4为云，5为土地，0为空值。
        """
        cloudDs = self.ds
        mask_array = cloudDs.GetRasterBand(13).ReadAsArray().reshape(-1)
        cloudPixelNum = np.sum(mask_array != 0)
        cloudRate = np.sum(mask_array == 4)
        cloudShadowRate = np.sum(mask_array == 2)
        cloudage = (cloudRate + cloudShadowRate) / cloudPixelNum
        return cloudage
