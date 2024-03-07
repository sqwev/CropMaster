# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import os
import uuid
import datetime
import numpy as np
import rasterio
import rasterio.mask
import fiona
import affine
import shapely
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
from collections import UserDict

class RestrictedDict(UserDict):
    def __init__(self, allowed_type):
        self.allowed_type = allowed_type
        super().__init__()

    def __setitem__(self, key, value):
        if isinstance(value, self.allowed_type):
            self.data[key] = value
        else:
            raise ValueError(f"Value must be of type {self.allowed_type}")




def calculate_utm_zone(longitude):
    """
    Use the given longitude to calculate the UTM zone.
    longitude: 经度，范围从-180到180
    返回: UTM区域的编号，范围从1到60
    """
    assert -180 <= longitude <= 180, "Longitude out of range"
    return int((longitude + 180) / 6) + 1


def latlon_to_utm(longitude, latitude) -> int:
    """
    Automatically determine the UTM projection zone based on the given longitude and latitude.

    Args:
        longitude:
        latitude:

    Returns:
        EPSG code of the UTM projection

    """
    # 自动确定UTM投影带
    utm_zone = calculate_utm_zone(longitude)
    # 判断是南半球还是北半球
    hemisphere = 'north' if latitude >= 0 else 'south'
    # 构建EPSG代码。南半球的代码是32600+区域代码，北半球是32700+区域代码
    epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
    return epsg_code


def auto_reco_crs(projection):
    """
    Automatically recognize the projection type and return the corresponding CRS object.

    Args:
        projection: the projection of the image, can be a string, an integer or a rasterio.crs.CRS object

    Returns:
        rasterio.crs.CRS object

    """
    if isinstance(projection, int):
        return rasterio.crs.CRS.from_epsg(projection)
    elif isinstance(projection, str):
        return rasterio.crs.CRS.from_string(projection)
    elif isinstance(projection, rasterio.crs.CRS):
        return projection
    else:
        raise ValueError(f"Unsupported projection type: {type(projection)}")


def auto_reco_transform(geoTransform):
    """
    Automatically recognize the geoTransform type and return the corresponding affine.Affine object.

    Args:
        geoTransform: the geoTransform of the image, can be a tuple or an affine.Affine object

    Returns:
        affine.Affine object

    """
    if isinstance(geoTransform, tuple) and len(geoTransform) == 6:
        return rasterio.transform.from_gcps(geoTransform)
    elif isinstance(geoTransform, affine.Affine):
        return geoTransform
    else:
        raise ValueError(f"Unsupported geoTransform type: {type(geoTransform)}")


def _is_value_in_dtype_range(value, dtype):
    """
    Check if the value is in the range of the given dtype.

    Args:
        value:
        dtype:

    Returns:
        Ture or False
    """
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


def _get_CHW_array_shape(arr: np.ndarray):
    """
    Get the shape of the array in the order of (channels, height, width)

    Args:
        arr: input array

    Returns:
        (height, width, channels)
    """
    if arr.ndim == 2:
        channels = 1
        height, width = arr.shape
    elif arr.ndim == 3:
        channels, height, width = arr.shape
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return height, width, channels


def _get_HWC_array_shape(arr: np.ndarray):
    """
    Get the shape of the array in the order of (height, width, channels)

    Args:
        arr: the input array

    Returns:
        (height, width, channels)
    """
    if arr.ndim == 2:
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:
        height, width, channels = arr.shape
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return height, width, channels

def merge_rsimg(rsimg_list):
    """
    Merge the RsImg object list into one RsImg object

    Args:
        rsimg_list: list   The list of the RsImg object

    Returns:
        RsImg object
    """
    datasets_to_merge = [rsimg.ds for rsimg in rsimg_list]
    merged_dataset_array, merged_transform = merge(datasets_to_merge)
    return RsImg.from_array(merged_dataset_array, nodatavalue=rsimg_list[0].nodatavalue,
                            projection=rsimg_list[0].projection, geoTransform=merged_transform)



class RsImg:
    """
    RsImg is a package for rasterio, which is used to process remote sensing images in agriculture scene,
    including UAV images and satellite images.

    Args:
        ds: rasterio.io.DatasetReader   The rasterio dataset object
        name: str   The name of the image
        date: str   The date of the image
        *args:   The other arguments
        **kwargs:   The other keyword arguments
    """

    def __init__(self, ds, *args, **kwargs) -> None:
        self.ds = ds
        self.args = args
        self.kwargs = kwargs
        self.name = kwargs.get("name", uuid.uuid1())
        self.date = kwargs.get("date", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.projection = ds.crs
        self.geoTransform = ds.transform
        self.nodatavalue = ds.nodata
        self.height = ds.height
        self.width = ds.width
        self.bands = ds.count
        self.shape = ds.shape
        self.dim = len(ds.shape)

    def print_attributes(self):
        """
        Print all the attributes of the image
        Returns:

        """
        # 使用 vars() 函数打印所有属性
        info = vars(self)
        for key, value in info.items():
            print(f"{key}: {value}")

    @classmethod
    def from_tif(cls, tif_path, *args, **kwargs):
        ds = rasterio.open(tif_path)
        return cls(ds=ds, *args, **kwargs)

    @classmethod
    def from_array(cls, array: np.ndarray, nodatavalue, projection, geoTransform, *args, **kwargs):
        # check
        assert len(array.shape) == 2 or len(array.shape) == 3, "array shape must be 2 or 3, but got {}".format(
            array.shape)
        # nodatavalue can't over the range of dtype
        array_dtype = array.dtype
        assert _is_value_in_dtype_range(nodatavalue, array_dtype), "nodatavalue can't over the range of dtype"

        projection = auto_reco_crs(projection)
        _geoTransform = auto_reco_transform(geoTransform)

        height, width, channels = _get_CHW_array_shape(array)
        # 创建数据集配置字典
        dataset_meta = {
            'driver': 'GTiff',
            'count': channels,
            'dtype': array_dtype,
            'width': width,
            'height': height,
            'nodata': nodatavalue,
            'transform': _geoTransform,
            'crs': projection
        }
        # print(dataset_meta)
        if channels == 1:
            array = array[np.newaxis, ...]

        with MemoryFile() as memfile:
            dataset = memfile.open(**dataset_meta)
            dataset.write(array)

            return cls(ds=dataset, *args, **kwargs)

    def to_tif(self, tif_path, compress="lzw"):
        """
        Save the image to tif file

        Args:
            tif_path: The path to save the tif file
            compress: if use compress, default is "lzw"

        Returns:

        """
        if compress is not None:
            self.ds.meta.update({
                'compress': compress,
            })
        with rasterio.open(tif_path, 'w', **self.ds.meta) as dst:
            dst.write(self.ds.read())

    @property
    def array(self):
        return self.ds.read()

    def __del__(self):
        self.ds.close()
        del self.ds

    @property
    def valid_mask(self):
        return self.ds.read_masks()

    @property
    def border(self):
        return self.ds.bounds

    def set_name(self, name: str):
        """
        Set the name of the image

        Args:
            name: The name of the image

        Returns:

        """
        self.name = name

    def crop(self, left, top, right, bottom):
        """
        Crop the image by pixel serial number
        Args:
            left: The left pixel serial number
            top: The top pixel serial number
            right: The right pixel serial number
            bottom: The bottom pixel serial number

        Returns:
            RsImg object
        """
        array = self.ds.read(window=((top, bottom), (left, right)))
        nodatavalue = self.nodatavalue
        projection = self.projection
        # 计算偏移量
        xoff, yoff = self.geoTransform * (left, top)
        new_transform = rasterio.transform.from_origin(xoff, yoff, self.geoTransform.a, -self.geoTransform.e)
        return RsImg.from_array(array, nodatavalue=nodatavalue, projection=projection, geoTransform=new_transform)

    def select_bands(self, band_list):
        """
        Select the bands of the image

        :param band_list: list   The list of the bands
        """
        array = self.ds.read(band_list)
        return RsImg.from_array(array, nodatavalue=self.nodatavalue, projection=self.projection,
                                geoTransform=self.geoTransform)

    def sliding_window_crop(self, block_size: int, repetition_rate: float, save_dir=None):
        """
        Sliding window crop the image

        :param save_dir: str   The path to save the crop image
        :param block_size: int   The size of the crop image
        :param repetition_rate: float   The repetition rate of the crop image
        :param nodatavalue:    The nodatavalue of the image
        """
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "sliding_window_crop")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        assert 0 < repetition_rate < 1, "repetition_rate should be in (0, 1)"
        # 计算重复率对应的像素数
        step_size = int(block_size * (1 - repetition_rate))

        y_size = self.height
        x_size = self.width
        for row in range(0, y_size - block_size + 1, step_size):
            for col in range(0, x_size - block_size + 1, step_size):
                block = self.crop(col, row, col + block_size, row + block_size)
                block.to_tif(os.path.join(save_dir, f"{self.name}_{row}_{col}.tif"))

    def cut_by_geometry(self, geometry):
        """
        Cut the image by geometry

        :param geometry:    The geometry of the cut
        """
        if isinstance(geometry, shapely.geometry.base.BaseGeometry):
            geometry = [geometry]
        else:
            pass

        out_image, out_transform = rasterio.mask.mask(self.ds, geometry, crop=True)
        height, width, channels = _get_CHW_array_shape(out_image)
        out_meta = self.ds.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": height,
                         "width": width,
                         "transform": out_transform})
        return self.from_array(out_image, nodatavalue=self.nodatavalue, projection=self.projection,
                                geoTransform=out_transform)

    def cut_by_shp(self, shp_path):
        """
        Cut the image by shapefile

        :param shp_path:    The path of the shapefile
        """
        with fiona.open(shp_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            return self.cut_by_geometry(shapes)

    def plot(self):
        """
        Plot the image
        """
        raise NotImplementedError("plot method is not implemented")

    def reproject(self, dst_crs_espg: int):
        """
        Reproject the image

        :param dst_crs:    The destination crs
        """
        src = self.ds
        dst_crs = f"EPSG:{dst_crs_espg}"
        # 执行重投影
        dst_array, dst_transform = reproject(
            source=src.read(),  # 输入数组
            src_crs=src.crs,
            src_transform=src.transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)  # 使用最近邻插值重采样

        return RsImg.from_array(dst_array, nodatavalue=self.nodatavalue,
                                projection=dst_crs_espg, geoTransform=dst_transform)

    def auto_reproject_to_utm(self):
        """
        Project the image to another projection

        :param dst_crs:    The destination crs
        """
        assert not self.projection.is_projected, "The CRS is not projected"
        dst_crs_espg = latlon_to_utm(self.border[0], self.border[1])
        print(dst_crs_espg)
        return self.reproject(dst_crs_espg)

    from .cluster import ImgCluster
    def cluster(self, cluster: ImgCluster):
        cluster_mask = cluster(self.array)
        return RsImg.from_array(cluster_mask,
                                nodatavalue=-1,
                                projection=self.projection,
                                geoTransform=self.geoTransform)

    def resample(self, scale_factor):
        """
        Resample the image

        :param scale_factor:    The scale factor
        """
        raise NotImplementedError("resample method is not implemented")


class Sentinel2RsImg(RsImg):
    def __init__(self, ds, *args, **kwargs) -> None:
        super().__init__(ds, *args, **kwargs)

    @classmethod
    def from_tif(cls, tif_path, *args, **kwargs):
        ds = rasterio.open(tif_path)
        return cls(ds=ds, *args, **kwargs)

    @classmethod
    def from_array(cls, array: np.ndarray, nodatavalue, projection, geoTransform, *args, **kwargs):
        return super().from_array(array, nodatavalue, projection, geoTransform, *args, **kwargs)

    def renderRGB(self):
        def normalizedArray2RGB(array):
            normalizedArray = np.clip(array, 0, 3000) / 3000 * 255
            return normalizedArray.astype(np.uint8)

        rgbarray = normalizedArray2RGB(self.array[[3, 2, 1]])
        return RsImg.from_array(rgbarray, nodatavalue=self.nodatavalue, projection=self.projection,
                                geoTransform=self.geoTransform)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.renderRGB().array.transpose(1, 2, 0))
        plt.show()
