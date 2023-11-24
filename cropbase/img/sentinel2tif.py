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


class Sentinel2Tif:
    photo_time = None
    def __init__(self, path, level):
        self.path = path
        self.level = level
        self.ds = gdal.Open(self.path)
        # 六参数模型
        self.geoTransform = self.ds.GetGeoTransform()
        # 投影信息投影参考系
        self.projection = self.ds.GetProjection()
        # 判断tif是地理坐标系还是投影坐标系
        # Judge whether tif is Geographic coordinate system or projection coordinate system
        if self.projection.startswith("PROJCS"):
            self.projType = "pro"
        elif self.projection.startswith("GEOGCS"):
            self.projType = "geo"
        else:
            self.projType = None

        self.x_min, self.x_res, self.rotation, self.y_min, self.rotation, self.y_res = self.geoTransform
        self.BANDS = self.ds.RasterCount
        self.WIDTH = self.ds.RasterXSize
        self.HEIGHT = self.ds.RasterYSize

        self.nodatavlaue = self.ds.GetRasterBand(1).GetNoDataValue()

        if self.BANDS == 1:
            self.valid_mask = np.where(np.isnan(self.ds.ReadAsArray()), 0, 1)
        else:
            template = self.ds.GetRasterBand(1).ReadAsArray()
            self.valid_mask = np.where(np.isnan(template), 0, 1)

    def get_espg(self):
        """
        return: epsg
        """
        with rasterio.open(self.path) as src:
            return src.crs.to_epsg()


    def renderRGB(self, save_path):
        raise NotImplementedError