# encoding: utf-8

"""
@author:Shenzhou Liu
@contact:913737515@qq.com
"""
import os
import time
import scipy
import glob
import psutil
import platform
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
import concurrent.futures
from datetime import date, datetime, timedelta
from osgeo import gdal, osr, ogr
from tqdm import tqdm, trange
from pathlib import Path

from .sentinel2safe import Sentinel2SAFE, Sentinel2SAFEL1C, Sentinel2SAFEL2A


class Sentinel2Tile():
    def __init__(self, L1CPath, L2APath):
        if L1CPath is not None and L2APath is not None:
            self.intact = 1
            self.L1CPath = L1CPath
            self.L2APath = L2APath

    def getTileName(self):
        # Name the entire combination with the name l1c
        # self.L2Ads = Sentinel2SAFEL2A(self.L2APath)
        self.L1Cds = Sentinel2SAFEL1C(self.L1CPath)
        self.name = self.L1Cds.date + "_" + self.L1Cds.tile + ".tif"
        return self.name

    def generateImg(self, savePath, resolution=10, bands_needed=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], cloudMask=True,
                    isBuildOverviews=False):
        """
        生成影像，其中，1-12波段分别为B1-B12，13波段为Fmask生成的云掩膜
        """
        NODATA_VALUE = 0

        # Approximately 10GB of memory space must be guaranteed
        mem = psutil.virtual_memory()
        total_system_memory = float(mem.total) / 1024 / 1024 / 1024
        already_used_memory = float(mem.used) / 1024 / 1024 / 1024
        free_memory = float(mem.available) / 1024 / 1024 / 1024
        if free_memory < 10:
            raise Exception(f"Insufficient memory to process, need at least 10GB/{free_memory}GB of memory space!")

        if not cloudMask:
            L2Ads = Sentinel2SAFEL2A(self.L2APath)
            L2Ads.tran2tif(savePath, resolution, bands_needed, isBuildOverviews)
        elif cloudMask:
            # 先读取云掩膜，resize后
            L1Cds = Sentinel2SAFEL1C(self.L1CPath)
            cloudmakspath = L1Cds.fmaskTifPath
            # Fmask的去云结果中，0为土地像素，1为水体，2为云阴影，3为雪，4为云，255为空值。
            # 云掩膜的大小为5490*5490，分辨率为20m
            # 使用gdal读取JPEG2000格式遥感影像
            mask_array = gdal.Open(cloudmakspath).ReadAsArray().astype(np.uint16)
            mask_array[mask_array == 0] = 5  # 将土地像素的值更改为5
            mask_array[mask_array == 255] = NODATA_VALUE  # 统一NoDataValue
            # 双线性插值将是order = 1，最临近插值的是order = 0，立方体是默认值order = 3。放缩云掩膜的时候使用最临近插值
            if resolution == 10:
                resample_mask_array = scipy.ndimage.zoom(mask_array, 2, order=0)
            elif resolution == 20:
                resample_mask_array = mask_array
            elif resolution == 60:
                resample_mask_array = scipy.ndimage.zoom(mask_array, 1 / 3, order=0)
            else:
                raise Exception("Wrong resolution! Please choose 10, 20 or 60.")

            L2Adir = os.listdir(os.path.join(self.L2APath, "GRANULE"))[0]
            img_data_dir = os.path.join(self.L2APath, "GRANULE", L2Adir, "IMG_DATA")
            r10m_dir = os.path.join(img_data_dir, "R10m")
            r20m_dir = os.path.join(img_data_dir, "R20m")
            r60m_dir = os.path.join(img_data_dir, "R60m")
            # different band path
            b1 = glob.glob(os.path.join(r60m_dir, "*_B01_60m.jp2"))[0]
            b2 = glob.glob(os.path.join(r10m_dir, "*_B02_10m.jp2"))[0]
            b3 = glob.glob(os.path.join(r10m_dir, "*_B03_10m.jp2"))[0]
            b4 = glob.glob(os.path.join(r10m_dir, "*_B04_10m.jp2"))[0]
            b5 = glob.glob(os.path.join(r20m_dir, "*_B05_20m.jp2"))[0]
            b6 = glob.glob(os.path.join(r20m_dir, "*_B06_20m.jp2"))[0]
            b7 = glob.glob(os.path.join(r20m_dir, "*_B07_20m.jp2"))[0]
            b8a = glob.glob(os.path.join(r10m_dir, "*_B08_10m.jp2"))[0]
            b8b = glob.glob(os.path.join(r20m_dir, "*_B8A_20m.jp2"))[0]
            b9 = glob.glob(os.path.join(r60m_dir, "*_B09_60m.jp2"))[0]
            b11 = glob.glob(os.path.join(r20m_dir, "*_B11_20m.jp2"))[0]
            b12 = glob.glob(os.path.join(r20m_dir, "*_B12_20m.jp2"))[0]
            allBands = [(b1, "b1"), (b2, "b2"), (b3, "b3"), (b4, "b4"),
                        (b5, "b5"), (b6, "b6"), (b7, "b7"), (b8a, "b8a"),
                        (b8b, "b8b"), (b9, "b9"), (b11, "b11"), (b12, "b12")]
            bands_path_list = []
            for pixelClass in bands_needed:
                bands_path_list.append(allBands[pixelClass - 1])
            bands_num = len(bands_path_list) + 1
            # 获取像素大小和GeoTransform、投影信息
            b2 = str(b2).replace("\\\\", "/")
            DataSet10 = gdal.Open(b2)
            geoTransform = DataSet10.GetGeoTransform()
            proj = DataSet10.GetProjection()
            cols10 = DataSet10.RasterXSize
            rows10 = DataSet10.RasterYSize
            del DataSet10
            DataSet20 = gdal.Open(b5)
            cols20 = DataSet20.RasterXSize
            rows20 = DataSet20.RasterYSize
            del DataSet20
            DataSet60 = gdal.Open(b1)
            cols60 = DataSet60.RasterXSize
            rows60 = DataSet60.RasterYSize
            del DataSet60

            order = 3
            # 双线性插值将是order = 1，最临近插值的是order = 0，立方体是默认值order = 3

            if resolution == 10:
                rows = rows10
                cols = cols10
                re_img = np.zeros((bands_num, rows, cols), np.uint16)
                for i, (imgpath, sign) in enumerate(bands_path_list):
                    if sign in ["b1", "b9", "b10"]:
                        mat = gdal.Open(imgpath).ReadAsArray()
                        re_img[i] = scipy.ndimage.zoom(mat, 6, order=order)
                    elif sign in ["b2", "b3", "b4", "b8a"]:
                        re_img[i] = gdal.Open(imgpath).ReadAsArray()
                    elif sign in ["b5", "b6", "b7", "b8b", "b11", "b12"]:
                        mat = gdal.Open(imgpath).ReadAsArray()
                        re_img[i] = scipy.ndimage.zoom(mat, 2, order=order)

                re_img[-1] = resample_mask_array
            # elif resolution == 20:
            #     rows = rows20
            #     cols = cols20
            #     re_img = np.zeros((bands_num, rows, cols), np.uint16)
            #     for i, (imgpath, sign) in enumerate(bands_path_list):
            #         print(f"Start processing Band {sign}")
            #         if sign in ["b1", "b9", "b10"]:
            #             mat = gdal.Open(imgpath).ReadAsArray()
            #             re_img[i] = scipy.ndimage.zoom(mat, 3, order=order)
            #         elif sign in ["b2", "b3", "b4", "b8a"]:
            #             mat = gdal.Open(imgpath).ReadAsArray()
            #             re_img[i] = scipy.ndimage.zoom(mat, 0.5, order=order)
            #         elif sign in ["b5", "b6", "b7", "b8b", "b11", "b12"]:
            #             re_img[i] = gdal.Open(imgpath).ReadAsArray()
            # elif resolution == 60:
            #     rows = rows60
            #     cols = cols60
            #     re_img = np.zeros((bands_num, rows, cols), np.uint16)
            #     for i, (imgpath, sign) in enumerate(bands_path_list):
            #         print(f"Start processing Band {sign}")
            #         if sign in ["b1", "b9", "b10"]:
            #             re_img[i] = gdal.Open(imgpath).ReadAsArray()
            #         elif sign in ["b2", "b3", "b4", "b8a"]:
            #             mat = gdal.Open(imgpath).ReadAsArray()
            #             re_img[i] = scipy.ndimage.zoom(mat, 1 / 6, order=order)
            #         elif sign in ["b5", "b6", "b7", "b8b", "b11", "b12"]:
            #             mat = gdal.Open(imgpath).ReadAsArray()
            #             re_img[i] = scipy.ndimage.zoom(mat, 1 / 3, order=order)

            driver = gdal.GetDriverByName("GTiff")
            # dataset = driver.Create(savePath, cols, rows, bands_num, gdal.GDT_UInt16,
            #                         options=['COMPRESS=LZW', 'BIGTIFF=YES'])
            dataset = driver.Create(savePath, cols, rows, bands_num, gdal.GDT_UInt16)
            dataset.SetGeoTransform(geoTransform)  # 写入仿射变换参数
            dataset.SetProjection(proj)  # 写入投影
            for pixelClass in range(bands_num):
                dataset.GetRasterBand(pixelClass + 1).WriteArray(re_img[pixelClass, :, :])
                dataset.GetRasterBand(pixelClass + 1).SetNoDataValue(NODATA_VALUE)
            if isBuildOverviews:
                dataset.BuildOverviews('average', [2, 4, 8, 16, 32])
            else:
                pass
            del dataset
            return savePath