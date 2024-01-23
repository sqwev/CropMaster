# encoding: utf-8

"""
@author:Shenzhou Liu
@contact:913737515@qq.com
"""
import os
import time
import scipy
import glob
import platform
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
import concurrent.futures
from datetime import date, datetime, timedelta
from osgeo import gdal, osr, ogr
from tqdm import tqdm, trange
# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from pathlib import Path

# private module
from ..utils import execute_system_command as executeSystemCommand


class Sentinel2SAFE:
    def __init__(self, path):
        self.path = path
        sys = platform.system()
        self.systemType = sys  # "Windows" or "Linux"
        if self.systemType == "Windows":
            self.__slash = '\\'
        elif self.systemType == "Linux":
            self.__slash = '/'
        self.name = path.split(self.__slash)[-1]
        self.__nameList = self.name.split('_')
        self.level = self.__nameList[1][3:]
        self.date = self.__nameList[2][0:8]
        self.secondDate = self.__nameList[6].split(".")[0]
        self.tile = self.__nameList[5]
        self.tifname = self.date + "_" + self.tile + ".tif"


class Sentinel2SAFEL1C(Sentinel2SAFE):
    def __init__(self, path):
        Sentinel2SAFE.__init__(self, path)
        GRANULEPath = os.path.join(path, "GRANULE")
        # ***.SAFE/GRANULE/L1C***这个文件夹
        self.l1cPath = os.path.join(GRANULEPath, os.listdir(GRANULEPath)[0])
        # Fmask生成的云掩膜所在的文件夹位置
        fmaskPath = os.path.join(self.l1cPath, "FMASK_DATA")
        if os.path.exists(fmaskPath):
            self.isFmask = 1
            self.fmaskTifPath = os.path.join(fmaskPath, os.listdir(fmaskPath)[0])
        else:
            self.isFmask = 0
            self.fmaskTifPath = None

    def fmask(self):
        """
        使用Fmask4_6算法对Sentinel-2的L1C级数据进行去云
        :return:成功返回1不成功返回0
        """
        if self.isFmask:
            print(f"{self.name} has been fmasked.")
        else:
            if self.systemType == "Windows":
                current_dir = os.getcwd()
                os.chdir(self.l1cPath)
                command = "Fmask_4_6"
                args = [r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe", command]  # run command in Powershell
                result = executeSystemCommand(command=args, max_tries=2)
                os.chdir(current_dir)
                return result
            elif self.systemType == "Linux":
                current_dir = os.getcwd()
                os.chdir(self.l1cPath)
                command = "/usr/GERS/Fmask_4_6/application/Fmask_4_6"  # 这条指令要根据安装的Fmask实际情况更改
                result = executeSystemCommand(command=command, max_tries=2)
                os.chdir(current_dir)
                return result
            else:
                raise Exception("System type error! Please use Windows or Linux.")

    def statisticsCloudage(self):
        # Fmask的去云结果中，0为土地像素，1为水体，2为云阴影，3为雪，4为云，255为空值。
        # 云掩膜的大小为5490*5490，分辨率为20m
        if not self.fmaskTifPath:
            return 0
        cloudDs = gdal.Open(self.fmaskTifPath)
        type = [0, 1, 3]  # 保留的像素种类设置为0，1，3
        # 使用gdal读取JPEG2000格式遥感影像
        mask_array = cloudDs.ReadAsArray().reshape(-1)
        cloudPixelNum = np.sum(mask_array != 255)
        cloudRate = np.sum(mask_array == 4)
        cloudShadowRate = np.sum(mask_array == 2)
        cloudage = (cloudRate + cloudShadowRate) / cloudPixelNum
        return cloudage


class Sentinel2SAFEL2A(Sentinel2SAFE):
    def __init__(self, path):
        super().__init__(path)

    def tran2tif(self, savePath:str, resolution:int=10,
                 bands_needed:list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], isBuildOverviews:bool=True):
        if self.level != "L2A":
            raise Exception("The level of the SAFE is not L2A!")
        if resolution not in [10, 20, 60]:
            raise Exception("The resolution is not supported! Please choose 10, 20 or 60.")



        NoData_value = 0
        safedir = self.path
        L2Adir = os.listdir(os.path.join(safedir, "GRANULE"))[0]
        img_data_dir = os.path.join(safedir, "GRANULE", L2Adir, "IMG_DATA")
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
        for i in bands_needed:
            bands_path_list.append(allBands[i - 1])
        bands_num = len(bands_path_list)
        # 获取像素大小和GeoTransform、投影信息
        b2 = str(b2).replace("\\\\", "/")
        DataSet10 = gdal.Open(b2)
        geoTransform = DataSet10.GetGeoTransform()
        proj = DataSet10.GetProjection()
        cols10 = DataSet10.RasterXSize
        rows10 = DataSet10.RasterYSize
        # del DataSet10
        DataSet20 = gdal.Open(b5)
        cols20 = DataSet20.RasterXSize
        rows20 = DataSet20.RasterYSize
        del DataSet20
        DataSet60 = gdal.Open(b1)
        cols60 = DataSet60.RasterXSize
        rows60 = DataSet60.RasterYSize
        del DataSet60

        order = 3
        # 双线性插值将是order = 1，
        # 最临近插值的是order = 0，
        # 立方体是默认值（顺序= 3）。
        if resolution == 10:
            rows = rows10
            cols = cols10
            re_img = np.zeros((bands_num, rows, cols), np.uint16)
            for i, (imgpath, sign) in enumerate(bands_path_list):
                # print(f"Start processing Band {sign}")
                if sign in ["b1", "b9", "b10"]:
                    mat = gdal.Open(imgpath).ReadAsArray()
                    re_img[i] = scipy.ndimage.zoom(mat, 6, order=order)
                elif sign in ["b2", "b3", "b4", "b8a"]:
                    re_img[i] = gdal.Open(imgpath).ReadAsArray()
                elif sign in ["b5", "b6", "b7", "b8b", "b11", "b12"]:
                    mat = gdal.Open(imgpath).ReadAsArray()
                    re_img[i] = scipy.ndimage.zoom(mat, 2, order=order)
        elif resolution == 20:
            rows = rows20
            cols = cols20
            re_img = np.zeros((bands_num, rows, cols), np.uint16)
            for i, (imgpath, sign) in enumerate(bands_path_list):
                # print(f"Start processing Band {sign}")
                if sign in ["b1", "b9", "b10"]:
                    mat = gdal.Open(imgpath).ReadAsArray()
                    re_img[i] = scipy.ndimage.zoom(mat, 3, order=order)
                elif sign in ["b2", "b3", "b4", "b8a"]:
                    mat = gdal.Open(imgpath).ReadAsArray()
                    re_img[i] = scipy.ndimage.zoom(mat, 0.5, order=order)
                elif sign in ["b5", "b6", "b7", "b8b", "b11", "b12"]:
                    re_img[i] = gdal.Open(imgpath).ReadAsArray()
        elif resolution == 60:
            rows = rows60
            cols = cols60
            re_img = np.zeros((bands_num, rows, cols), np.uint16)
            for i, (imgpath, sign) in enumerate(bands_path_list):
                # print(f"Start processing Band {sign}")
                if sign in ["b1", "b9", "b10"]:
                    re_img[i] = gdal.Open(imgpath).ReadAsArray()
                elif sign in ["b2", "b3", "b4", "b8a"]:
                    mat = gdal.Open(imgpath).ReadAsArray()
                    re_img[i] = scipy.ndimage.zoom(mat, 1 / 6, order=order)
                elif sign in ["b5", "b6", "b7", "b8b", "b11", "b12"]:
                    mat = gdal.Open(imgpath).ReadAsArray()
                    re_img[i] = scipy.ndimage.zoom(mat, 1 / 3, order=order)

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(savePath, cols, rows,
                                bands_num, gdal.GDT_UInt16)
        dataset.SetGeoTransform(geoTransform)  # 写入仿射变换参数
        dataset.SetProjection(proj)  # 写入投影
        for i in range(bands_num):
            dataset.GetRasterBand(i + 1).WriteArray(re_img[i, :, :])
            dataset.GetRasterBand(i + 1).SetNoDataValue(NoData_value)
        if isBuildOverviews == 1:
            dataset.BuildOverviews('average', [2, 4, 8, 16, 32])
        else:
            pass
        del dataset