import os, sys
import time

import pandas as pd
from osgeo import gdal
dataset_path = r"D:\phd\CropMaster"
sys.path.append(dataset_path)

import cropbase as cb

def read_yield_df(dir):
    point_shp_name = 'measurement.shp'
    point_shp_path = os.path.join(dir, point_shp_name)
    point_shp = cb.CropDisasterSamplePoints(point_shp_path)
    lossratedf = point_shp.read_property()
    # 进行后处理
    yielddf = lossratedf[["lat", "lon", "lossrate"]]
    yielddf = yielddf.rename(columns={"lossrate": "yield"})
    yielddf.columns = ["latitude", "longitude", "cropyield"]

    def read_yield(tif_dir):
        # 得到此目录下的所有shp文件
        shp_list = [os.path.join(tif_dir, shp) for shp in os.listdir(tif_dir) if shp.endswith(".shp")]
        yielddf_list = []
        for shp in shp_list:
            # 判断shp格式,有3个_
            # print(shp)
            if len(os.path.basename(shp).split('_')) != 4:
                continue
            yielddf_list.append(cb.read_yield_from_shp(shp))
        yielddf = pd.concat(yielddf_list)
        yielddf.reset_index(drop=True, inplace=True)
        return yielddf

    myyielddf = read_yield(point_shp_dir)
    myyielddf = myyielddf.rename(columns={"lossrate": "cropyield", })
    myyielddf = myyielddf[["latitude", "longitude", "cropyield"]]
    # 合并两个
    yielddf = pd.concat([yielddf, myyielddf])
    yielddf.reset_index(drop=True, inplace=True)
    return yielddf



def read_disaster(dir):
    available_disaster_list = ["waterlogging", "flood", "drought", "lodge", "nodisaster"]
    shp_name = [disaster + ".shp" for disaster in available_disaster_list]

    for i in range(len(shp_name)):
        shp_name[i] = os.path.join(dir, shp_name[i])
        # 尝试读取shp文件
        if not os.path.exists(shp_name[i]):
            continue
        point_shp = cb.CropDisasterSamplePoints(point_shp_path)
        lossratedf = point_shp.read_property()
        # 进行后处理
        yielddf = lossratedf[["lat", "lon", "lossrate"]]
        yielddf = yielddf.rename(columns={"lossrate": "yield"})
        yielddf.columns = ["latitude", "longitude", "cropyield"]
