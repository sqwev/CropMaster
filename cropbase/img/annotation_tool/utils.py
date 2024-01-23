# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.

from skimage import measure
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util

def singleMask2rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons



import fiona
import fiona.crs
from shapely.geometry import mapping, Point
# 将含有经纬度、灾损率的csv文件转换为shp文件

def clean_df(df:pd.DataFrame, shp_save_path:str):

    df = df.dropna()
    df = df.reset_index(drop=True)

    if 'lon' in df.columns:
        df = df.rename(columns={'lon': 'longitude'})
    if 'lat' in df.columns:
        df = df.rename(columns={'lat': 'latitude'})


    # 含有lon, lat, 或 longitude, latitude的列名
    # 导出为shp文件
    # Define the schema
    properties = {}
    for col in df.columns:
        type_col = df[col].dtype
        if type_col == np.float64:
            type_col = 'float'
        elif type_col == np.int64:
            type_col = 'int'
        else:
            type_col = 'str'
        properties[col] = type_col
    schema = {
        'geometry': 'Point',
        'properties': properties
    }

    # Export the DataFrame to a shapefile
    with fiona.open(shp_save_path, 'w', driver='ESRI Shapefile', crs=fiona.crs.from_epsg(4490), schema=schema) as c:
        for _, row in df.iterrows():
            point = Point(float(row['longitude']), float(row['latitude']))
            properties = {}
            for col in df.columns:
                properties[col] = row[col]
            c.write({
                'geometry': mapping(point),
                'properties': properties,
            })


    print("shp file saved to " + shp_save_path)
    return df












