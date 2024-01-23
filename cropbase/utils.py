
import fiona
import rtree
import json
import os
import rasterio
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from shapely.geometry import Point, Polygon, shape, mapping



def find_points_in_which_polygon(df:pd.DataFrame, shp_path):
    """
    :param df: 一个dataframe，包含经纬度信息
    :param shp_path: 田块shp文件路径
    :return: result_dict：一个字典，key是田块id，value是一个pandas dataframe
    fieldid_hash_dict：一个字典，key是田块id，value是一个字典，包含田块的geom和properties

    """
    df_columns = df.columns.tolist()
    # Create an R-tree index and store the polygons in it
    idx = rtree.index.Index()    
    fieldid_hash_dict = {} # 这个是用于把field hash和field index匹配起来的一个字典
    with fiona.open(shp_path) as shp: # 建立起田块index与地理信息的关系，放进rtree里
        print(type(shp)) # fiona.collection.Collection'
        field_index = -1
        for i, feature in enumerate(shp):
            geom = shape(feature['geometry'])
            properties = feature['properties']
            if geom.geom_type == 'MultiPolygon':
                for polygon in geom:
                    field_index += 1
                    small_field = polygon
                    # small_field.bounds是一个含有四个元素的元组
                    fieldid_hash_dict[field_index] = {"geom":small_field,
                                                      "properties": properties}
                    idx.insert(field_index, small_field.bounds)

            elif geom.geom_type == "Polygon":
                field_index += 1
                small_field = geom
                fieldid_hash_dict[field_index] = {"geom":small_field,
                                                "properties": properties}
                idx.insert(field_index, geom.bounds)
            else:
                raise ValueError(f"geom type error: {geom.geom_type}")


    result_dict = {} # 建立一个字典，存储每个田块的id与其对应的经纬度点 
    # 查询每个点是否在某个polygon中
    for i in range(len(df)):
        lon= df.iloc[i]["longitude"]
        lat= df.iloc[i]["latitude"]
        point = Point(lon, lat)        
        # 使用索引进行查询o
        possible_points = list(idx.intersection(point.bounds))
        intersected_polygons = []
        for polygon_id in possible_points:
            if point.within(fieldid_hash_dict[polygon_id]["geom"]):
                intersected_polygons.append(polygon_id)
        # 此时intersected_polygons里存储的是这个点在哪些shp里
        
        # 将点添加到对应的polygon中
        if len(intersected_polygons) != 0:
            for polygon_id in intersected_polygons:
                if polygon_id not in result_dict.keys():
                    result_dict[polygon_id] = pd.DataFrame(columns=df_columns)
                
                result_dict[polygon_id] = result_dict[polygon_id].append(df.iloc[i], ignore_index=True)

        else:
            print(f"point: {point} is not in any polygon")
            
    return result_dict, fieldid_hash_dict



def find_points_in_which_polygon_v2(df:pd.DataFrame, shp_path):

    # check input
    # should have longitude, latitude, lossrate
    essential_columns = ["longitude", "latitude"]
    for column in essential_columns:
        if column not in df.columns:
            raise Exception(f"column: {column} not in df.columns: {df.columns}")
    # do not have nan
    if df.isnull().values.any():
        raise Exception(f"inputdf contains nan")
    # only select the essential columns
    # df = df[essential_columns]
    # -----------------------------------------------
    # 这里只需要df含有longitude, latitude，返回的时候还是返回df中的所有元素
    result_dict, fieldid_hash_dict = find_points_in_which_polygon(df=df, 
                                                                    shp_path=shp_path)
    # 返回田块的id，以及田块的geometry，properties
    # print(f"The following field has loss points:")
    field_res = {}
    for fieldid, info in result_dict.items():
        # print(f"fieldid: {fieldid}")
        # print(f"info: {info}")
        field_properties = fieldid_hash_dict[fieldid]["properties"]
        field_geom = fieldid_hash_dict[fieldid]["geom"]
        field_points = info
        field_res[fieldid] = {
            "properties": field_properties,
            "geom": field_geom,
            "points": field_points
        }

    # format_print_dict(field_res)

    
    return field_res


def gdal_array_type(np_datatype):
    np_datatype = str(np_datatype)

    dtype_to_gdal = {
        'uint8': gdal.GDT_Byte,
        'uint16': gdal.GDT_UInt16,
        'int16': gdal.GDT_Int16,
        'uint32': gdal.GDT_UInt32,
        'int32': gdal.GDT_Int32,
        'float32': gdal.GDT_Float32,
        'float64': gdal.GDT_Float64
    }
    supported_dtypes = list(dtype_to_gdal.keys())

    assert np_datatype in supported_dtypes, f"np_datatype:{np_datatype} not supported"
    return dtype_to_gdal[np_datatype]





# 读取自己打的取样点

def read_yield_from_shp(shp_path):
    """
    Read yield from shapefile
    :param shp_path: path of shapefile
    :return: yield
    """
    shp_name = os.path.basename(shp_path)
    farm, date, _, lossrate = shp_name.split('.')[0].split('_')
    lossrate = int(lossrate)/100
    with fiona.open(shp_path, 'r') as shp:
        # assert if it is point shapefile
        assert shp.schema['geometry'] == 'Point', 'Shapefile is not point shapefile'

        # read longitude and latitude
        loc_list = []
        # print(f"the length of shp is {len(shp)}")
        for feature in shp:
            longitude, latitude = feature['geometry']['coordinates']
            loc_list.append([longitude, latitude, farm, date, lossrate])

    loc_df = pd.DataFrame(loc_list, columns=['longitude', 'latitude', 'farm', 'date', 'lossrate'])
    return loc_df









