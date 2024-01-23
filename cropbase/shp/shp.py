import os
import warnings
import numpy as np
import pandas as pd
import fiona
import rtree
import json
import time
import geopandas as gpd
from osgeo import gdal, gdal_array, ogr, gdal, osr
from tqdm import tqdm
from shapely.geometry import Point, Polygon, shape, mapping

class Farmshp:
    def __init__(self, farm_shp_path) -> None:
        self.farm_shp_path = farm_shp_path
        self.IFMULTIPOLYGON = self.judge_multipolygon_exist()
        self.features = fiona.open(self.farm_shp_path)
        print(f"IFMULTIPOLYGON: {self.IFMULTIPOLYGON}")


        
    def split_shp(self):
        # 如果shp含有polygon或者multipolygon以外的，就报错
        with fiona.open(self.farm_shp_path) as shp:
            for feature in shp:
                geom = shape(feature['geometry'])
                if geom.geom_type not in ["Polygon", "MultiPolygon"]:
                    raise Exception(f"geom type error: {geom.geom_type}")
                
        newshp_path = self.farm_shp_path.replace(".shp", "_split.shp")
        # 将shp中的multipolygon拆分成多个polygon
        with fiona.open(self.farm_shp_path, 'r') as src:
            polygon_count = -1
            # 新建字段uniqueid
            # src.schema['properties']['uni_id'] = 'int'
            with fiona.open(newshp_path, 'w', driver=src.driver, crs=src.crs, schema=src.schema) as dst:
                for feature in src:
                    geom = shape(feature['geometry'])
                    properties = feature['properties']
                    if feature['geometry']['type'] == 'Polygon':
                        polygon_count += 1
                        # properties['uni_id'] = polygon_count
                        dst.write({
                            'geometry': mapping(geom),
                            'properties': properties
                        })
                    if feature['geometry']['type'] == 'MultiPolygon':
                        properties = feature['properties']
                        for single_polygon in geom:
                            polygon_count += 1
                            # properties['uni_id'] = polygon_count
                            dst.write({
                                'geometry': mapping(single_polygon),
                                'properties': properties
                            })

        return newshp_path
    



    def judge_multipolygon_exist(self):
        with fiona.open(self.farm_shp_path) as shp:
            for feature in shp:
                geom = shape(feature['geometry'])
                if geom.geom_type == "MultiPolygon":
                    return True
            return False

    def get_espg(self):
        with fiona.open(self.farm_shp_path) as shp:
            return shp.crs.to_epsg()
        
    def read_property(self):
        # return pd dataframe with geometry

        properties = []
        for feature in self.features:
            properties.append(feature['properties'])
        df = pd.DataFrame(properties)
        df['geometry'] = [shape(feature['geometry']) for feature in shp]
        return df

    def reproject(self, epsg:int):
        # reproject to epsg
        # epsg: int
        # return shp_path
        # save_path: str
        newshp_path = self.farm_shp_path.replace(".shp", "_" + str(epsg) + ".shp")
        gdf = gpd.read_file(self.farm_shp_path)
        # Reproject to the target coordinate reference system (CRS)
        gdf_reprojected = gdf.to_crs(epsg=epsg)
        # Save the reprojected shapefile
        gdf_reprojected.to_file(newshp_path, driver='ESRI Shapefile')
        return newshp_path

    # 根据位置获得porperty
    def read_property_by_position(self, lon, lat):
        # return pd dataframe with geometry

        properties = {}
        for feature in self.features:
            geom = shape(feature['geometry'])
            if geom.contains(Point(lon, lat)):
                properties['geometry'] = geom
                for key, value in feature['properties'].items():
                    properties[key] = value

        return properties

    def find_position_in_which_polygon(self, position):
        # position: (lon, lat)
        # return: [polygon_id1:{}]

        if isinstance(position, tuple):
            position_list = [position]
        elif isinstance(position, list):
            position_list = position
        else:
            raise Exception("position must be tuple or list")

        idx = rtree.index.Index()

        for i, feature in enumerate(self.features):
            geom = shape(feature['geometry'])
            properties = feature['properties']
            # 这里我们不考虑multipolygon和polygon的区别
            content = {
                "geom": geom,
                "properties": properties
            }
            idx.insert(i, geom.bounds)

        result = []
        for position in position_list:
            lon, lat = position
            point = Point(lon, lat)
            # 使用索引进行查询o
            possible_polygons = list(idx.intersection(point.bounds))
            intersected_polygons = []
            for polygon_id in possible_polygons:
                feature = self.features[polygon_id]
                geom = shape(feature['geometry'])
                properties = feature['properties']
                # 这里我们不考虑multipolygon和polygon的区别
                content = {
                    "id": polygon_id,
                    "geom": geom,
                    "properties": properties
                }
                if point.within(geom):
                    intersected_polygons.append(content)
            # 此时intersected_polygons里存储的是这个点在哪些shp里

            # 将点添加到对应的polygon中
            if len(intersected_polygons) != 0:
                result.append(intersected_polygons)
            else:
                result.append(None)

        if isinstance(position, tuple):
            return result[0]
        elif isinstance(position, list):
            return result















class CropDisasterSamplePoints():
    """
    灾损取样点
    """
    def __init__(self, shp_path, year=None):
        self.shp_path = shp_path
        self.year = year

        # 检查是否为point 
        with fiona.open(self.shp_path) as shp:
            for feature in shp:
                geom = shape(feature['geometry'])
                if geom.geom_type != "Point":
                    raise Exception(f"geom type error: {geom.geom_type}")
                
    def get_espg(self):
        with fiona.open(self.shp_path) as shp:
            return shp.crs.to_epsg()
        
    def read_property(self):
        # return pd dataframe with geometry
        with fiona.open(self.shp_path) as shp:
            properties = []
            for feature in shp:
                properties.append(feature['properties'])
            df = pd.DataFrame(properties)
            df['geometry'] = [shape(feature['geometry']) for feature in shp]
            return df