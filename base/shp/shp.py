import os
import warnings
import numpy as np
import pandas as pd
import fiona
import rtree
import json
import time
from osgeo import gdal, gdal_array, ogr, gdal, osr
from tqdm import tqdm
from shapely.geometry import Point, Polygon, shape, mapping

class Farmshp:
    def __init__(self, farm_shp_path) -> None:
        self.farm_shp_path = farm_shp_path
        self.IFMULTIPOLYGON = self.judge_multipolygon_exist()
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