import os
import datetime as dt
import numpy as np
import fiona
import json
import subprocess
import shutil
import geopandas as gpd
from sentinelhub import CRS, BBox, DataCollection, SHConfig
from sentinelhub import SentinelHubCatalog


def tran_shp2bbox(shp_path):
    """
    shp_path: shapefile path
    """
    # check shapefile path
    shp_dir, shp_file = os.path.split(shp_path)
    shp_name, shp_ext = os.path.splitext(shp_file)
    shx_path = os.path.join(shp_dir, shp_name + '.shx')
    dbf_path = os.path.join(shp_dir, shp_name + '.dbf')
    # check all above exist
    if not os.path.exists(shp_path):
        raise ValueError('shp_path not exist')
    if not os.path.exists(shx_path):
        raise ValueError('shx_path not exist')
    if not os.path.exists(dbf_path):
        raise ValueError('dbf_path not exist')
    # read shapefile by fiona
    with open(shp_path, 'r') as f:
        shp = fiona.open(shp_path)
        bbox = shp.bounds
    return bbox


class SentinelhubRetrivel:
    def __init__(self, region, start_date, end_date):
        """
        shp_path: path
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        addtional: the last date won't be included
        """

        # region: geojson or shp, or bbox, or geojson path
        if isinstance(region, str):
            # 判断是不是geojson
            if region.endswith('.geojson') or region.startswith('{'):
                self.region = gpd.read_file(region)
            elif region.endswith('.shp'):
                self.region = gpd.read_file(region)
            else:
                raise ValueError("region must be geojson or shp or geojson path")

        elif isinstance(region, tuple):
            self.bbox = region # (min_x, min_y, max_x, max_y)

        else:
            raise ValueError("region must be geojson or shp or geojson path or bbox")

        # bbox 增加一点点范围
        add_region_sacle = 0.01
        min_x, min_y, max_x, max_y = self.bbox
        width = max_x - min_x
        height = max_y - min_y
        add_width = width * add_region_sacle
        add_height = height * add_region_sacle
        self.bbox = (min_x - add_width, min_y - add_height, max_x + add_width, max_y + add_height)
        self.bbox = BBox(bbox=self.bbox, crs=CRS.WGS84)


        self.config = SHConfig()
        if self.config.sh_client_id == "" or self.config.sh_client_secret == "":
            print(
                "Warning! To use Sentinel Hub Catalog API, please provide the credentials (client ID and client secret).")
        self.catalog = SentinelHubCatalog(config=self.config)
        self.start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

    def search_sentinel2(self, bbox, start_date, end_date, max_cloud_cover, level):

        # max_cloud_cover 0-100 int
        assert 0 <= max_cloud_cover <= 100 and isinstance(max_cloud_cover, int), ("max_cloud_cover must be int between "
                                                                                  "0 and 100")
        # YYYY-MM-DD
        assert isinstance(start_date, str) and isinstance(end_date, str), ("start_date and end_date must be str")
        # level L1C or L2A
        assert level in ["L1C", "L2A"], ("level must be L1C or L2A")
        # bbox

        if level == "L1C":
            search_iterator = self.catalog.search(
                DataCollection.SENTINEL2_L1C,
                bbox=bbox,
                time=(start_date, end_date),
                filter="eo:cloud_cover < " + str(max_cloud_cover),
                fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []}

            )
        elif level == "L2A":
            search_iterator = self.catalog.search(
                DataCollection.SENTINEL2_L2A,
                bbox=bbox,
                time=(start_date, end_date),
                filter="eo:cloud_cover < " + str(max_cloud_cover),
                fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []}
            )
        else:
            raise ValueError("level must be L1C or L2A")

        search_results = list(search_iterator)
        # result content: {'id': 'S2B_MSIL1C_20210106T025109_N0209_R132_T51UXP_20210106T040647', 'properties': {'datetime': '2021-01-06T02:54:14Z', 'eo:cloud_cover': 99.87}}
        # search_res = []
        # for i in search_results:
        #     search_res.append(i['id'])
        return search_results

    def get_tile_feature(self, sensor_type, tile_id):
        """
        sensor_type: L1C or L2A
        tile_id: tile id
        """
        if sensor_type == "L1C":
            tile_feature = self.catalog.get_feature(DataCollection.SENTINEL2_L1C, tile_id)
        elif sensor_type == "L2A":
            tile_feature = self.catalog.get_feature(DataCollection.SENTINEL2_L2A, tile_id)
        else:
            raise ValueError("sensor_type must be L1C or L2A")
        return tile_feature

    # 将各种坐标系的shp和geojson转换为wgs84坐标系的geojson
    def region2bbox(self, region, epsg=4326):
        wgs84_region = region.to_crs(epsg=epsg)
        min_x = min(wgs84_region.bounds['minx'])
        min_y = min(wgs84_region.bounds['miny'])
        max_x = max(wgs84_region.bounds['maxx'])
        max_y = max(wgs84_region.bounds['maxy'])
        return min_x, min_y, max_x, max_y

    def get_support_retrival(self):
        support_retrival = [DataCollection.SENTINEL2_L1C, DataCollection.SENTINEL2_L2A]
        return support_retrival

