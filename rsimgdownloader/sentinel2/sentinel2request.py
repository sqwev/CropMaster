import os
import pandas as pd
import numpy as np
import glob
import datetime
import time
import json
import fiona
import shutil
from osgeo import gdal


class SentinelhubRequester:
    """
    - dataset
        - farm
        - S2_L1C
        - S2_L2A
        - useful_file
    """

    def __init__(self, config, time_interval, region_path, dataset_path, farm_name):
        self.config = config
        self.time_interval = time_interval  # such as ('20180101', '20181231')
        self.region_path = region_path
        self.dataset_path = dataset_path
        self.farm_name = farm_name

        farm_dir = os.path.join(self.dataset_path, "farm", self.farm_name)
        if not os.path.exists(farm_dir):
            os.makedirs(farm_dir)

        self.check_init()
        self.start_date = datetime.datetime.strptime(self.time_interval[0], '%Y%m%d')
        self.end_date = datetime.datetime.strptime(self.time_interval[1], '%Y%m%d')

    def check_init(self):
        """
        time_interval: must be tuple of len 2, and type == str and len = 8
        shp_path: must be shapefile path, and shx, dbf file must exist
        """
        # check time_interval
        if type(self.time_interval) != tuple:
            raise ValueError('time_interval must be tuple')
        if len(self.time_interval) != 2:
            raise ValueError('time_interval must be tuple of len 2')
        # type == str and len = 8
        if type(self.time_interval[0]) != str or len(self.time_interval[0]) != 8:
            raise ValueError('time_interval[0] must be str and len = 8')
        if type(self.time_interval[1]) != str or len(self.time_interval[1]) != 8:
            raise ValueError('time_interval[1] must be str and len = 8')

        start_date = datetime.datetime.strptime(self.time_interval[0], '%Y%m%d')
        end_date = datetime.datetime.strptime(self.time_interval[1], '%Y%m%d')

        if start_date > end_date:
            raise ValueError('time_interval[0] must be smaller than time_interval[1]')

        # check shapefile path
        try:
            with fiona.open(self.shp_path):
                pass
        except:
            raise ValueError('shp_path must be shapefile path, and shx, dbf file must exist')

        # check farm_name
        if type(self.farm_name) != str:
            raise ValueError('farm_name must be str')



    def get_regoin(self):
        """
        shp, geojson
        """