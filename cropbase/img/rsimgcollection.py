# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.

import os
import re
import datetime
import rasterio
from prettytable import PrettyTable
from tqdm import tqdm

from . import Sentinel2RSImg


class RSImgCollection:

    def __init__(self, dir):
        self.dir = dir
        self.img_list = self.getImg()

    def __str__(self):
        columns = ["tif_name", "tif_date", "tif_epsg", "tif_shape", "cloudage"]

        tb = PrettyTable()
        tb.field_names = columns
        for tif_path in tqdm(self.img_list):
            tif_info = self.get_tif_info(tif_path)
            row = []
            for col_name in columns:
                row.append(tif_info[col_name])
            tb.add_row(row)
        table_content = tb.get_string()
        return table_content

    def getImg(self):
        tif_file_list = []
        for root, dirs, files in os.walk(self.dir):
            for file in files:
                if file.endswith(".tif"):
                    tif_file_list.append(os.path.join(root, file))
        return tif_file_list

    def get_tif_info(self, tif_path):
        tif_name = os.path.basename(tif_path)
        # find continuous 8 numbers in tif_name
        matches = re.findall(r'\d{8}', tif_name)

        if len(matches) > 1:
            raise Exception("tif_name should only contain one continuous 8 numbers")

        if len(matches) == 0:
            raise Exception("tif_name should contain one continuous 8 numbers")

        tif_date = matches[0]
        tif_date = datetime.datetime.strptime(tif_date, "%Y%m%d")

        tif_epsg = rasterio.open(tif_path).crs.to_epsg()

        tif_border = rasterio.open(tif_path).bounds

        # pixel size
        tif_shape = rasterio.open(tif_path).shape

        cloudage = Sentinel2RSImg(tif_path).get_cloudage()

        cloudage = round(cloudage, 2)

        info = {
            "tif_path": tif_path,
            "tif_name": tif_name,
            "tif_date": tif_date,
            "tif_epsg": tif_epsg,
            "tif_shape": tif_shape,
            "tif_border": tif_border,
            "cloudage": cloudage
        }
        return info
