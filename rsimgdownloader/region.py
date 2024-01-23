import os
import warnings
import numpy as np
import pandas as pd
import fiona
import rtree
import json
import time
import rasterio
from rasterio.mask import raster_geometry_mask
import pandas as pd
import numpy as np
from osgeo import gdal, gdal_array, ogr, gdal, osr
from tqdm import tqdm
from shapely.geometry import Point, Polygon, shape, mapping


# support input types: shp, geojson, wkt, wkb, shapely.geometry


class FeatureReader:
    def __init__(self, feature):
        self.feature = self.read()


    def read(self, path):
        supported_types = ["shp", "geojson", "wkt", "wkb", "shapely.geometry"]

        if isinstance(path, str):
            # check file type
            if path.endswith(".shp"):
                return self.read_shp(self.feature)
            elif path.endswith(".geojson"):
                return self.read_geojson(self.feature)
            else:
                raise Exception("Unsupported file type: %s" % self.feature)




    def read_shp(self, shp_path):
        with fiona.open(shp_path) as f:
            features = [feature for feature in f]
        return features

    def read_geojson(self, geojson_path):
        with open(geojson_path) as f:
            features = json.load(f)
        return features

    def read_wkt(self, wkt):
        return ogr.CreateGeometryFromWkt(wkt)

    def read_wkb(self, wkb):
        return ogr.CreateGeometryFromWkb(wkb)

    def read_shapely(self, shapely_geometry):
        return shapely_geometry

