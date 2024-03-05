# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import pandas as pd
import numpy as np
import geopandas as gpd

from ..img import RsImg, RestrictedDict




class Field:
    """
    My design of Field class:
    The Field class is a field, which can be defined by polygon/multipolygon or raster image, or only by an index.

    The public properties of a field is a pandas series, which only contains scalar values.(Now I think)

    In the future: Following the writing style of PCSE, soil, crops, site, agricultural management, weather,
    Status variables, Speed variables


    :param index: int, the index of the field
    :param img_dict: dict, the dict of RSImg
    :param crs: str, the projection of the field
    :param geometry: shapely.geometry.Polygon or shapely.geometry.MultiPolygon, the geometry of the field
    :param pdseries: pd.Series, the public properties of the field
    :name str, the name of the field
    """
    index = None
    geometry = None
    crs = None
    name = None
    # dict
    public_properties = pd.Series(dtype=object)
    _private_properties = {
        'img_dict': RestrictedDict(RsImg)  # 存储这个field的图像
    }

    def __init__(self, index: int = None, img_dict=None, crs=None, geometry=None, *args, **kwargs) -> None:

        if index is not None:
            self.index = index
        if geometry is not None:
            self.geometry = geometry
        if crs is not None:
            self.crs = crs
        if img_dict is not None:
            for k, v in img_dict.items():
                self._private_properties["img_dict"][k] = v

        pdseries = kwargs.get("pdseries", None)
        if pdseries is not None:
            self.geometry = pdseries["geometry"]
            self.public_properties = pdseries
            self.index = pdseries.name

        self.name = kwargs.get("name", None)

    def __str__(self):
        return f"Field: {self.index}"

    def set_properties(self, properties: dict):
        """
        set public properties of the field

        :param properties: dict
        """
        for key, value in properties.items():
            self.public_properties[key] = value

    def register_img(self, img: RsImg, name: str):
        """
        register a RSImg with name to the field. The RSImg will be stored in the img_dict of the private
        property of the field object.

        :param img: RSImg
        :param name: str The name of the RSImg
        """
        self._private_properties["img_dict"][name] = img

    def deregister_img(self, name: str):
        """
        deregister a RSImg with name to the field. The RSImg will be removed from the img_dict of the private

        :param name: str The name of the RSImg
        """
        del self._private_properties["img_dict"][name]

    def get_img(self, name: str):
        """
        get a RSImg with name to the field.

        :param name: str The name of the RSImg
        """
        return self._private_properties["img_dict"][name]

    def to_geodataframe(self):
        """
        Create a GeoDataFrame from the public properties and geometry of the field

        :return: gpd.GeoDataFrame
        """
        # 根据公共属性和几何信息创建GeoDataFrame
        gdf = gpd.GeoDataFrame([self.public_properties], geometry=[self.geometry], crs=self.crs)
        return gdf

    @staticmethod
    def filter_point_value_df(point_df: gpd.GeoDataFrame, img):
        """
        add col_idx, row_idx to point_df and del point not in field

        :param point_df: gpd.GeoDataFrame It must be only contain point geometry
        :param img: RSImg
        :return: gpd.GeoDataFrame
        """
        assert point_df.shape[0] > 0, "point_df is empty"
        # value column mest be numeric


        # if crs is the same between point_df and img
        if point_df.crs != img.projection:
            point_df = point_df.to_crs(img.projection)

        img_valid_mask = img.valid_mask

        # obtain col_idx, row_idx of img for each point, del if point not in img
        loc_list = []
        for i in range(len(point_df)):
            # get x, y of point from geometry
            longitude = point_df.iloc[i]['geometry'].x
            latitude = point_df.iloc[i]['geometry'].y
            # 计算在田块中的位置
            col_idx = int(np.ceil((longitude - img.x_min) / img.x_res)) - 1
            row_idx = int(np.ceil((latitude - img.y_min) / img.y_res)) - 1
            # 如果valid_mask中的值为0，说明这个点不在田块内，不考虑这个点
            if img_valid_mask[row_idx, col_idx] == 0:
                print(f"point:{longitude}, {latitude} not in field")
                loc_list.append([col_idx, row_idx, 0])
                continue
            loc_list.append([col_idx, row_idx, 1])

        # 整合到point_df
        loc_df = pd.DataFrame(loc_list, columns=["col_idx", "row_idx", 'if_in_field'])

        point_df['col_idx'] = loc_df['col_idx'].tolist()
        point_df['row_idx'] = loc_df['row_idx'].tolist()
        point_df['if_in_field'] = loc_df['if_in_field'].tolist()

        # drop point not in field
        point_df = point_df[point_df['if_in_field'] == 1]
        del point_df['if_in_field']
        return point_df

    def gen_aug_mask(self, cluster_mask: np.ndarray, location_df: pd.DataFrame, type="mean"):
        """
        Input a cluster_mask and a location_df, then generate a new mask, mapper the location_df to the cluster_mask.

        :param cluster_mask: np.ndarray The cluster mask
        :param location_df: pd.DataFrame or gpd.GeoDataFrame must contain col_idx, row_idx, value fields
        :param type: str, mean or max_point_number
        :return: np.ndarray
        """

        # 首先，locationdf中的点，必须在cluster_mask中，而且不能在一个像元中有重复的
        raise Exception("The method has been deprecated")
        assert location_df.shape[0] > 0, "locationdf is empty"

        # 统计每个点落在哪个聚类中
        cluster_label_list = []
        for i, row in location_df.iterrows():
            # print(row)
            col_idx = row["col_idx"]
            row_idx = row["row_idx"]
            point_value = row["value"]
            col_idx = int(col_idx)
            row_idx = int(row_idx)
            cluster_label = cluster_mask[row_idx, col_idx]
            cluster_label_list.append(cluster_label)
        location_df["cluster_label"] = cluster_label_list

        if type == "mean":
            # 如果有多个点落在同一类中，取平均值
            cluster_label_value_df = location_df.groupby("cluster_label").mean()
            cluster_label_value_df = cluster_label_value_df.reset_index()
            # 将cluster_label_value_df中的值映射到cluster_mask中
            cluster_label_value_df = cluster_label_value_df[["cluster_label", "value"]]

            cluster_img = cluster_mask.copy()
            # 将-1等级为nan
            grades = list(set(cluster_img.flatten()))

            labeled_grade = list(set(cluster_label_value_df["cluster_label"]))
            unlabeld_grade = list(set(grades) - set(labeled_grade))
            # print(f"grades: {grades}")
            # print(f"labeled_grade: {labeled_grade}")
            # print(f"unlabeld_grade: {unlabeld_grade}")
            cluster_img = cluster_img.astype(np.float32)

            # 将没有损失率对应的等级置为nan
            for unlabeld in unlabeld_grade:
                cluster_img[cluster_mask == unlabeld] = np.nan

            for i, row in cluster_label_value_df.iterrows():
                cluster_label, value = row
                cluster_label = float(cluster_label)
                print(f"cluster_label: {cluster_label}, value: {value}")
                cluster_img[cluster_mask == cluster_label] = value
        elif type == "max_point_number":
            # 如果有多个点落在同一类中，取最多的那个点的值
            cluster_label_value_df = location_df.groupby(["cluster_label", "value"]).size().reset_index()
            cluster_label_value_df.columns = ["cluster_label", "value", "count"]
            # 如果一个cluster_label上有多个value，取数量多的为这个等级的灾损类型
            cluster_label_value_df = cluster_label_value_df.sort_values(by=["cluster_label", "count"], ascending=False)
            cluster_label_value_df.drop_duplicates(subset=["cluster_label"], keep="first", inplace=True)

            cluster_label_value_df = cluster_label_value_df[cluster_label_value_df["cluster_label"] != -1]
            # print(cluster_label_value_df)
            # 将cluster_label_value_df中的值映射到cluster_mask中
            cluster_label_value_df = cluster_label_value_df[["cluster_label", "value"]]
            cluster_img = cluster_mask.copy()
            # 将-1等级为nan
            grades = list(set(cluster_img.flatten()))
            labeled_grade = list(set(cluster_label_value_df["cluster_label"]))
            unlabeld_grade = list(set(grades) - set(labeled_grade))

            cluster_img = cluster_img.astype(np.float32)
            # 将没有损失率对应的等级置为nan
            for unlabeld in unlabeld_grade:
                cluster_img[cluster_mask == unlabeld] = np.nan

            for i, row in cluster_label_value_df.iterrows():
                cluster_label, value = row
                cluster_label = float(cluster_label)
                # print(f"cluster_label: {cluster_label}, value: {value}")
                cluster_img[cluster_mask == cluster_label] = value

        return cluster_img

    @staticmethod
    def aug_mask(cluster_mask: np.ndarray, location_df: pd.DataFrame, type: str, value_col_name: str):
        """
        Input a cluster_mask and a location_df, then generate a new mask, mapper the location_df to the cluster_mask.

        :param cluster_mask: np.ndarray The cluster mask
        :param location_df: pd.DataFrame or gpd.GeoDataFrame must contain col_idx, row_idx, value fields
        :param type: str, mean or max_point_number
        :return: np.ndarray The augmented mask, nodata is nan
        """
        support_type = ["mean", "max_point_number"]
        assert location_df.shape[0] > 0, "location_df is empty"
        assert type in support_type, f"type must be in {support_type}"

        # 直接利用 Numpy 索引功能来获取聚类标签
        cluster_labels = cluster_mask[location_df['row_idx'].astype(int), location_df['col_idx'].astype(int)]
        location_df['cluster_label'] = cluster_labels

        if type == "mean":
            # 使用聚类标签的平均值
            cluster_values = location_df.groupby('cluster_label')[value_col_name].mean()
        elif type == "max_point_number":
            # 首先计算每个聚类标签的点数，然后取点数的最大值
            cluster_values = location_df.groupby('cluster_label')[value_col_name].agg(
                lambda x: x.value_counts().index[0])
        else:
            raise ValueError("Type must be 'mean' or 'max_point_number'")

        # 创建一个新的 numpy 数组，用 nan 初始化
        augmented_mask = np.full(cluster_mask.shape, np.nan, dtype=np.float32)

        # 更新 augmented_mask
        for cluster_label, value in cluster_values.items():
            augmented_mask[cluster_mask == cluster_label] = value

        return augmented_mask


class Fields:
    def __init__(self, fields: dict) -> None:
        self.fields = fields
