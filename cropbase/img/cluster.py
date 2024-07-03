# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import os
import inspect
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering, Birch

from .rsimg2 import _get_CHW_array_shape


def _get_inputs_args(imgprocess, inputs):
    if imgprocess.input_args is None:
        prms = list(inspect.signature(imgprocess.get_transform).parameters.items())
        if len(prms) == 1:
            names = ("image",)
        else:
            names = []
            for name, prm in prms:
                if prm.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                ):
                    raise TypeError(
                        f""" \
                The default implementation of `{type(imgprocess)}.__call__` does not allow \
                `{type(imgprocess)}.get_transform` to use variable-length arguments (*args, **kwargs)! \
                If arguments are unknown, reimplement `__call__` instead. \
                """
                    )
                names.append(name)
    args = []
    for f in imgprocess.input_args:
        try:
            args.append(getattr(inputs, f))
        except AttributeError as e:
            raise AttributeError(
                f"{type(imgprocess)}.get_transform needs input attribute '{f}', "
                f"but it is not an attribute of {type(inputs)}!"
            ) from e
    return args


class ImgProcess:
    input_args: Optional[Tuple[str]] = None

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args):
        raise NotImplementedError

    def __call__(self, inputs):
        args = _get_inputs_args(self, inputs)
        tfm = self.get_transform(*args)
        inputs.transform(tfm)
        return tfm


def cal_ndvi(red, nir):
    return (nir - red) / (nir + red)


def sort_labels_by_ndvi(labels, ndvi_list):
    ndvi_list = np.array(ndvi_list)
    sort_index = np.argsort(ndvi_list)
    labels_dict = {sort_index[i]: i for i in range(len(sort_index))}
    labels = [labels_dict[i] + 1 for i in labels]
    return labels


class ImgCluster:
    def __init__(self, cluster_number: int, add_position_encoding: bool = False):
        self.cluster_number = cluster_number
        self.add_position_encoding = add_position_encoding

    def __call__(self, array):
        dim = len(array.shape)
        assert dim == 2 or dim == 3, f"array dim should be 2 or 3, but got {dim}"
        height, width, channels = _get_CHW_array_shape(array)
        if self.add_position_encoding:
            array = self._add_position_encoding(array)
            columns_name = [f"B{i + 1}" for i in range(channels)] + ["x", "y"]
            channels += 2
        else:
            columns_name = [f"B{i + 1}" for i in range(channels)]
        df = self.array2df(array)
        df.columns = columns_name
        filtered_df = self._field_filter(df, nodatavalue=0)

        if len(filtered_df) == 0:
            raise Exception("can't be use cluster, because all the value is nodatavalue")

        # uniform filtered_df each column to 0-1
        filtered_df_copy = filtered_df.copy()
        for col in filtered_df_copy.columns:
            filtered_df_copy[col] = (filtered_df_copy[col] - filtered_df_copy[col].min()) / (
                        filtered_df_copy[col].max() - filtered_df_copy[col].min())

        clustered_df = self.cluster(filtered_df_copy)
        cluster_label = np.full(width * height, np.nan)  # 创建一个填充了 -1 的数组
        indices = clustered_df.index.values
        mask = np.isin(np.arange(width * height), indices)  # 创建一个布尔掩码来检查索引是否存在于 clustered_df 中
        cluster_label[mask] = clustered_df.loc[indices, "label"].values  # 使用布尔掩码来更新 cluster_label
        save_array = cluster_label.reshape(height, width)
        return save_array

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _add_position_encoding(self, array) -> np.ndarray:
        height, width, channels = _get_CHW_array_shape(array)
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = np.expand_dims(x, axis=-1)  # 将x扩展为与array的通道数相同
        y = np.expand_dims(y, axis=-1)  # 将y扩展为与array的通道数相同
        # H * W * C to C * H * W
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        position_encoding = np.concatenate((x, y), axis=0)
        stack_array = np.concatenate((array, position_encoding), axis=0)
        return stack_array

    def array2df(self, array: np.ndarray) -> pd.DataFrame:
        """
        将array转换为DataFrame
        """
        height, width, channels = _get_CHW_array_shape(array)
        array = np.transpose(array, (1, 2, 0))  # C * H * W to H * W * C
        array = array.reshape(height * width, channels)
        df = pd.DataFrame(array)
        return df

    def _field_filter(self, df, nodatavalue):
        df = df[~df['B1'].isin([nodatavalue])]
        return df


class KmeansCluster(ImgCluster):
    def __init__(self, cluster_number: int, add_position_encoding: bool = False):
        super().__init__(cluster_number, add_position_encoding)

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        self.cluster_number = min(self.cluster_number, len(df))
        clustering = KMeans(n_clusters=self.cluster_number).fit(df)
        df_copy.loc[:, "label"] = clustering.labels_
        return df_copy


class DBSCANCluster(ImgCluster):
    def __init__(self,
                 cluster_number: int,
                 add_position_encoding: bool = False,
                 eps: float = 0.5):
        super().__init__(cluster_number, add_position_encoding)
        self.eps = eps
        self.min_samples = cluster_number

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        clustering = DBSCAN(eps=0.9, min_samples=self.min_samples).fit(df)
        df["label"] = clustering.labels_
        return df


def recode_labels(label_array, ndvi_array):
    """
    Recode labels based on the ndvi value of each pixel
    """
    # statistics the ndvi value of each label
    unique_labels = np.unique(label_array)
    # if nan in unique_labels, del
    unique_labels = unique_labels[~np.isnan(unique_labels)]

    label_ndvi_stats = {}
    for label in unique_labels:
        # 计算每个标签对应的 NDVI 值统计信息
        label_ndvi_stats[label] = np.mean(ndvi_array[label_array == label])

    # recode the labels based on the ndvi value from low to high
    sorted_labels = sorted(label_ndvi_stats, key=lambda x: label_ndvi_stats[x], reverse=True)

    # np nan like

    recode_label_array = np.full_like(label_array, np.nan)
    for idx, label in enumerate(sorted_labels):
        recode_label_array[label_array == label] = idx
    return recode_label_array
