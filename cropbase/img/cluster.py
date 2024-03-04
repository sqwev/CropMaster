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



def fix_cluster(df: pd.DataFrame, cluster_number, method):
    if method == "kmeans":
        clustering = KMeans(n_clusters=cluster_number).fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_


def auto_cluster(df: pd.DataFrame, cluster_number, method):
    if method == "dbscan":
        clustering = DBSCAN(eps=0.3, min_samples=cluster_number).fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_
    pass

def cal_ndvi(red, nir):
    return (nir - red) / (nir + red)

def sort_labels_by_ndvi(labels, centers, ndvi_list):
    ndvi_list = np.array(ndvi_list)
    sort_index = np.argsort(ndvi_list)
    labels_dict = {sort_index[i]: i for i in range(len(sort_index))}
    labels = [labels_dict[i] + 1 for i in labels]
    return labels


def builtin_cluster(df, cluster_number, method='kmeans'):
    """
    对df进行聚类，返回聚类结果，并根据聚类结果对等级进行排序，分为1到n级
    """
    if method == "kmeans":
        clustering = KMeans(n_clusters=cluster_number).fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_
    elif method == "dbscan":
        clustering = DBSCAN(eps=0.3, min_samples=10).fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_
    elif method == "meanshift":
        clustering = MeanShift().fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_
    elif method == "spectral":
        clustering = SpectralClustering(n_clusters=cluster_number).fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_
    elif method == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=cluster_number).fit(df)
        labels = clustering.labels_
        centers = clustering.cluster_centers_
    elif method == "gaussian":
        from sklearn.mixture import GaussianMixture
        clustering = GaussianMixture(n_components=cluster_number).fit(df)
        labels = clustering.predict(df)
        centers = clustering.cluster_centers_
    elif method == "hierarchical":
        from sklearn.cluster import Birch
        clustering = Birch(n_clusters=cluster_number).fit(df)
        labels = clustering.predict(df)
        centers = clustering.cluster_centers_
    else:
        raise Exception(f"method: {method} not supported")

    # print(f"centers: {centers}")
    # 计算每个center对应的ndvi
    ndvi_list = []

    def cal_ndvi(red, nir):
        return (nir - red) / (nir + red)

    for center in centers:
        ndvi = cal_ndvi(center[2], center[3])
        ndvi_list.append(ndvi)
    # print(f"ndvi_list: {ndvi_list}")

    # 根据聚类中心的ndvi值对等级进行排序
    ndvi_list = np.array(ndvi_list)
    sort_index = np.argsort(ndvi_list)
    labels_dict = {sort_index[i]: i for i in range(len(sort_index))}
    labels = [labels_dict[i] + 1 for i in labels]
    # sort_index = np.argsort(ndvi_list)
    # # print(f"sort_index: {sort_index}")
    # labels_dict = {}
    # for i in range(len(sort_index)):
    #     labels_dict[sort_index[i]] = i
    #
    # labels = [labels_dict[i] + 1 for i in labels]
    # # print(f"labels: {labels}")
    df["label"] = labels
    return df
