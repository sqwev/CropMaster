# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.

import colorsys
import math
import numpy as np
from enum import Enum, unique
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap

class Sentienl2ColorConverter:
    """
    Convert Sentinel-2 images to RGB images. default input shape is CHW
    """

    def __init__(self):
        pass

    @staticmethod
    def rgb_432(array):
        """
        Convert 432 bands to RGB,
        """
        # unify the range of img_array 0-3000 to 0-255
        array = np.asarray(array).clip(0, 3000).astype(np.uint16)
        img_array = array / 3000 * 255
        img_array = img_array.clip(0, 255)
        img_array = img_array.astype(np.uint8)
        return img_array
    @staticmethod
    def rgb_832(array):
        # first band clip to 0-8000, second band clip to 0-3000, third band clip to 0-3000

        # array_mode = indentify_CHW_HWC(array)
        array = np.asarray(array)
        array[0] = array[0].clip(0, 8000)
        array[1] = array[1].clip(0, 3000)
        array[2] = array[2].clip(0, 3000)

        rgb_array = np.zeros((3, array.shape[1], array.shape[2]), dtype=np.uint8)
        rgb_array[0] = array[0] / 8000 * 255
        rgb_array[1] = array[1] / 3000 * 255
        rgb_array[2] = array[2] / 3000 * 255
        rgb_array = rgb_array.clip(0, 255)
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array



