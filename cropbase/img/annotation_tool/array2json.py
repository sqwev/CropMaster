# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import json
import pycocotools.mask as mask_util
import numpy as np


def is_sparse_array(array, nan=np.nan):
    """
    default nan is null value, statistical  nan proportion in array, if nan proportion is greater than 0.5, return True
    Args:
        array:

    Returns:

    """
    if np.isnan(nan):
        return np.isnan(array).sum() > 0.5 * array.size
    else:
        return (array == nan).sum() > 0.5 * array.size

def get_mask(array, nan=np.nan):
    """
    Convert array to mask
    Args:
        array:
        nan:

    Returns:

    """
    if np.isnan(nan):
        return np.isnan(array)
    else:
        return array == nan
def array2json(array, decimal=2, nan=np.nan):
    """
    Convert array to json format
    Args:
        array:
        decimal:

    Returns:

    """
    is_sparse = is_sparse_array(array)
    array = np.around(array, decimal)
    if is_sparse:
        mask = get_mask(array)
        rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        print(rle)
        return rle

        # quantized_values = np.around(values / quantization_step).astype(int)
    else:
        json_str = json.dumps({
            "shape": array.shape,
            "dtype": str(array.dtype),
            "data": array.tolist()
        })
        import bz2
        # compressed_data = bz2.compress(json_str)
        return json_str



if __name__ == "__main__":
    # random array
    array = np.random.rand(256, 256)
    array = np.around(array, 2)

    # random add nan
    array[0, 0] = np.nan
    array[1, 1] = np.nan
    print(array)

    array_bytes = array2json(array)
    print(array_bytes)