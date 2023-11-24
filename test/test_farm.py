import os,sys
import numpy as np
sys.path.append(r"D:\phd\CropMaster")

# 切换到项目根目录
# os.chdir(r"D:\phd\CropMaster")

import cropbase



def get_ndvi(red, nir):
    return (nir - red) / (nir + red + 0.0000000001)


def gen_rgb(red, green, blue):
    # 0-3000 -> 0-255
    red = red / 3000 * 255
    green = green / 3000 * 255
    blue = blue / 3000 * 255
    # (3, 256, 256)
    rgb = np.stack([red, green, blue], axis=0)
    rgb = rgb.astype(np.uint8)
    return rgb


def pf(**kwargs):
    field_image = kwargs["field_image"]

    properties = kwargs["properties"]
    # red = field_image[3]
    # nir = field_image[7]
    # ndvi = get_ndvi(red, nir)


    rgb = gen_rgb(field_image[3], field_image[2], field_image[1])
    return rgb



if __name__ == "__main__":
    sentinel2tif_path = r"D:\phd\phd2_first\20231104ygn_new_model\Data Annotations\dataset\sentinel2img\Rongjun2023\Rongjun_20230819.tif"
    farm_shp_path = r"D:\phd\phd2_first\20231104ygn_new_model\Data Annotations\2023farmshp\荣军地块2023_split.shp"  
    farm = cropbase.farm.FarmWithOneImg(sentinel2tif_path, farm_shp_path)
    farm.process_by_field(farm.features, pf)