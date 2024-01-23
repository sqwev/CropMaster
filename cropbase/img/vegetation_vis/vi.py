import numpy


class VegetationVIs:
    def __init__(self, img=None):
        self.img = img

    def NDVI(self, nir, red):
        """
        计算归一化植被指数
        :param nir: 近红外波段
        :param red: 红波段
        :return: ndvi
        """
        return (nir - red) / (nir + red)

    def EVI(self, nir, red, blue):
        """
        计算增强型植被指数
        :param nir: 近红外波段
        :param red: 红波段
        :param blue: 蓝波段
        :return: evi
        """
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


