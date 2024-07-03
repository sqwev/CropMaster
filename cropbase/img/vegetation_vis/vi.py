import numpy


class VegetationVIs:
    def __init__(self):
        pass

    @staticmethod
    def NDVI(nir, red):
        """
        计算归一化植被指数
        :param nir: 近红外波段
        :param red: 红波段
        :return: ndvi
        """
        return (nir - red) / (nir + red)

    @staticmethod
    def EVI(nir, red, blue):
        """
        计算增强型植被指数
        :param nir: 近红外波段
        :param red: 红波段
        :param blue: 蓝波段
        :return: evi
        """
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

    @staticmethod
    def GNDVI(nir, green):
        """
        计算归一化绿光植被指数
        :param nir: 近红外波段
        :param green: 绿波段
        :return: gndvi
        """
        return (nir - green) / (nir + green)

    @staticmethod
    def NDWI(nir, swir):
        """
        计算归一化水体指数
        :param nir: 近红外波段
        :param swir: 短波红外波段
        :return: ndwi
        """
        return (nir - swir) / (nir + swir)
