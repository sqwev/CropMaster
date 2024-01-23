# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.


class Satellite:
    sentinel2 = {
        "bands":{
            'B1': {
                'name': 'Coastal aerosol',
                'wavelength': '442.7nm',
                'resolution': '60m',
            },
            'B2': {
                'name': 'Blue',
                'wavelength': '492.4nm',
                'resolution': '10m',
            },
            'B3': {
                'name': 'Green',
                'wavelength': '559.8nm',
                'resolution': '10m',
            },
            'B4': {
                'name': 'Red',
                'wavelength': '664.6nm',
                'resolution': '10m',
            },
            'B5': {
                'name': 'Red Edge 1',
                'wavelength': '704.1nm',
                'resolution': '20m',
            },
            'B6': {
                'name': 'Red Edge 2',
                'wavelength': '740.5nm',
                'resolution': '20m',
            },
            'B7': {
                'name': 'Red Edge 3',
                'wavelength': '782.8nm',
                'resolution': '20m',
            },
            'B8': {
                'name': 'NIR',
                'wavelength': '832.8nm',
                'resolution': '10m',
            },
            'B8A': {
                'name': 'Red Edge 4',
                'wavelength': '864.7nm',
                'resolution': '20m',
            },
            'B9': {
                'name': 'Water vapour',
                'wavelength': '945nm',
                'resolution': '60m',
            },
            'B10': {
                'name': 'SWIR - Cirrus',
                'wavelength': '1373.5nm',
                'resolution': '60m',
            },
            'B11': {
                'name': 'SWIR 1',
                'wavelength': '1613.7nm',
                'resolution': '20m',
            },
            'B12': {
                'name': 'SWIR 2',
                'wavelength': '2202.4nm',
                'resolution': '20m',
            },
        },
        "visualize": {
            "bands": ['B4', 'B3', 'B2'],
            "min": 0,
            "max": 3000,
        },
    }




