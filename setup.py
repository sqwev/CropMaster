# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.

from setuptools import setup, find_packages

setup(
    name='cropmaster',
    version='0.1.0',
    author='Shenzhou Liu',
    author_email='shenzhouliu@whu.edu.cn',
    description='A brief description of your package',
    packages=find_packages(),  # 自动发现和包含所有的包
    install_requires=[  # 任何你的包所依赖的第三方库
        'numpy',
        'pandas',
    ],
)