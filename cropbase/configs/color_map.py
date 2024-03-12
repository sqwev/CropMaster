# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
CROP_COLOR_MAP = {
    "rice": (34,181,113),
    "maize": (48,69,48),
    "soybean": (249,185,113),
    "wheat": (173,135,91),
}

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
for crop, color in CROP_COLOR_MAP.items():
    color = [c/255 for c in color]
    ax.scatter(0, 0, c=[color], label=crop)
ax.legend()
plt.show()