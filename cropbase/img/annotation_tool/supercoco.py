# -*- coding: utf-8 -*-
# @author: Shenzhou Liu
# @email: shenzhouliu@whu.edu.cn
# @copyright: © 2024 Shenzhou Liu. All rights reserved.
import itertools
from pycocotools.coco import COCO, _isArrayLike
from collections import defaultdict


class SuperCOCO(COCO):
    """
    SuperCOCO is a subclass of COCO, which is used to load the regression annotations.
    The additional attributes are:
    "regressions": [
        {
            "id": 1,
            "image_id": 2,
            "reg_id": 1,
            "array": "Heshan_l1c_2023_204_4080.tif_yield.npy"
        },
    ],
    "reg_categories": [
        {
            "id": 1,
            "name": "yield"
        }
    ],
    """

    def __init__(self, annotation_file=None):
        super(SuperCOCO, self).__init__(annotation_file)
        self.reg_anns, self.reg_cats = dict(), dict()
        self.imgToRegAns, self.RegcatToImgs = defaultdict(list), defaultdict(list)
        print(f"SuperCOCO creating index...")
        reg_anns, reg_cats = {}, {}
        imgToRegAns, RegcatToImgs = defaultdict(list), defaultdict(list)
        if "regressions" in self.dataset:
            for reg_ann in self.dataset["regressions"]:
                imgToRegAns[reg_ann['image_id']].append(reg_ann)
                reg_anns[reg_ann['id']] = reg_ann
        if "reg_categories" in self.dataset:
            for reg_cat in self.dataset["reg_categories"]:
                reg_cats[reg_cat['id']] = reg_cat

        if "regressions" in self.dataset and "reg_categories" in self.dataset:
            for reg_ann in self.dataset["regressions"]:
                RegcatToImgs[reg_ann['reg_id']].append(reg_ann['image_id'])

        print(f"SuperCOCO index created!")
        self.reg_anns = reg_anns
        self.reg_cats = reg_cats
        self.imgToRegAns = imgToRegAns
        self.RegcatToImgs = RegcatToImgs

    def getImgIds(self, imgIds=[], catIds=[], reg_catIds=[]):
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        reg_catIds = reg_catIds if _isArrayLike(reg_catIds) else [reg_catIds]

        if len(imgIds) == len(catIds) == len(reg_catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and catId:
                    ids &= set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
            for i, reg_catId in enumerate(reg_catIds):
                if i == 0 and reg_catId:
                    ids &= set(self.RegcatToImgs[reg_catId])
                else:
                    ids &= set(self.RegcatToImgs[reg_catId])

        return list(ids)

    def getRegIds(self, imgIds=[], reg_catIds=[]):
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        reg_catIds = reg_catIds if _isArrayLike(reg_catIds) else [reg_catIds]

        if len(imgIds) == len(reg_catIds) == 0:
            anns = self.dataset['regressions']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToRegAns[imgId] for imgId in imgIds if imgId in self.imgToRegAns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['regressions']

            anns = anns if len(reg_catIds) == 0 else [ann for ann in anns if ann['reg_id'] in reg_catIds]

        ids = [ann['id'] for ann in anns]
        return ids

    def getRegCatIds(self, reg_catIds=[]):
        reg_catIds = reg_catIds if _isArrayLike(reg_catIds) else [reg_catIds]
        if len(reg_catIds) == 0:
            cats = self.dataset['reg_categories']
        else:
            cats = self.dataset['reg_categories']
            cats = cats if len(reg_catIds) == 0 else [cat for cat in cats if cat['id'] in reg_catIds]

        ids = [cat['id'] for cat in cats]
        return ids

    def loadRegs(self, ids=[]):
        if _isArrayLike(ids):
            return [self.reg_anns[id] for id in ids]
        elif type(ids) == int:
            return [self.reg_anns[ids]]

    def loadRegCats(self, ids=[]):
        if _isArrayLike(ids):
            return [self.reg_cats[id] for id in ids]
        elif type(ids) == int:
            return [self.reg_cats[ids]]
