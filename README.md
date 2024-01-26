# CropMaster

The Crop Master is a python package created for digital agriculture. For now, the core of
this package is to process the relationship between farm, field and image.  

- Farm: farm can be considered as a collection of fields, and a farm can have many fields.
- Field: field is a field planted with crops
- Image: image is a remote sensing image, which can be satellite image or drone image.

There are many useful packages which can be used for anlysis remote sensing image, but they 
usually don't consider the relationship between farm, field and image. This package is created
to solve this problem.

The doc for this project has published on [Read the Docs](https://cropmaster.readthedocs.io/en/latest/)

This project is only for learning, if you want to use it in your project, please contact me. All rights reserved.

## Installation

It relies on gdal and rasterio to read and write remote sensing image, 
so you need to install gdal first.

I recommend to install gdal with conda.

```bash
conda install -c conda-forge gdal==3.7.1
conda install -c conda-forge rasterio
```

Then install CropMaster with pip.

```bash
pip install -r requirements.txt
```

