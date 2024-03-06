# Introduction for Crop Master

The Crop Master is a python package created for digital agriculture. For now, the core of
this package is to process the relationship between farm, field and image.

- Farm: farm can be considered as a collection of fields, and a farm can have many fields.
- Field: field is a field planted with crops
- Image: image is a remote sensing image, which can be satellite image or drone image.

There are many useful packages which can be used for anlysis remote sensing image, but they
usually don't consider the relationship between farm, field and image. This package is created
to solve this problem.

The `RsImg` class is the core of this package, it is used to process remote sensing image.

# RsImg

Now I abandon the `GDAL` package, and transfer to `rasterio` package, which is more convenient to use.
the `ds` property of `RsImg` object is a `rasterio` dataset, which is compatible with `rasterio` API.

We provide 2 ways to create RsImg object. One is to read from a file, the other is to create from a numpy array.

## Create RsImg object

```python
# Read from a file
rsimg = RsImg.from_tif('path/to/file.tif')
# Read from a numpy array
array = np.random.rand(3, 100, 100)  # input array should be C * H * W
nodatavalue = 0
geotransform = (0, 1, 0, 0, 0, 1)  # geo transform support gdal format and affine format
projection = 4326  # projection support gdal format, code and rasterio.crs.CRS format
rsimg = RsImg.from_array(array, nodatavalue, geotransform, projection)
```

We recommend to create RsImg object with property `name` and `date`

if `RsImg` object is created without `name` and `date`, the `name` will be set to `uuid.uuid1()`,
the date will be set to `datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`.

the useful property of `RsImg` object is:

- `ds`: rasterio dataset, campatible with rasterio API
- `name`: (str) the name of the image
- `date`: (str) the date of the image
- `nodatavalue`: the nodatavalue of the image
- `geotransform`: the geotransform of the image
- `projection`: the projection of the image
- `width`: the width of the image
- `height`: the height of the image
- `bands`: the bands of the image
- `dim`: the dimension of the image

## Some useful methods

Get image array

```python
array = rsimg.array
```

Get valid mask

```python
mask = rsimg.valid_mask
```

the `mask` is a numpy array with shape (HEIGHT, WIDTH), the value is 0 or 1, 0 means no data, 1 means has valid data.

Get tif border

```python
border = rsimg.border
```

Crop image by pixel position, it will return a new RsImg object with shape (100, 100)

```python
rsimg.crop_by_pixel(0, 0, 100, 100)
```

Slide crop image, It will save the croped images to `save_dir` with name `{name}_x_y.tif`

```python
rsimg.slide_crop(
    block_size=256,
    overlap=0.2,
    save_dir='path/to/save/dir'
)
```

Cut image by shapefile

```python
cut_rsimg = rsimg.cut_by_shp('path/to/file.shp')
```

Reproject image by espg code

```python
rsimg_4326 = rsimg.reproject(4326)
```

Auto reproject image to utm zone

```python
rsimg_utm = rsimg.auto_reproject_to_utm()
```

Save to tif file

```python
rsimg.to_tif('path/to/file.tif')
```

Select some bands from image, It will return a new RsImg object with only 3 bands.

```python
rsimg.select_bands([1, 2, 3])
```

## Sentinel2RsImg

Sentinel2RsImg is a subclass of RsImg, it is used to process sentinel2 image.
Now I provide `renderRGB` method to render the image to RGB image.

```python
rgb = sentinel2_rsimg.renderRGB()
```








