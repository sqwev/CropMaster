# RsImg


We provide 2 ways to create RsImg object. 
One is to read from a file, the other is to create from a numpy array.

## Read from a file

```python
# Read from a file
rsimg = RsImg.from_tif('path/to/file.tif')
# Read from a numpy array
rsimg = RsImg.from_array(array, nodatavalue, geotransform, projection)
```

We recommend to create RsImg object with property `name` and `date`


if `RsImg` object is created without `name` and `date`, the `name` will be set to `uuid.uuid1()`, 
the date will be set to `datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`.

the useful property of `RsImg` object is:

- `ds`: gdal dataset, campatible with gdal API
- `name`: the name of the image
- `date`: the date of the image
- `nodatavalue`: the nodatavalue of the image
- `geotransform`: the geotransform of the image
- `projection`: the projection of the image
- `WIDTH`: the width of the image
- `HEIGHT`: the height of the image
- `BANDS`: the bands of the image
- `dim`: the dimension of the image
- `x_min, x_max, y_min, y_max`: the boundary of the image



## Get valid mask

```python
mask = rsimg.valid_mask()
```
the `mask` is a numpy array with shape (HEIGHT, WIDTH), the value is 0 or 1, 0 means no data, 1 means has valid data.

## Save to tif file

```python
rsimg.to_tif('path/to/file.tif')
```