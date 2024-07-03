# Farm

Farm is a collection of fields, and a farm can have many fields. In default
setting, we regard a farm as a combination of fields, and a field is a polygon.

We usually want to do statistics on the fields properties, such as area, so in 
Farm class, I use geodataframe to store the fields' information.

## Create a Farm object

```python
your_farm = Farm(file='path/to/your/file.shp',
                 name="your_farm_name")
```
In some cases, the name rule of shapefile fields or geojson fields is not standard, 
so you can use `convert_dict` to convert the fields name to standard name.

```python
convert_dict = {
    "col_name": {
        "plot": "field"
    },
    "col_value": {
        x: lambda x: x * 1000
    },
}
your_farm = Farm(file='path/to/your/file.shp',
                 name="your_farm_name",
                 convert_dict=convert_dict)
```
Farm is a collection of fields, so I design fields in Farm object is a ordered list,
however, the field index in Arcgis is start from 0, and in Qgis is start from 1, 
so you can use `gis_index_type` to set the index type.

```python
your_farm = Farm(file='path/to/your/file.shp',
                 name="your_farm_name",
                 gis_index_type='arcgis')
```

## some useful methods

get public property of fields
```python
farm.get_geoDataFrame()
```

export Farm to json or shapefile

```python
farm.to_file('path/to/file.json')
farm.to_file('path/to/file.shp')
```

split multi-polygon to single polygon

```python
farm.split_multipolygon_fields()
```

find points in which field, `df` must be a point GeoDataFrame

```python
farm.find_points_in_which_field(df: gpd.GeoDataFrame, split_multipolygon: bool)
```

Plot the farm

```python
farm.plot()
```

## Farm with one Img

We often use a shapefile or geojson and a tif file to do some analysis or
statistics, so I design a class to combine the shapefile and tif file.

```python
farm_with_img = FarmWithImg(shp_file='path/to/your/file.shp',
                            img_file='path/to/your/file.tif',)
```

Select a field; the field_0 object is a Field object, and you can use `get_img()` 
method to get the img cutted by the field's geometry.
    
```python
field_0 = farm_with_img[0]
```

