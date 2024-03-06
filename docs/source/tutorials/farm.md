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

### some useful methods

get public property of fields
```python
farm.get_geoDataFrame()
```

export to json or shapefile

```python
farm.to_file('path/to/file.json')
farm.to_file('path/to/file.shp')
```

split multi-polygon to single polygon

```python
farm.split_multipolygon_fields()
```

find points in which field

```python
farm.find_points_in_which_field(df: gpd.GeoDataFrame, split_multipolygon: bool)
```
`df` must be a point GeoDataFrame