## Farm

Farm is a collection of fields, and a farm can have many fields.

Suggested transfer params:

- :param file: shp or geojson
- :param convert_dict: convert cols name in shp to standard cols name
- :param name: str Farm name
- :param gis_index_type: arcgis or qgis  This is used for init the index of field index,
for Arcgis start from 0, and Qgis start from 1.

The file can be a shp file or a geojson file, which is used to read by geopandas.
Then return a `field_df` object, which is property of `Farm` object.

The `fields` property is a list of `Field` object, which is created from `field_df`.




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