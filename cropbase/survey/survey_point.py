import os
import geopandas as gpd
import pandas as pd



class SurveyPoints:
    def __init__(self):
        self.path = 'path'
        # self.points_df = gpd.read_file(self.path)
        # self.check_feature_type(self.points_df, ["Point"])



    def check_feature_type(self, df, supported_feature_type):
        feature_type = df.geom_type.unique()
        for feature in feature_type:
            assert feature in supported_feature_type, f"feature type {feature} not supported"
        return True

    @staticmethod
    def get_lossrate_df(dir):
        """
        check shapefile in dir contains measurement.shp/***_20000202_lr_***.shp
        """

        def check_format_lrshp(lrshp_path):
            """
            check lrshp format
            """
            shp_name = os.path.basename(lrshp_path)
            underline_num = shp_name.count('_')
            if underline_num == 3:
                if shp_name.split('_')[2] == 'lr':
                    return True
                else:
                    return False
            else:
                return False

        default_espg = 4326
        lr_df_list = []
        for i in os.listdir(dir):
            if i.endswith('.shp'):
                if i == 'measurement.shp':
                    measurement_df = gpd.read_file(os.path.join(dir, i))
                    # only save lossrate and geometry
                    measurement_df = measurement_df[['lossrate', 'geometry']]
                    measurement_df = measurement_df.to_crs(epsg=default_espg)
                    lr_df_list.append(measurement_df)
                    print(f'detect measurement.shp')

                elif check_format_lrshp(i):
                    head = i.split('.')[0]
                    farm, date, lr_str, lr100 = head.split('_')
                    lr = int(lr100)/100
                    _lr_df = gpd.read_file(os.path.join(dir, i))
                    _lr_df['lossrate'] = lr
                    _lr_df = _lr_df.to_crs(epsg=default_espg)
                    lr_df_list.append(_lr_df)

                    print(f'detect {i}')
                else:
                    continue

        lr_df = pd.concat(lr_df_list)

        # reproject to default_espg
        lr_df = lr_df.to_crs(epsg=default_espg)
        measurement_df = measurement_df.to_crs(epsg=default_espg)
        total_df = pd.concat([measurement_df, lr_df])
        total_df.reset_index(drop=True, inplace=True)

        try:
            del total_df['Id']
        except:
            pass
        return total_df


    def get_losstype_df(self, dir):
        no_disaster_name = 'nodisaseter.shp'
        waterlogging_name = 'waterlogging.shp'
        lodging_name = 'lodging.shp'



