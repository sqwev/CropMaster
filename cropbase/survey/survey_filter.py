import os
import numpy as np
import pandas as pd

from tqdm import tqdm

class SurveyData:
    supported_columns = ["latitude", "longitude", "cropyield", "lossrate"]
    def __init__(self, df) -> None:
        self.filepath = df

    def read_df(self, filepath):

        if isinstance(filepath, pd.DataFrame):
            df = filepath
        elif isinstance(filepath, str):
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                raise Exception("File format not supported")
        else:
            raise Exception("File format not supported")


        # unify to longitude, latitude
        same_meaning_columns = {
            "longitude": ["lon", "经度"],
            "latitude": ["lat", "纬度"],
            "lossrate": ["loss", "损失率"],
        }
        for key, value in same_meaning_columns.items():
            for v in value:
                if v in df.columns:
                    df = df.rename(columns={v: key})


        # check latitude, longitude in right ranges
