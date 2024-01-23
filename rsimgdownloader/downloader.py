import argparse
import sqlite3
import os

from abc import ABCMeta, abstractclassmethod


class SatelitteDownloader(metaclass=ABCMeta):

    @classmethod
    def download(self):
        pass





def default_arguments_parser():
    parser = argparse.ArgumentParser(description='Download satellite images')
    parser.add_argument('--satelitte', type=str, default='sentinel2', help='Satelitte name')
    parser.add_argument('--start_date', type=str, help='Date of image')
    parser.add_argument('--end_date', type=str, help='Date of image')
    parser.add_argument('--region', type=str, help='Region of image')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to save image')
    return parser


class Sentinel2TileDownloader(SatelitteDownloader):
    def __init__(self, dir):
        self.dir = dir
        self.create_db()


    def create_db(self):
        db_path = os.path.join(self.dir, 'sentinel2.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sentinel2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            level TEXT,
            date DATE,
            path TEXT,
            cloud_cover REAL,
        )''')
        conn.commit()
        conn.close()



    def download(self):
        print("Downloading sentinel2 image on date: {} to path: {}".format(self.date, self.path))
        return True


