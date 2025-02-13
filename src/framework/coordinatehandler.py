# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:38:41 2024

@author: peter
"""

#%% Import modules

from pathlib import Path

import numpy as np
import pandas as pd
import logging
import re
import yaml
from pyproj import Transformer
from tqdm import tqdm
from gp_package.core.base import BaseFunction

#%% Aufgaben

#%% CoordinateHandler class

class CoordinateHandler(BaseFunction):
    def __init__(self, basedir=None, log_filename=None):
        self.basedir = Path(basedir) if basedir is not None else Path.cwd()
        self.log_filename = Path(log_filename) if log_filename is not None else None

        self.logger = self._setup_logger(log_path=self.log_filename)
        self.logger.info('Setting up CoordinateHandler')
        reproj_path = Path(__file__).parents[1] / 'templates' / 'reproj.yml'
        self.reproj_dir = yaml.safe_load(Path(reproj_path).read_text())

        self.needed_cols = ['Name', 'Latitude', 'Longitude', 'Elevation']
        self.selected_reprojection = None
        self.coordinate_path = None
        self.coordinates = None
        self.extracted_coords = {}


    def read_coord_file(self, coords, sep=','):
        self.coordinate_path = Path(coords)
        self.coordinates = pd.read_csv(self.coordinate_path, sep=sep)
        self.logger.info(f'read_file: Coordinates read from file {self.coordinate_path.name}')


    def check_cols(self):
        self.logger.info('check_cols: Checking _coordinates_proc columns')

        missing_cols = [col for col in self.needed_cols if col not in self.coordinates.columns]
        must_rename = len(missing_cols) != 0
        if must_rename:
            self.logger.warning('check_cols: Not all necessary columns found.')
            self.logger.warning(f'check_cols: Missing columns: {missing_cols}')
        return must_rename


    def rename_points(self, renaming_dict:dict):
        for key, value in renaming_dict.items():
            self.coordinates['Name'] = self.coordinates['Name'].str.replace(key, value)
        self.logger.info(f'rename_points: Renaming of points was successful.')


    def rename_columns(self, renaming_dict:dict):
        self.coordinates.rename(columns=renaming_dict, inplace=True)
        self.logger.info(f'rename_columns: Renaming of columns was successful.')


    def sort(self, inplace=True, **kwargs):
        self.logger.info('sort: Sorting self._coordinates_proc.')
        if inplace:
            self.coordinates.sort_values(inplace=inplace, **kwargs)
        else:
            return self.coordinates.sort_values(**kwargs)


    @staticmethod
    def _split_name(name, pattern):
        match = re.match(pattern, name)
        if match:
            return match.groups()
        return [None] * len(re.findall(r"\((.*?)\)", pattern))


    def sort_points(self, inplace=True, ascending=True, pattern=r'([A-Za-z]+)(\d*)_(\d*)'):

        df = self.coordinates.copy()
        num_groups = len(re.findall(r"\((.*?)\)", pattern))

        df[[f'Part{i + 1}' for i in range(num_groups)]] = df['Name'].apply(
            lambda x: self._split_name(x, pattern)).apply(pd.Series)

        for i in range(num_groups):
            col_name = f'Part{i + 1}'
            try:
                df[col_name] = pd.to_numeric(df[col_name])
            except (ValueError, TypeError) as e:
                self.logger.warning(f'Could not convert {col_name} to numeric. Error: {e}')

        df.sort_values(by=[f'Part{i + 1}' for i in range(num_groups)], inplace=True, ascending=ascending)
        df.drop(columns=[f'Part{i + 1}' for i in range(num_groups)], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.logger.info(f'sort_points: Sorting points was successful.')
        if inplace:
            self.coordinates = df
        else:
            return df


    def to_file(self, path, sep=','):
        self.coordinates.write(path, sep=sep)
        self.logger.info(f'to_file: Coordinates saved to file {path}')


    def reproject(self, reproj_key='wgs84_utm33n', correct_height=True):
        self.selected_reprojection = self.reproj_dir.get(reproj_key)
        if self.selected_reprojection is None:
            self.logger.error('reproject: Key not found.')
            raise KeyError('Key not found. Try custom_reprojection instead.')
        elif self.selected_reprojection.get('initial') is None or self.selected_reprojection.get('to') is None:
            self.logger.error('reproject: Parsed dictionary is missing necessary keys.')
            raise KeyError('Parsed dictionary is missing necessary keys.')

        self.logger.info(f'reproject: Using key: {reproj_key}')
        epsg_in, epsg_out = self.selected_reprojection.get('initial'), self.selected_reprojection.get('to')
        transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)

        lons = self.coordinates['Longitude'].values
        lats = self.coordinates['Latitude'].values

        eastings, northings = transformer.transform(lons, lats)

        self.coordinates['Easting'] = eastings
        self.coordinates['Northing'] = northings

        self.coordinates['Code'] = self.selected_reprojection.get('to')


        if 'Antenna height' in self.coordinates.columns and 'Ellipsoidal height' in self.coordinates.columns:
            if correct_height:
                elevations = self.coordinates['Ellipsoidal height'].values - self.coordinates['Antenna height'].values #correcting for antenna height
                self.logger.info('reproject: Corrected height for antenna length.')
            else:
                elevations = self.coordinates['Ellipsoidal height'].values
            self.coordinates['Elevation'] = elevations

        else:
            if correct_height:
                self.logger.warning('Necessary columns for correction not found.')
        self.logger.info('reproject: Successfully reprojected _coordinates_proc.')

    def custom_reprojection(self, from_crs, to_crs, correct_height=True):
        """
        This function makes it possible to parse two CRS-codes which pyproj.Transformer recognizes for easy custom reprojection.
        :param correct_height: Correct Elevation for antenna height
        :param from_crs: Initial CRS
        :param to_crs: Target CRS
        :return: None. Selected CRS are stored in self._selected_reprojection.
        """
        self.reproj_dir['custom'] = {'initial': from_crs, 'to': to_crs}
        self.reproject(reproj_key='custom', correct_height=correct_height)


    def _interpolate_profile(self, group):
        profile = group['Profile'].unique()[0]
        group = group.sort_values('Point')
        try:
            full_range = pd.DataFrame({'Point': range(int(group['Point'].min()), int(group['Point'].max()) + 1)})
        except ValueError:
            self.logger.info(f'Skipped interpolation of {profile}.')
            return group

        full_group = pd.merge(full_range, group, on='Point', how='left')

        # Interpolate missing values
        full_group['Latitude'] = full_group['Latitude'].interpolate()
        full_group['Longitude'] = full_group['Longitude'].interpolate()

        if 'Ellipsoidal height' in full_group.columns:
            full_group['Ellipsoidal height'] = full_group['Ellipsoidal height'].interpolate()

        if 'Antenna height' in full_group.columns:
            full_group['Antenna height'] = full_group['Antenna height'].ffill()

        if full_group['Easting'].notna().sum() > 1 and full_group['Northing'].notna().sum() > 1:
            full_group['Easting'] = full_group['Easting'].interpolate()
            full_group['Northing'] = full_group['Northing'].interpolate()
            full_group['Code'] = full_group['Code'].ffill()
        else:
            self.logger.warning(f'_interpolate_profile: {profile}: Easting or Northing column was mostly empty, skipping interpolation.')

        full_group['Elevation'] = full_group['Elevation'].interpolate()

        full_group['Description'] = full_group['Description'].astype(object)
        full_group.loc[full_group['Name'].isna(), 'Description'] = 'interpolated'
        full_group['Profile'] = full_group['Profile'].ffill()
        full_group['Name'] = full_group['Name'].fillna(full_group['Profile'] + '_' + full_group['Point'].astype(int).astype(str))

        return full_group


    def interpolate_points(self):
        self.coordinates['Profile'] = self.coordinates['Name'].str.split('_').str[0]
        self.coordinates['Point'] = self.coordinates['Name'].str.split('_').str[1]
        self.coordinates['Point'] = pd.to_numeric(self.coordinates['Point'], errors='coerce')
        if 'Description' not in self.coordinates.columns:
            self.coordinates['Description'] = np.nan

        # self._coordinates_proc = self._coordinates_proc.groupby('Profile').apply(self._interpolate_profile).reset_index(drop=True)

        grouped = self.coordinates.groupby('Profile')
        result = []
        for name, group in tqdm(grouped, total=len(grouped), desc='Interpolating Profiles'):
            result.append(self._interpolate_profile(group))
        self.coordinates = pd.concat(result).reset_index(drop=True)

        cols = [col for col in self.coordinates.columns if col != 'Point']
        cols.append('Point')
        self.coordinates = self.coordinates[cols]


    def extract_coords(self):
        if self.coordinates['Easting'].notna().sum() <= 1 or self.coordinates['Northing'].notna().sum() <= 1:
            self.reproject()
        self.interpolate_points()

        for profile, group in self.coordinates.groupby('Profile'):
            if len(group) > 1:
                self.extracted_coords[profile] = group[['Easting', 'Northing', 'Elevation']].rename(columns={
                    'Easting': 'x',
                    'Northing': 'y',
                    'Elevation': 'z'
                })
            else:
                point = group.iloc[0]
                self.extracted_coords[profile] = {
                    'x': point['Easting'],
                    'y': point['Northing'],
                    'z': point['Elevation']
                }

        self.logger.info('extract_coords: Successfully extracted _coordinates_proc.')
        return self.extracted_coords


    @staticmethod
    def create_dummy(elec_sep, elec_num):
        x_values = [i * elec_sep for i in range(elec_num)]
        y_values = [0] * elec_num
        z_values = [0] * elec_num

        return pd.DataFrame({'x': x_values, 'y': y_values, 'z': z_values})


    def close(self):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
                self.logger.info("Coordinate handler closed and removed.")
                break

        self.coordinates = None
        self.extracted_coords = None