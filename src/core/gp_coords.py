# -*- coding: utf-8 -*-
"""
Created on Sat Nov 02 17:45:22 2024

@author: peter
"""

#%% Import modules

from pathlib import Path
from typing import Optional, Literal
import numpy as np
import pandas as pd
import re
import yaml
from pyproj import Transformer
from tqdm import tqdm

from gp_package.core.base import BaseFunction

#%% Aufgaben

#todo: choose reprojection based on _coordinates_proc --> correct zone (utm33n, utm34n, etc.)

#%% GPcoords class

class GPcoords(BaseFunction):
    """
    This class handles the coordinates of geophysical data.
    """
    def __init__(self, data: Optional[pd.DataFrame] = None, log_path: [Path, str]=  None) -> None:
        """
        This method initializes the GPcoords class.
        It contains a list with the necessary columns for the _coordinates_proc.
        It also contains a dictionary with the reprojection directory, read from a YAML file.

        Parameters
        ----------
        data : pd.DataFrame
            The coordinates of the geophysical data.
        log_path : Path or str
            The path to the log file.
        """
        self.logger = self._setup_logger(log_path=log_path)
        self.logger.info('Setting up CoordinateHandler')
        reproj_path = Path(__file__).parents[1] / 'templates' / 'reproj.yml'
        self._reproj_dir = yaml.safe_load(Path(reproj_path).read_text())

        self._needed_cols = ['Name', 'Latitude', 'Longitude', 'Elevation']
        self._selected_reprojection = None
        self._coordinate_path = None
        self._coordinates = None if data is None else data
        self._extracted_coords = {}

    def needed_cols(self) -> list:
        """
        Returns the necessary columns for the _coordinates_proc.

        Returns
        -------
        list
            The necessary columns for the coordinates.
        """
        return self._needed_cols

    def coordinates(self) -> pd.DataFrame:
        """
        Returns the _coordinates_proc.

        Returns
        -------
        pd.DataFrame
            The coordinates.
        """
        return self._coordinates

    def reproj_dir(self) -> dict:
        """
        Returns the reprojection directory.

        Returns
        -------
        dict
            The reprojection directory.
        """
        return self._reproj_dir

    def selected_reprojection(self) -> dict:
        """
        Returns the selected reprojection.

        Returns
        -------
        dict
            The selected reprojection.
        """
        return self._selected_reprojection

    def coordinate_path(self) -> Path:
        """
        Returns the path to the coordinates.

        Returns
        -------
        Path
            The path to the coordinates.
        """
        return self._coordinate_path

    def read(self, file_path: [Path, str], sep: str = ',') -> None:
        """
        This function reads a file with _coordinates_proc and stores it in self._coordinates_proc.

        Parameters
        ----------
        file_path: Path or str
            The path to the file with _coordinates_proc.
        sep: str
            The separator used in the file.

        Returns
        -------
        None
        """
        self._coordinate_path = Path(file_path)
        self._coordinates = pd.read_csv(self._coordinate_path, sep=sep)
        self.logger.info(f'read_file: Coordinates read from file {self._coordinate_path.name}')
        self._check_cols()

    def _check_cols(self) -> bool:
        """
        This function checks if all necessary columns are present in self._coordinates_proc.

        Returns
        -------
        bool
            Returns True if all necessary columns are present, False otherwise.
        """
        self.logger.info('check_cols: Checking _coordinates_proc columns')
        missing_cols = [col for col in self._needed_cols if col not in self._coordinates.columns]
        must_rename = len(missing_cols) != 0
        if must_rename:
            self.logger.warning('check_cols: Not all necessary columns found.')
            self.logger.warning(f'check_cols: Missing columns: {missing_cols}')
        return must_rename

    def rename_columns(self, renaming_dict: dict) -> None:
        """
        This function renames the columns of self._coordinates_proc.
        Then it checks if all necessary columns are present.

        Parameters
        ----------
        renaming_dict : dict
            A dictionary with the old column names as keys and the new column names as values.

        Returns
        -------
        None
        """
        self._coordinates.rename(columns=renaming_dict, inplace=True)
        self.logger.info(f'rename_columns: Renaming of columns was successful.')
        self._check_cols()

    def rename_points(self, renaming_dict: dict) -> None:
        """
        This function renames the points in self._coordinates_proc.

        Parameters
        ----------
        renaming_dict : dict
            A dictionary with the old point names as keys and the new point names as values.
            (Tip: This function replaces all occurrences of a given key. For instance,
            if you want to replace all '_' with '-', you can use {'_':'-'} as renaming_dict.)

        Returns
        -------
        None
        """
        for key, value in renaming_dict.items():
            self._coordinates['Name'] = self._coordinates['Name'].str.replace(key, value)
        self.logger.info(f'rename_points: Renaming of points was successful.')

    def sort(self, inplace: [Literal[True], Literal[False]] = True, **kwargs) -> Optional[pd.DataFrame]:
        """
        This function sorts the _coordinates_proc in self._coordinates_proc.
        (using pandas.DataFrame.sort_values)

        Parameters
        ----------
        inplace: bool
            If True, the function sorts the _coordinates_proc in place.
        kwargs: dict
            Additional keyword arguments for pandas.DataFrame.sort_values.

        Returns
        -------
        None or pd.DataFrame
        """
        self.logger.info('sort: Sorting self._coordinates_proc.')
        if inplace:
            self._coordinates.sort_values(inplace=inplace, **kwargs)
        else:
            return self._coordinates.sort_values(**kwargs)

    @staticmethod
    def _split_name(name: str, pattern: str) -> [list, tuple]:
        """
        This function splits a name using a regex pattern.

        Parameters
        ----------
        name : str
            The name to be split.
        pattern : str
            The regex pattern to split the name.

        Returns
        -------
        list or tuple
        """
        match = re.match(pattern, name)
        if match:
            return match.groups()
        return [None] * len(re.findall(r"\((.*?)\)", pattern))

    def sort_points(self, inplace: bool = True, ascending: bool = True, pattern: str = r'([A-Za-z]+)(\d*)_(\d*)') -> Optional[pd.DataFrame]:
        """
        This function sorts the points in self._coordinates_proc.

        Parameters
        ----------
        inplace : bool
            If True, the function sorts the points in place.
        ascending : bool
            If True, the function sorts the points in ascending order.
        pattern : str
            The regex pattern to split the name.

        Returns
        -------
        None or pd.DataFrame
        """
        df = self._coordinates.copy()
        num_groups = len(re.findall(r"\((.*?)\)", pattern))

        df[[f'Part{i + 1}' for i in range(num_groups)]] = df['Name'].apply(
            lambda x: self._split_name(x, pattern)).apply(pd.Series)

        for i in range(num_groups):
            col_name = f'Part{i + 1}'
            try:
                df[col_name] = self.safe_to_numeric(data=df[col_name])
            except (ValueError, TypeError) as e:
                self.logger.warning(f'Could not convert {col_name} to numeric. Error: {e}')

        df.sort_values(by=[f'Part{i + 1}' for i in range(num_groups)], inplace=True, ascending=ascending)
        df.drop(columns=[f'Part{i + 1}' for i in range(num_groups)], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.logger.info(f'sort_points: Sorting points was successful.')
        if inplace:
            self._coordinates = df
        else:
            return df

    def write(self, file_path: [Path, str], sep: str = ',') -> None:
        """
        This function saves the _coordinates_proc in self._coordinates_proc to a file.
        (using pandas.DataFrame.to_csv)

        Parameters
        ----------
        file_path : Path or str
            The path to the file where the _coordinates_proc will be saved.
        sep : str
            The separator used in the file.

        Returns
        -------
        None
        """
        path = Path(file_path)
        self._coordinates.to_csv(path, sep=sep, index=False)
        self.logger.info(f'to_file: Coordinates saved to file {str(path)}.')

    def reproject(self, reproj_key: str = 'wgs84_utm33n', correct_height: bool = True) -> None:
        """
        This function transforms the _coordinates_proc in self._coordinates_proc to a new CRS.
        The new CRS is defined in self._reproj_dir.

        Parameters
        ----------
        reproj_key : str
            The key of the desired reprojection in self._reproj_dir.
        correct_height : bool
            If True, the function corrects the elevation for the antenna height.

        Returns
        -------
        None
        """
        self._selected_reprojection = self._reproj_dir.get(reproj_key)
        if self._selected_reprojection is None:
            self.logger.error('reproject: Key not found.')
            raise KeyError('Key not found. Try custom_reprojection instead.')
        elif self._selected_reprojection.get('initial') is None or self._selected_reprojection.get('to') is None:
            self.logger.error('reproject: Parsed dictionary is missing necessary keys.')
            raise KeyError('Parsed dictionary is missing necessary keys.')

        self.logger.info(f'reproject: Using key: {reproj_key}')
        epsg_in, epsg_out = self._selected_reprojection.get('initial'), self._selected_reprojection.get('to')
        transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)

        lons = self._coordinates['Longitude'].values
        lats = self._coordinates['Latitude'].values

        eastings, northings = transformer.transform(lons, lats)

        self._coordinates['Easting'] = eastings
        self._coordinates['Northing'] = northings

        self._coordinates['Code'] = self._selected_reprojection.get('to')

        if 'Antenna height' in self._coordinates.columns and 'Ellipsoidal height' in self._coordinates.columns:
            if correct_height:
                elevations = self._coordinates['Ellipsoidal height'].values - self._coordinates['Antenna height'].values #correcting for antenna height
                self.logger.info('reproject: Corrected height for antenna length.')
            else:
                elevations = self._coordinates['Ellipsoidal height'].values
            self._coordinates['Elevation'] = elevations

        else:
            if correct_height:
                self.logger.warning('Necessary columns for correction not found.')
        self.logger.info('reproject: Successfully reprojected _coordinates_proc.')

    def custom_reprojection(self, from_crs: str, to_crs: str, correct_height: bool = True) -> None:
        """
        This function transforms the _coordinates_proc in self._coordinates_proc to a new CRS.
        The new CRS is defined by the user.

        Parameters
        ----------
        from_crs : str
            The initial CRS of the _coordinates_proc.
        to_crs : str
            The desired CRS of the _coordinates_proc.
        correct_height : bool
            If True, the function corrects the elevation for the antenna height.

        Returns
        -------
        None
        """
        self._reproj_dir['custom'] = {'initial': from_crs, 'to': to_crs}
        self.reproject(reproj_key='custom', correct_height=correct_height)

    def _interpolate_profile(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        This function interpolates the _coordinates_proc of a profile.

        Parameters
        ----------
        group : pd.DataFrame
            The _coordinates_proc of a profile

        Returns
        -------
        pd.DataFrame
        """
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

    def interpolate_points(self) -> None:
        """
        This function interpolates the _coordinates_proc of the profiles in self._coordinates_proc.

        Returns
        -------
        None
        """
        self._coordinates['Profile'] = self._coordinates['Name'].str.split('_').str[0]
        self._coordinates['Point'] = self._coordinates['Name'].str.split('_').str[1]
        self._coordinates['Point'] = self.safe_to_numeric(data=self._coordinates['Point'])
        if 'Description' not in self._coordinates.columns:
            self._coordinates['Description'] = np.nan

        grouped = self._coordinates.groupby('Profile')
        result = []
        for name, group in tqdm(grouped, total=len(grouped), desc='Interpolating Profiles'):
            result.append(self._interpolate_profile(group))
        self._coordinates = pd.concat(result).reset_index(drop=True)

        cols = [col for col in self._coordinates.columns if col != 'Point']
        cols.append('Point')
        self._coordinates = self._coordinates[cols]


    def extract_coords(self):
        """
        This function extracts the _coordinates_proc from self._coordinates_proc and stores them in self._extracted_coords.

        Returns
        -------
        dict
            A dictionary with the _coordinates_proc of the profiles.
        """
        if 'Profile' not in self._coordinates.columns:
            self._coordinates['Profile'] = self._coordinates['Name'].str.split('_').str[0]

        for profile, group in self._coordinates.groupby('Profile'):
            if len(group) > 1:
                self._extracted_coords[profile] = group[['Easting', 'Northing', 'Elevation']].rename(columns={
                    'Easting': 'x',
                    'Northing': 'y',
                    'Elevation': 'z'
                })
            else:
                point = group.iloc[0]
                self._extracted_coords[profile] = {
                    'x': point['Easting'],
                    'y': point['Northing'],
                    'z': point['Elevation']
                }

        self.logger.info('extract_coords: Successfully extracted _coordinates_proc.')
        return self._extracted_coords


    @staticmethod
    def create_dummy(elec_sep: [int, float], elec_num: int) -> pd.DataFrame:
        """
        This function creates a dummy _coordinates_proc for testing purposes.

        Parameters
        ----------
        elec_sep : int or float
            separation between electrodes
        elec_num : int
            number of electrodes

        Returns
        -------
        pd.DataFrame
            A DataFrame with the dummy _coordinates_proc.
        """
        x_values = [i * elec_sep for i in range(elec_num)]
        y_values = [0] * elec_num
        z_values = [0] * elec_num

        return pd.DataFrame({'x': x_values, 'y': y_values, 'z': z_values})


    def close(self) -> None:
        """
        This function closes the logger and resets the class variables.

        Returns
        -------
        None
        """
        self.close_logger()
        self._coordinates = None
        self._extracted_coords = None
        self._coordinate_path = None
        self._selected_reprojection = None
        self._reproj_dir = None