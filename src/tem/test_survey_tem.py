# -*- coding: utf-8 -*-
"""
Created on Sun Nov 03 19:31:36 2024

@author: peter & jakob
"""

#%% Import modules

from pathlib import Path
import warnings
from tqdm import tqdm
from typing import Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# from pandas.conftest import axis_1
from pygimli.viewer.mpl import drawModel1D
from scipy.interpolate import griddata, CubicSpline
from scipy.optimize import curve_fit #todo: schauen ob man das sinnvoll einbauen kann?? @jakob

from src.core.gp_file import GPfile
from src.tem.TEM_frwrd.TEM_inv import tem_inv_smooth1D
from src.framework.survey_base import SurveyBase

warnings.filterwarnings('ignore')


#%% Aufgaben

#%% SurveyTEM class

class SurveyTEM(SurveyBase):
    def __init__(self, project_directory: [Path, str], dir_template: str = 'tem_default') -> None:
        
        self._data_modelled = None
        self._mu = 4 * np.pi * 10 ** (-7)
        plt.close('all')
        plt.rcParams['figure.dpi'] = 300
        
        super().__init__(project_dir=project_directory, dir_structure=dir_template)

        self._data_raw_path = None
        self._data_preprocessed_path = None
        self._data_filtered_paths = None
        self._data_inverted_paths = None

        self._data_raw = None
        self._data_preprocessed = None
        self._data_filtered = None
        self._data_inverted = None
        
        self.chosen_data = {}
        self.path_plot_sounding = None

    def data_raw_path(self) -> Path:
        """
        Returns the path of the raw data.

        Returns
        -------
        Path
            Path of the raw data.
        """
        return self._data_raw_path

    def data_preprocessed_path(self) -> Path:
        """
        Returns the path of the preprocessed data.

        Returns
        -------
        Path
            Path of the preprocessed data.
        """
        return self._data_preprocessed_path

    def data_filtered_paths(self) -> list:
        """
        Returns the paths of the filtered data.

        Returns
        -------
        list
            Paths of the filtered data.
        """
        self._data_filtered_paths = [path for path in self._folder_structure.get('data_filtered').iterdir() if path.is_file()]
        return self._data_filtered_paths

    def data_inverted_paths(self) -> list:
        """
        Returns the paths of the inverted data.

        Returns
        -------
        list
            Paths of the inverted data.
        """
        self._data_inverted_paths = [path for path in self._folder_structure.get('data_inversion').iterdir() if path.is_file()]
        return self._data_inverted_paths

    def data_raw(self) -> dict:
        """
        Returns the raw data.

        Returns
        -------
        dict
            Raw data.
        """
        return self._data_raw

    def data_preprocessed(self) -> dict:
        """
        Returns the preprocessed data.

        Returns
        -------
        dict
            Preprocessed data.
        """
        return self._data_preprocessed

    def data_filtered(self) -> dict:
        """
        Returns the filtered data.

        Returns
        -------
        dict
            Filtered data.
        """
        return self._data_filtered

    def data_inverted(self) -> dict:
        """
        Returns the inverted data.

        Returns
        -------
        dict
            Inverted data.
        """
        return self._data_inverted

    def _check_data(self, data_dictionary: dict) -> None:
        """
        This function checks if the data dictionary contains the necessary keys and columns.
        The necessary keys are:
        - 'current', 'tloop', 'turn', 'timerange', 'filter', 'rloop', 'x', 'y', 'z'
        The necessary columns are:
        - 'Time', 'E/I[V/A]', 'Err[V/A]'

        Parameters
        ----------
        data_dictionary : dict
            Dictionary containing the data and metadata.

        Returns
        -------
        None.

        Raises
        ------
        KeyError
            If the data dictionary does not contain the necessary keys or columns.
        """
        metadata_key = ['current', 'tloop', 'turn', 'timerange', 'filter', 'rloop', 'x', 'y', 'z']
        data_columns = ['Time', 'E/I[V/A]', 'Err[V/A]']

        for value in tqdm(data_dictionary.values(), desc='Checking data', unit='sounding'):
            if value.get('data') is None:
                self.logger.error('Data not found in dictionary.')
                raise KeyError('Data not found in dictionary.')
            elif value.get('metadata') is None:
                self.logger.error('Metadata not found in dictionary.')
                raise KeyError('Metadata not found in dictionary.')
            elif not self._is_sublist(mainlist=list(value.get('metadata').keys()), sublist=metadata_key):
                self.logger.error('Metadata keys are missing.')
                raise KeyError('Metadata keys are missing.')
            elif not self._is_sublist(mainlist=list(value.get('data').columns), sublist=data_columns):
                self.logger.error('Data columns are missing.')
                raise KeyError('Data columns are missing.')

        self.logger.info('Data check successful.')

    def data_read(self, data: [Path, str] = None) -> None:
        """
        This function reads the raw data from the given file path.
        If no file path is given, it tries to read the raw data from the directory structure.

        Parameters
        ----------
        data : Path or str, optional
            File path to the raw data file. The default is None.

        Returns
        -------
        None.
        """
        data_raw = self._folder_structure.get('data_raw')
        
        if data is None:
            if data_raw.exists():
                raw_paths = [path for path in data_raw.iterdir() if path.is_file()]
                if len(raw_paths) > 1:
                    self.logger.warning('Multiple raw data files found. Using the first one.')
                if len(raw_paths) == 0:
                    self.logger.error('No raw data files found.')
                else:
                    data_path = raw_paths[0]
                    self._data_raw = GPfile().read(file_path=data_path)
                    self._data_raw_path = data_path
        else:
            data = Path(data)
            if data.exists():
                new_data = data_raw / data.name
                self._gp_folder.move_files(from_path=data, to_path=data_raw)
                self._data_raw = GPfile().read(file_path=new_data)
                self._data_raw_path = data
            else:
                self.logger.warning('Data file not found. Tries to read the data from the directory structure.')
                if data is not None:
                    self.data_read(data=None)

    def data_rename_columns(self, rename_dict: dict) -> Optional[dict]:
        """
        This function renames the columns of the data dictionary.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing the old and new column names.

        Returns
        -------
        dict
            Dictionary containing the data and metadata with the renamed columns.
        """
        new_data_dict = self._data_raw.copy()

        if new_data_dict is not None:
            for key, value in tqdm(new_data_dict.items(), desc='Renaming columns', unit='sounding'):
                df_data = value.pop('data')
                df_data.rename(columns=rename_dict, inplace=True)
                new_data_dict[key]['data'] = df_data
            return new_data_dict
        else:
            self.logger.error('data_rename_columns: No data found.')
            raise KeyError('data_rename_columns: No data found.')


    def _add_coords_to_data(self, data_dict: dict, parsing_dict: dict = None) -> Optional[dict]:
        """
        This function adds the coordinates to the metadata of the data dictionary.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data and metadata.
        parsing_dict : dict, optional
            Dictionary to parse the keys of the data dictionary. The default is None.
            This parameter can be used if the keys of the data dictionary are not the same as the coordinates keys.
            Or if multiple measurements should have the same coordinates. Example:
            {'T001': 'Mtest', 'T002': 'Mtest'} -> T001 and T002 will both be assigned the coordinates of Mtest.

        Returns
        -------
        dict
            Dictionary containing the data and metadata with the coordinates added.
        """
        if self._coordinates_grouped is None:
            self.logger.error('_add_coords_to_data: No coordinates found. Run coords_extract_save() first.')
            raise KeyError('_add_coords_to_data: No coordinates found.')

        new_data_dict = data_dict.copy()

        if new_data_dict is not None:
            for key, value in tqdm(new_data_dict.items(), desc='Adding coordinates to data', unit='sounding'):
                if parsing_dict is not None:
                    coords_key = parsing_dict.get(key, key)
                else:
                    coords_key = key

                # coordinates = self._coordinates_grouped.get(coords_key, {}).get(('x', 'y', 'z'), (0, 0, 0))
                value['metadata']['x'] = self._coordinates_grouped.get(coords_key, {}).get('x', 0)
                value['metadata']['y'] = self._coordinates_grouped.get(coords_key, {}).get('y', 0)
                value['metadata']['z'] = self._coordinates_grouped.get(coords_key, {}).get('z', 0)
            return new_data_dict
        else:
            self.logger.error('_add_coords_to_data: No data found.')
            raise KeyError('_add_coords_to_data: No data found.')

    def _normalize_rhoa(self, data_dict: dict) -> Optional[dict]:
        """
        This function normalizes the data dictionary and calculates the apparent resistivity and conductivity.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data and metadata.

        Returns
        -------
        dict
            Dictionary containing the normalized data and metadata.
        """
        norm_data_dict = data_dict.copy()

        if norm_data_dict is not None:
            for key, value in tqdm(norm_data_dict.items(), desc='Normalizing data', unit='sounding'):
                df_data = value.get('data')

                if 'rhoa' in df_data.columns:
                    norm_data_dict[key]['data'] = df_data
                else:
                    df_metadata = value.get('metadata')
                    current = df_metadata.get('current')
                    tloop = df_metadata.get('tloop')
                    turn = df_metadata.get('turn')

                    df_data['E/I[V/A]'] = current * df_data['E/I[V/A]'] / (tloop ** 2 * turn)
                    df_data['Err[V/A]'] = current * df_data['Err[V/A]'] / (tloop ** 2 * turn)
                    df_data['Time'] = df_data['Time'] / 1e6
                    mag_momen = turn * current * tloop ** 2
                    df_data['rhoa'] = 1 / np.pi * (mag_momen / (20 * df_data['E/I[V/A]'])) ** (2 / 3) * (self._mu / df_data['Time']) ** (5 / 3)
                    df_data['sigma'] = 1 / (df_data['rhoa'])
                    norm_data_dict[key]['data'] = df_data

            return norm_data_dict
        else:
            self.logger.error('_normalize_rhoa: No data found.')
            raise KeyError('_normalize_rhoa: No data found.')

    def data_preprocess(self, data_dict: Optional[dict] = None, parsing_dict: dict = None) -> None:
        """
        This function preprocesses the data dictionary. If no dictionary is given, it uses the raw data.
        It adds the coordinates to the metadata, normalizes the data, and calculates the apparent resistivity and conductivity.
        It also saves the preprocessed data if not done already.

        Parameters
        ----------
        data_dict : dict, optional
            Dictionary containing the data and metadata. The default is None.
        parsing_dict : dict, optional
            Dictionary to parse the keys of the data dictionary. The default is None.
            This parameter can be used if the keys of the data dictionary are not the same as the coordinates keys.
            Or if multiple measurements should have the same coordinates. Example:
            {'T001': 'Mtest', 'T002': 'Mtest'} -> T001 and T002 will both be assigned the coordinates of Mtest.

        Returns
        -------
        None.
        """
        preproc_path = self._folder_structure.get('data_preproc')

        if data_dict is None:
            data_dict = self._data_raw.copy()
            data_path = preproc_path / f'{self._data_raw_path.stem}_proc{self._data_raw_path.suffix}'
        else:
            data_path = preproc_path / f'preprocessed_data.tem'

        if data_path.exists():
            self.logger.info('data_preprocess: Preprocessed data already exists. Reading file.')
            self._data_preprocessed = GPfile().read(file_path=data_path)
            self._data_preprocessed_path = data_path
        else:
            self.logger.info('data_preprocess: Preprocessing data.')
            self._check_data(data_dictionary=data_dict)
            data_dict = self._add_coords_to_data(data_dict=data_dict, parsing_dict=parsing_dict)
            data_dict = self._normalize_rhoa(data_dict=data_dict)

            self._data_preprocessed = data_dict
            self._data_preprocessed_path = data_path
            GPfile().write(data=data_dict, file_path=data_path)

    def _filter_sounding(self, data_dict: dict,
                         filter_times: Tuple[Union[float, int], Union[float, int]],
                         noise_floor: [float, int]) -> dict:
        """
        This function filters one sounding based on the given time range and noise floor.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data and metadata.
        filter_times : Tuple[Union[float, int], Union[float, int]]
            Tuple containing the start and end time for the filter.
        noise_floor : [float, int]
            Noise floor for the relative error.

        Returns
        -------
        dict
            Dictionary containing the filtered data and metadata.
        """
        filtered_dict = data_dict.copy()
        df = data_dict.get('data')
        if df is None:
            self.logger.error('No data found.')
            raise KeyError('No data found.')

        mask_time = (df['Time'] > filter_times[0] / 1e6) & (df['Time'] < filter_times[1] / 1e6)

        df_f0 = df[mask_time]
        df_f0['rel_err'] = abs(df_f0['Err[V/A]'].values) / df_f0['E/I[V/A]'].values  # observed relative error

        df_f0.loc[df_f0['rel_err'] < noise_floor, 'rel_err'] = noise_floor

        # Find the index of the first negative value
        negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index
        while not negative_indices.empty:

            first_negative_index = int(negative_indices.min())

            if not pd.isnull(first_negative_index):
                # Find the index again to ensure its validity after potential DataFrame modifications
                negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index
                first_negative_index = negative_indices.min()
                # Determine if the negative value is in the first half or second half
                if first_negative_index < len(df_f0) / 2:
                    # Delete all rows before the first negative value
                    df_f0 = df_f0[df_f0.index >= first_negative_index + 1]
                    negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index
                elif first_negative_index > len(df_f0) / 2:
                    # Delete all rows after the first negative value
                    df_f0 = df_f0[df_f0.index < first_negative_index]
                    negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index

        df_f1 = df_f0.copy()

        # Find first lower index
        diff = df_f1['E/I[V/A]'] - df_f1['Err[V/A]']
        lower_indices = diff[diff <= 0].index

        # Truncate the DataFrame if such a condition exists
        if not lower_indices.empty:

            first_lower_index = int(lower_indices.min())
            if not pd.isnull(first_lower_index):
                df_f1 = df_f1.loc[:first_lower_index-1]

        df_f2 = df_f1.copy()

        df_f2.reset_index(drop=True, inplace=True)
        df_f2['rel_err'] = abs(df_f2['Err[V/A]'].values) / df_f2['E/I[V/A]'].values  # observed relative error

        df_f2.loc[df_f2['rel_err'] < noise_floor, 'rel_err'] = noise_floor

        filtered_dict['data'] = df_f2
        filtered_dict['metadata']['name'] = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        return filtered_dict

    def data_filter(self, filter_times: Tuple[Union[float, int], Union[float, int]] = (7, 700),
                    noise_floor: [float, int] = 0.025,
                    subset: list = None) -> None:
        """
        This function filters the data dictionary based on the given time range and noise floor.
        If no subset is given, it filters all soundings.

        Parameters
        ----------
        filter_times : Tuple[Union[float, int], Union[float, int]], optional
            Tuple containing the start and end time for the filter. The default is (7, 700).
        noise_floor : [float, int], optional
            Noise floor for the relative error. The default is 0.025.
        subset : list, optional
            List of keys to filter. The default is None.

        Returns
        -------
        None.
        """
        if subset is None:
            subset = list(self._data_preprocessed.keys())
        else:
            subset = [key for key in subset if key in self._data_preprocessed.keys()]
            invalid_subset = [key for key in subset if key not in self._data_preprocessed.keys()]
            if invalid_subset:
                self.logger.warning(f'Invalid subset keys: {invalid_subset}')

        filter_dir_path = self._folder_structure.get('data_filtered')
        filter_key = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        if self._data_filtered is None:
            self._data_filtered = {}

        for key in subset:
            file_path_filtered = filter_dir_path / f'{key}.tem'

            if file_path_filtered.exists():
                data_filtered = GPfile().read(file_path=file_path_filtered, verbose=False)

                if data_filtered.get(filter_key) is None:
                    data = self._data_preprocessed.get(key)
                    data_filtered[filter_key] = self._filter_sounding(data_dict=data, filter_times=filter_times,
                                                                      noise_floor=noise_floor)
                    GPfile().write(data=data_filtered, file_path=file_path_filtered, verbose=False)

            else:
                data_filtered = {}
                data = self._data_preprocessed.get(key)
                data_filtered[filter_key] = self._filter_sounding(data_dict=data, filter_times=filter_times,
                                                                  noise_floor=noise_floor)
                GPfile().write(data=data_filtered, file_path=file_path_filtered, verbose=False)

            if self._data_filtered.get(key) is None:
                self._data_filtered[key] = {}
            self._data_filtered[key][filter_key] = data_filtered[filter_key]

    def _inversion_sounding(self, data_dict: dict,
                            depth_vector: np.ndarray,
                            inversion_key: str,
                            start_model: np.ndarray = None,
                            verbose: bool = True) -> dict:
        filtered_data = data_dict.get('data')
        filtered_rhoa = filtered_data['rhoa'].values
        filtered_signal = filtered_data['E/I[V/A]'].values
        filtered_relerr = filtered_data['rel_err'].values
        # # testing
        # filtered_relerr = np.full_like(filtered_relerr, 0.05)
        filtered_time = filtered_data['Time'].values

        filtered_metadata = data_dict.get('metadata')
        tloop = filtered_metadata.get('tloop')
        rloop = filtered_metadata.get('rloop')
        current = filtered_metadata.get('current')
        turn = filtered_metadata.get('turn')
        timerange = filtered_metadata.get('timerange')
        filter_pl = filtered_metadata.get('filter')

        split_inversion_key = [float(i) for i in inversion_key.split('_')]
        lam, filter_min, filter_max = split_inversion_key

        if start_model is None:
            rhoa_median = np.round(np.median(filtered_rhoa), 4)
            start_model = np.full_like(depth_vector, rhoa_median)

        setup_device = {
            "timekey": timerange,
            "currentkey": np.round(current),
            "txloop": tloop,
            "rxloop": rloop,
            "current_inj": current,
            "filter_powerline": filter_pl
        }

        # setup inv class and calculate response of homogeneous model
        tem_inv = tem_inv_smooth1D(setup_device=setup_device)

        self.test_resp = tem_inv.prepare_fwd(depth_vector=depth_vector,
                                start_model=start_model,
                                times_rx=filtered_time)
        tem_inv.prepare_inv(maxIter=20, verbose=verbose)  # prepare the inversion, keep the kwargs like this

        # start inversion
        res_mdld = tem_inv.run(dataVals=filtered_signal, errorVals=filtered_relerr,
                               startModel=start_model, lam=lam)

        resp_sgnl = tem_inv.response
        thks = np.diff(tem_inv.depth_fixed)  # convert depths to layer thicknesses
        chi2 = tem_inv.chi2()
        rrms = tem_inv.relrms()
        absrms = tem_inv.absrms()
        phi_model = tem_inv.phiModel()
        mag_momen = turn * current * tloop ** 2
        response_rhoa = 1 / np.pi * (mag_momen / (20 * resp_sgnl)) ** (2 / 3) * (self._mu / filtered_time) ** (
                5 / 3)

        inversion_df = pd.DataFrame()
        max_length = max(len(res_mdld), len(resp_sgnl), len(thks))
        inversion_df = inversion_df.reindex(range(max_length))

        inversion_df['depth_vector'] = pd.Series(depth_vector)
        inversion_df['start_model'] = pd.Series(start_model)
        inversion_df['resistivity_model'] = pd.Series(res_mdld)
        inversion_df['conductivity_model'] = pd.Series(1 / res_mdld)
        inversion_df['E/I[V/A]'] = pd.Series(resp_sgnl)
        inversion_df['modelled_thickness'] = pd.Series(thks)
        inversion_df['rhoa'] = pd.Series(response_rhoa)
        inversion_df['sigma'] = pd.Series(1 / response_rhoa)

        inversion_metadata = {
            'lambda': lam,
            'filtertime_min': filter_min,
            'filtertime_max': filter_max,
            'chi2': chi2,
            'relrms': rrms,
            'absrms': absrms,
            'phi_model': phi_model
        }

        return {'data': inversion_df, 'metadata': inversion_metadata}

    def _forward_sounding(self, data_dict: dict,
                            depth_vector: np.ndarray,
                            start_model: np.ndarray = None) -> pd.DataFrame:
        filtered_data = data_dict.get('data')
        filtered_rhoa = filtered_data['rhoa'].values
        filtered_time = filtered_data['Time'].values

        filtered_metadata = data_dict.get('metadata')
        tloop = filtered_metadata.get('tloop')
        rloop = filtered_metadata.get('rloop')
        current = filtered_metadata.get('current')
        turn = filtered_metadata.get('turn')
        timerange = filtered_metadata.get('timerange')
        filter_pl = filtered_metadata.get('filter')

        if start_model is None:
            rhoa_median = np.round(np.median(filtered_rhoa), 4)
            start_model = np.full_like(depth_vector, rhoa_median)

        setup_device = {
            "timekey": timerange,
            "currentkey": np.round(current),
            "txloop": tloop,
            "rxloop": rloop,
            "current_inj": current,
            "filter_powerline": filter_pl
        }

        # setup inv class and calculate response of homogeneous model
        tem_inv = tem_inv_smooth1D(setup_device=setup_device)
        test_resp = tem_inv.prepare_fwd(depth_vector=depth_vector,
                                start_model=start_model,
                                times_rx=filtered_time)

        mag_momen = turn * current * tloop ** 2
        response_rhoa = 1 / np.pi * (mag_momen / (20 * test_resp)) ** (2 / 3) * (self._mu / filtered_time) ** (
                5 / 3)

        df = pd.DataFrame()
        max_length = max(len(depth_vector), len(response_rhoa))
        df = df.reindex(range(max_length))

        df['depth_vector'] = pd.Series(depth_vector)
        df['start_model'] = pd.Series(start_model)
        df['E/I[V/A]'] = pd.Series(test_resp)
        df['rhoa'] = pd.Series(response_rhoa)
        df['sigma'] = pd.Series(1 / response_rhoa)
        df['Time'] = pd.Series(filtered_time)

        return df

    def data_inversion(self, lam: [int, float] = 600,
                       layer_type: str = 'linear',
                       layers: [int, float, dict, np.ndarray] = 4.5,
                       max_depth: [float, int] = None,
                       filter_times: Tuple[Union[float, int], Union[float, int]] = (7, 700),
                       start_model: np.ndarray = None,
                       noise_floor: [float, int] = 0.025,
                       subset: list = None,
                       verbose: bool = True) -> None:

        if subset is None:
            subset = list(self._data_preprocessed.keys())
        else:
            subset = [key for key in subset if key in self._data_preprocessed.keys()]
            invalid_subset = [key for key in subset if key not in self._data_preprocessed.keys()]
            if invalid_subset:
                self.logger.warning(f'Invalid subset keys: {invalid_subset}')

        self.data_filter(filter_times=filter_times, noise_floor=noise_floor, subset=subset)

        inversion_dir_path = self._folder_structure.get('data_inversion')
        inversion_key = f'{lam}_{filter_times[0]}_{filter_times[1]}'
        filter_key = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'

        if self._data_inverted is None:
            self._data_inverted = {}

        if layer_type == 'linear':
            if verbose:
                self.logger.info(f'inversion: Inversion with linear layer thickness. Layer thickness: {layers}.')
            depth_vector = np.arange(0, max_depth, step=layers)

        elif layer_type == 'log':
            if verbose:
                self.logger.info(f'inversion: Inversion with logarithmic layer thickness. Number of layers: {layers}.')
            depth_vector = np.logspace(-1, np.log10(max_depth + 0.1), round(layers)) - 0.1

        elif layer_type == 'dict':
            if not isinstance(layers, dict):
                if verbose:
                    self.logger.error('inversion: layers must be a dictionary.')
                raise TypeError('Layers must be a dictionary.')
            if verbose:
                self.logger.info(f'inversion: Inversion with layer thicknesses extracted from the layers dict.')

            if all(key < max_depth for key in layers.keys()):
                lay_keys = sorted(list(layers.keys()))
            else:
                lay_keys = sorted([key for key in layers.keys() if key < max_depth])
            lay_keys.append(max_depth)

            layer_list = [lay_keys[0]]
            cur_depth = lay_keys[0]
            for i in range(len(lay_keys) - 1):
                while cur_depth <= lay_keys[i + 1]:
                    cur_depth += layers[lay_keys[i]]
                    if cur_depth <= max_depth:
                        layer_list.append(cur_depth)
            depth_vector = np.array(layer_list)

        elif layer_type == 'custom':
            if not isinstance(layers, np.ndarray):
                if verbose:
                    self.logger.error('inversion: layers must be an numpy array.')
                raise TypeError('Layers must be an numpy array.')
            if verbose:
                self.logger.info('inversion: Inversion with custom layer thicknesses. layers was read as the depth vector.')
            depth_vector = layers

        else:
            if verbose:
                self.logger.error(f'inversion: {layer_type} is an unknown keyword.')
            raise KeyError(f'{layer_type} is an unknown keyword.')


        for key in subset:
            file_path_inversion = inversion_dir_path / f'{key}.tim'
            data_filtered = self._data_filtered.get(key).get(filter_key)
            filtered_df = data_filtered.get('data')
            filtered_metadata = data_filtered.get('metadata')
            tloop = filtered_metadata.get('tloop')
            turn = filtered_metadata.get('turn')
            current = filtered_metadata.get('current')

            if max_depth is None:
                max_depth = np.round(np.sqrt(tloop ** 2 * turn * current), 2)
                # max_depth = 4 * tloop todo: which is correct, ours or this? 3 (Adrian) or 4 (Lukas)?

            if start_model is None:
                rhoa_median = np.round(np.median(filtered_df['rhoa']), 4)
                start_model = np.full_like(depth_vector, rhoa_median)

            inv_key = None
            if file_path_inversion.exists():
                data_inversion = GPfile().read(file_path=file_path_inversion, verbose=False)
                key_list = [key for key in data_inversion.keys() if key.startswith(inversion_key)]

                data_missing = True
                if key_list:
                    for inv_key in key_list:
                        found_data = data_inversion.get(inv_key)
                        found_df = found_data.get('data')
                        found_start_model = found_df['start_model'].values
                        found_depth_vector = found_df['depth_vector'].values

                        if depth_vector.size != found_depth_vector.size:
                            depth_same = False
                        else:
                            depth_same = np.allclose(depth_vector, found_depth_vector, atol=1e-5)
                        if start_model.size != found_start_model.size:
                            start_same = False
                        else:
                            start_same = np.allclose(start_model, found_start_model, atol=1e-5)

                        if start_same and depth_same:
                            if verbose:
                                self.logger.info(f'inversion: Inversion already exists. Added {inversion_key} to {key}.')
                            data_missing = False
                            break

                if data_missing:
                    if key_list:
                        inv_key = f'{inversion_key}_{len(key_list) + 1}'
                    else:
                        inv_key = f'{inversion_key}_1'

            else:
                data_inversion = {}
                inv_key = f'{inversion_key}_1'
                data_missing = True

            if data_missing:
                data_inversion[inv_key] = self._inversion_sounding(data_dict=data_filtered,
                                                                   depth_vector=depth_vector,
                                                                   inversion_key=inversion_key,
                                                                   start_model=start_model,
                                                                   verbose=verbose)
                data_inversion[inv_key]['metadata']['name'] = inv_key
                data_inversion[inv_key]['metadata']['max_depth'] = max_depth
                GPfile().write(data=data_inversion, file_path=file_path_inversion, verbose=False)

            if self._data_inverted.get(key) is None:
                self._data_inverted[key] = {}
            self._data_inverted[key][inversion_key] = data_inversion.get(inv_key)

    def data_forward(self,
                       layer_type: str = 'linear',
                       layers: [int, float, dict, np.ndarray] = 4.5,
                       max_depth: [float, int] = None,
                       filter_times: Tuple[Union[float, int], Union[float, int]] = (5, 1000),
                       start_model: np.ndarray = None,
                       noise_floor: [float, int] = 0.025,
                       subset: list = None,
                       verbose: bool = True) -> None:

        if subset is None:
            subset = list(self._data_preprocessed.keys())
        else:
            subset = [key for key in subset if key in self._data_preprocessed.keys()]
            invalid_subset = [key for key in subset if key not in self._data_preprocessed.keys()]
            if invalid_subset:
                self.logger.warning(f'Invalid subset keys: {invalid_subset}')

        self.data_filter(filter_times=filter_times, noise_floor=noise_floor, subset=subset)
        filter_key = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'

        for key in subset:
            data_filtered = self._data_filtered.get(key).get(filter_key)
            filtered_df = data_filtered.get('data')
            filtered_metadata = data_filtered.get('metadata')
            tloop = filtered_metadata.get('tloop')
            turn = filtered_metadata.get('turn')
            current = filtered_metadata.get('current')

            if max_depth is None:
                max_depth = np.round(np.sqrt(tloop ** 2 * turn * current), 2)
                # max_depth = 4 * tloop todo: which is correct, ours or this? 3 (Adrian) or 4 (Lukas)?

            if layer_type == 'linear':
                if verbose:
                    self.logger.info(f'inversion: Inversion with linear layer thickness. Layer thickness: {layers}.')
                depth_vector = np.arange(0, max_depth, step=layers)

            elif layer_type == 'log':
                if verbose:
                    self.logger.info(
                        f'inversion: Inversion with logarithmic layer thickness. Number of layers: {layers}.')
                depth_vector = np.logspace(-1, np.log10(max_depth + 0.1), round(layers)) - 0.1

            elif layer_type == 'dict':
                if not isinstance(layers, dict):
                    if verbose:
                        self.logger.error('inversion: layers must be a dictionary.')
                    raise TypeError('Layers must be a dictionary.')
                if verbose:
                    self.logger.info(f'inversion: Inversion with layer thicknesses extracted from the layers dict.')

                if all(key < max_depth for key in layers.keys()):
                    lay_keys = sorted(list(layers.keys()))
                else:
                    lay_keys = sorted([key for key in layers.keys() if key < max_depth])
                lay_keys.append(max_depth)

                layer_list = [lay_keys[0]]
                cur_depth = lay_keys[0]
                for i in range(len(lay_keys) - 1):
                    while cur_depth <= lay_keys[i + 1]:
                        cur_depth += layers[lay_keys[i]]
                        if cur_depth <= max_depth:
                            layer_list.append(cur_depth)
                depth_vector = np.array(layer_list)

            elif layer_type == 'custom':
                if not isinstance(layers, np.ndarray):
                    if verbose:
                        self.logger.error('inversion: layers must be an numpy array.')
                    raise TypeError('Layers must be an numpy array.')
                if verbose:
                    self.logger.info(
                        'inversion: Inversion with custom layer thicknesses. layers was read as the depth vector.')
                depth_vector = layers

            else:
                if verbose:
                    self.logger.error(f'inversion: {layer_type} is an unknown keyword.')
                raise KeyError(f'{layer_type} is an unknown keyword.')

            if start_model is None:
                rhoa_median = np.round(np.median(filtered_df['rhoa']), 4)
                start_model = np.full_like(depth_vector, rhoa_median)

            if self._data_modelled is None:
                self._data_modelled = {}

            self._data_modelled[key] = self._forward_sounding(data_dict=data_filtered,
                                                               depth_vector=depth_vector,
                                                               start_model=start_model)


    def choose_from_csv(self, filepath: Path, chosen_points: tuple = (), line_name: str = '') -> list[Any]:
        filepath = Path(filepath)
        target_folder = self._folder_structure.get('coordinates_choose') #
        if target_folder is None:
            raise ValueError('No target folder for coordinates found.')
        new_file = target_folder / filepath.name #saving new filepath based on current filepath and folder from class structure
        self._gp_folder.move_files(from_path=filepath, to_path=target_folder)

        #todo: was passiert ab hier?
        tem_line_table = pd.read_csv(new_file, sep=';') #reading the csv file from the tem line
        if not chosen_points:
            chosen_points = list(tem_line_table['Name']) #if points are given as chosen, they are used, else the points from the line are used
        cur_all = self.chosen_data.get('all', []) #what has previously been chosen is loaded
        cur_all.extend(chosen_points) #the newly chosen points are added to the previously chosen points
        self.chosen_data['all'] = cur_all #updating the chosen list of all points
        self.chosen_data[line_name] = chosen_points #updating the chosen list of points for the current line
        #todo: log entry for chosen data - ja, hab log noch nicht verstanden
        return chosen_points

    # def inversion_chosen(self, from_csv:Path=None,
    #                      chosen_points:tuple=(),
    #                      subset:list=None, unit='rhoa',
    #                      lam=600,
    #                      layer_type='linear',
    #                      layers=4.5,
    #                      max_depth=None,
    #                      filter_times=(7, 700)):
    #
    #     if from_csv is not None:
    #         inversion_list = self.choose_from_csv(from_csv, chosen_points)
    #     else:
    #         inversion_list = self.chosen_data.get('all')
    #
    #     if subset is not None:
    #         self.chosen_data['subset'] = subset
    #         inversion_list = subset
    #     #todo: log entry for inverted data
    #     self.plot_inversion(subset=inversion_list, lam=lam, layer_type=layer_type, unit=unit, layers=layers, max_depth=max_depth, filter_times=filter_times)

    @staticmethod
    def visualize_sounding(raw_dict, filtered_dict, sounding_name, color, plot='raw', unit='rhoa', scale='log', fig=None, ax1=None, ax2=None, legend=True):
        # set plot parameters
        alpha = 1

        if scale == 'log':
            change_scale = True
        elif scale == 'lin':
            change_scale = False
        else:
            raise SyntaxError('input {} not valid for Argument scale'.format(scale))

        if unit == 'rhoa':
            unit_name = 'Apparent Resistivity'
            unit_label = r'$\rho_a$ [$\Omega$m]'
        elif unit == 'sigma':
            unit_name = 'Apparent Conductivity'
            unit_label = r'$\sigma_a$ [mS/m]'
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        raw_data = raw_dict.get('data')
        filtered_data = filtered_dict.get('data')

        if plot == 'raw' and raw_data is not None:
            xaxis = raw_data['Time']
            yaxis1 = raw_data['E/I[V/A]']
            yaxis2 = raw_data[unit]
            line = 'solid'
            col = color
            zorder = 3
            label = None
            title_name = 'Raw Data'
            name_1 = 'Signal of Measurement'
        elif plot == 'err' and raw_data is not None:
            xaxis = raw_data['Time']
            yaxis1 = raw_data['Err[V/A]']
            yaxis2 = None  # raw_data[unit]
            alpha = .4
            line = 'dashed'
            col = '#808080'
            zorder = 2
            label = None
            title_name = 'Data Error'
            name_1 = 'Noise Level'
        elif plot == 'filtered' and filtered_data is not None:
            xaxis = filtered_data['Time']
            yaxis1 = filtered_data['E/I[V/A]']
            yaxis2 = filtered_data[unit]
            col = color
            label = sounding_name
            line = 'solid'
            zorder = 4
            title_name = 'Filtered Data'
            name_1 = 'Signal of Measurement'
        elif plot == 'raw_grey' and raw_data is not None:
            xaxis = raw_data['Time']
            yaxis1 = raw_data['E/I[V/A]']
            yaxis2 = raw_data[unit]
            col = '#808080'
            line = 'solid'
            label = None
            zorder = 1
            alpha = .3
            title_name = 'Raw Data as Background'
            name_1 = 'Signal of Measurement'
        else:
            raise SyntaxError('input {} not valid for Argument plot'.format(plot))

        # check if fig, ax1, ax2 are given and if necessary creating them
        if fig is None and ax1 is None and ax2 is None:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax1, ax2 = axs[0], axs[1]
            ax1.set_title(name_1, fontsize=16)
            ax2.set_title(unit_name, fontsize=16)
            plt.suptitle('Plotting of the {}'.format(title_name),
                         fontsize=20, fontweight='bold')
            plt.tight_layout()

        elif fig is None or ax1 is None or ax2 is None:
            raise SyntaxError('not all necessary values for fig, ax1, ax2 were given and neither all all empty')

        ax1.set_xlabel('Time [s]', fontsize=16)
        ax1.set_ylabel(r'$\partial B_z/\partial t$ [V/m²]', fontsize=16)
        ax1.plot(xaxis, yaxis1, label=label, alpha=alpha, color=col, zorder=zorder, marker='.', linestyle=line)
        if change_scale:
            ax1.loglog()

        if ax1.get_legend_handles_labels()[1] and legend:  # Prüft, ob Labels vorhanden sind
            ax1.legend(loc='lower left', fontsize=12)
        ax1.grid(True, which="both", alpha=.3)

        ax2.set_xlabel('Time [s]', fontsize=16)
        ax2.set_ylabel(unit_label, fontsize=16)
        if yaxis2 is not None:
            ax2.plot(xaxis, yaxis2, alpha=alpha, color=col, zorder=zorder, marker='.', linestyle=line)
        if change_scale:
            ax2.loglog()
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.grid(True, which="both", alpha=.3)
        return fig

    def plot_raw_filtered(self, subset: list = None, unit: str = 'rhoa', scale: str = 'log',
                          filter_times: Tuple[Union[int, float], Union[int, float]] = (7, 700),
                          noise_floor: [int, float] = 0.025, legend=True):

        plot_list = [point for point in self._data_raw.keys() if subset is None or point in subset]
        self.data_filter(subset=subset, filter_times=filter_times, noise_floor=noise_floor)



        fig, axs = plt.subplots(2, 2, figsize=(13, 13))
        ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(plot_list))]

        for key, col in zip(plot_list, colors):
            raw_data = self._data_preprocessed[key]
            filtered_data = self._data_filtered[key][f'{filter_times[0]}_{filter_times[1]}_{noise_floor}']
            # Top row:
            # plot raw
            self.visualize_sounding(sounding_name=key, raw_dict=raw_data, filtered_dict=filtered_data,
                                    color=col, plot='raw', unit=unit, scale=scale,
                                    fig=fig, ax1=ax1, ax2=ax2, legend=legend)
            # plot error
            self.visualize_sounding(sounding_name=key, raw_dict=raw_data, filtered_dict=filtered_data,
                                    color=col, plot='err', unit=unit, scale=scale,
                                    fig=fig, ax1=ax1, ax2=ax2, legend=legend)

            # Bottom row
            # plot filtered
            self.visualize_sounding(sounding_name=key, raw_dict=raw_data, filtered_dict=filtered_data,
                                    color=col, plot='filtered', unit=unit, scale=scale,
                                    fig=fig, ax1=ax3, ax2=ax4, legend=legend)

            self.visualize_sounding(sounding_name=key, raw_dict=raw_data, filtered_dict=filtered_data,
                                    color=col, plot='err', unit=unit, scale=scale,
                                    fig=fig, ax1=ax3, ax2=ax4, legend=legend)
            self.visualize_sounding(sounding_name=key, raw_dict=raw_data, filtered_dict=filtered_data,
                                    color=col, plot='raw_grey', unit=unit, scale=scale,
                                    fig=fig, ax1=ax3, ax2=ax4, legend=legend)

        if unit == 'rhoa':
            unit_label = 'Apparent Resistivity'
        else:
            unit_label = 'Apparent Conductivity'

        for ax, label, title in zip([ax1, ax2, ax3, ax4], ['(a)', '(b)', '(c)', '(d)'],
                                    ['Raw Impulse Response', 'Raw {}'.format(unit_label), 'Filtered Impulse Response',
                                     'Filtered {}'.format(unit_label)]):
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.text(0.96, 0.08, label, transform=ax.transAxes, fontsize=18, zorder=5,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='xkcd:light grey', boxstyle='round,pad=0.5'))
            ax.set_title(title, fontsize=20, fontweight='bold')

        plt.tight_layout()
        fig.show()
        target_dir = self._folder_structure.get('data_first_look')
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.savefig(target_dir / f'raw_filtered_{time}_{filter_times[0]}_{filter_times[1]}_{unit}.png')

    def plot_forward_model(self, subset: list = None, unit: str = 'rhoa', scale: str = 'log',
                          filter_times: Tuple[Union[int, float], Union[int, float]] = (5, 1000),
                          layer_type = 'linear', layers = 1, max_depth = None,
                           start_model = None, verbose=True, legend=False):

        plot_list = [point for point in self._data_raw.keys() if subset is None or point in subset]
        self.data_forward(layer_type=layer_type, layers=layers, max_depth=max_depth, filter_times=filter_times,
                          start_model=start_model, subset=subset, verbose=verbose)

        fig, axs = plt.subplots(1, 2, figsize=(13, 8))
        ax1, ax2 = axs[0], axs[1]

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(plot_list))]

        if scale == 'log':
            change_scale = True
        elif scale == 'lin':
            change_scale = False
        else:
            raise SyntaxError('input {} not valid for Argument scale'.format(scale))

        if unit == 'rhoa':
            unit_label = r'$\rho_a$ [$\Omega$m]'
        elif unit == 'sigma':
            unit_label = r'$\sigma_a$ [mS/m]'
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        for key, col in zip(plot_list, colors):
            model_data = self._data_modelled[key]
            label = key if legend else None
            xaxis = model_data['Time']
            yaxis1 = model_data['E/I[V/A]']
            yaxis2 = model_data[unit]

            ax1.set_xlabel('Time [s]', fontsize=16)
            ax1.set_ylabel(r'$\partial B_z/\partial t$ [V/m²]', fontsize=16)
            ax1.plot(xaxis, yaxis1, label=label, color=col, marker='.')
            if change_scale:
                ax1.loglog()

            if ax1.get_legend_handles_labels()[1] and legend:  # Prüft, ob Labels vorhanden sind
                ax1.legend(loc='lower left', fontsize=12)
            ax1.grid(True, which="both", alpha=.3)

            ax2.set_xlabel('Time [s]', fontsize=16)
            ax2.set_ylabel(unit_label, fontsize=16)
            ax2.plot(xaxis, yaxis2, color=col, marker='.')
            if change_scale:
                ax2.loglog()
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.grid(True, which="both", alpha=.3)

        if unit == 'rhoa':
            unit_label = 'Apparent Resistivity'
        else:
            unit_label = 'Apparent Conductivity'

        for ax, label, title in zip([ax1, ax2], ['(a)', '(b)'],
                                    ['Modelled Impulse Response', 'Modelled {}'.format(unit_label)]):
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.text(0.96, 0.08, label, transform=ax.transAxes, fontsize=18, zorder=5,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='xkcd:light grey', boxstyle='round,pad=0.5'))
            ax.set_title(title, fontsize=20, fontweight='bold')

        plt.tight_layout()
        fig.show()
        target_dir = self._folder_structure.get('data_first_look')
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.savefig(target_dir / f'forward_model_{time}_{filter_times[0]}_{filter_times[1]}_{unit}.png')

    def _plot_one_inversion(self, dict_key,
                           lam: [int, float] = 600,
                           filter_times: Tuple[Union[float, int], Union[float, int]] = (7, 700),
                           noise_floor: [int, float] = 0.025,
                           layer_type: str = 'linear',
                           layers: [int, float, dict, np.ndarray] = 4.5,
                           unit: str = 'rhoa'):

        inv_name = f'{lam}_{filter_times[0]}_{filter_times[1]}'
        filter_name = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        inverted_data = self._data_inverted.get(dict_key, {}).get(inv_name)
        filtered_data = self._data_filtered.get(dict_key, {}).get(filter_name)

        if inverted_data is None:
            self.logger.error(f'No inversion data found for {dict_key}.')
            return

        if filtered_data is None:
            self.logger.error(f'No filtered data found for {dict_key}.')
            return

        filtered_data = filtered_data.get('data')
        inversion_data = inverted_data.get('data')
        inversion_metadata = inverted_data.get('metadata')

        obs_unit = filtered_data[unit]
        response_unit = inversion_data[unit].dropna()
        thks = inversion_data['modelled_thickness'].dropna()
        resp_sgnl = inversion_data['E/I[V/A]'].dropna()
        chi2 = inversion_metadata.get('chi2')
        rrms = inversion_metadata.get('relrms')

        if unit == 'rhoa':
            unit_label_ax = r'$\rho_a$ [$\Omega$m]'
            unit_title = 'Apparent Resistivity'
            unit_title_mod = 'Resistivity'
            unit_label_mod = r'$\rho$ [$\Omega$m]'
            model_unit = inversion_data['resistivity_model'].dropna()
            pos_1 = 'right'
            pos_2 = 'left'
        else:
            unit_label_ax = r'$\sigma_a$ [S/m]'
            unit_title = 'Apparent Conductivity'
            unit_title_mod = 'Conductivity'
            unit_label_mod = r'$\sigma$ [S/m]'
            model_unit = inversion_data['conductivity_model'].dropna()
            pos_1 = 'left'
            pos_2 = 'right'


        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        ax[0].loglog(filtered_data['Time'], resp_sgnl, '-k', label='inversion', zorder=3)
        ax[0].plot(filtered_data['Time'], filtered_data['E/I[V/A]'], marker='v', label='observed', zorder=2) #color=self.col,
        ax[0].plot(filtered_data['Time'], filtered_data['Err[V/A]'], label='error', zorder=1, alpha=0.4, linestyle='dashed') #color=self.col,
        ax[0].set_xlabel('time [s]', fontsize=16)
        ax[0].set_ylabel(r'$\partial B_z/\partial t$ [V/m²]', fontsize=16)
        ax[0].grid(True, which="both", alpha=.3)

        ax[1].plot(filtered_data['Time'], response_unit, '-k', label='inversion', zorder=3)
        ax[1].plot(filtered_data['Time'], obs_unit, marker='v', label='observed', zorder=2) #color=self.col,
        ax[1].set_xlabel('time [s]', fontsize=16)
        ax[1].set_ylabel(unit_label_ax, fontsize=16)
        ax[1].set_xscale('log')
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position("right")
        ax[1].grid(True, which="both", alpha=.3)

        drawModel1D(ax[2], thks, model_unit, color='k', label='pyGIMLI')
        ax[2].set_xlabel(unit_label_mod, fontsize=16)
        ax[2].set_ylabel('depth [m]', fontsize=16)
        ax[2].yaxis.tick_right()
        ax[2].yaxis.set_label_position("right")

        for a, title, pos in zip(ax, ['Impulse Response', unit_title, 'Model of {} at Depth'.format(unit_title_mod)], ['lower left', 'lower {}'.format(pos_1), 'lower {}'.format(pos_2)]):
            a.legend(loc=pos)
            a.set_title(title, fontsize=18, pad=12)
            a.tick_params(axis='both', which='major', labelsize=14)

        if layer_type == 'linear':
            fig.suptitle(f'Lambda = {lam:<8.0f} Layer Thickness = {layers:<.2f}m\n\u03C7\u00B2 = {chi2:<8.2f} Relative RMS = {rrms:<.2f}%', fontsize=22, fontweight='bold')
        else:
            fig.suptitle(f'Lambda = {lam:<8.0f} Layer Thickness = {layer_type}\n\u03C7\u00B2 = {chi2:<8.2f} Relative RMS = {rrms:<.2f}%', fontsize=22, fontweight='bold')

        fig.show()

        target_dir = self._folder_structure.get('data_inversion_plot')
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.savefig(target_dir / f'{dict_key}_{time}_{unit}.png')

    def plot_inversion(self, subset:list=None,
                       lam: [int, float] = 600,
                       layer_type: str = 'linear',
                       layers: [int, float, dict, np.ndarray] = 4.5,
                       max_depth: [float, int] = None,
                       start_model: np.ndarray = None,
                       noise_floor: [float, int] = 0.025,
                       unit: str = 'rhoa',
                       filter_times=(7, 700),
                       verbose: bool = True):

        plot_list = [point for point in self._data_preprocessed.keys() if subset is None or point in subset]

        self.data_inversion(subset=plot_list, lam=lam, layer_type=layer_type, layers=layers,
                            max_depth=max_depth, filter_times=filter_times,
                            start_model=start_model, noise_floor=noise_floor,
                            verbose=verbose)

        for key in plot_list:
            self._plot_one_inversion(dict_key=key,
                                     lam=lam,
                                     filter_times=filter_times,
                                     noise_floor=noise_floor,
                                     layer_type=layer_type,
                                     layers=layers,
                                     unit=unit)

    @staticmethod
    def _find_inflection_index(xvalues: [np.ndarray, list], yvalues: [np.ndarray, list]) -> list:
        x_array = np.array(xvalues)
        y_array = np.array(yvalues)
        first_diff = np.diff(y_array) / np.diff(x_array)
        second_diff = np.diff(first_diff) / np.diff(x_array[:-1])
        inflection_index = np.argmin(np.abs(second_diff[:-1] + second_diff[1:]))
        # inflection_value_index = xvalues[inflection_indices]
        return [inflection_index, first_diff, second_diff]

    def _find_inflection_point(self, xvalues: [np.ndarray, list], yvalues: [np.ndarray, list]) -> tuple:
        inflection_index, first_diff, second_diff = self._find_inflection_index(xvalues, yvalues)
        start_x = xvalues[inflection_index]
        start_y = second_diff[inflection_index]
        end_x = xvalues[inflection_index + 1]
        end_y = second_diff[inflection_index + 1]
        y = abs(start_y) + abs(end_y)
        x = end_x - start_x  # x goes along the x-Axis, here only the difference between start and end is needed
        inflection_x = start_y / y * x + start_x
        return inflection_x, 0

    def analyse_inversion(self, sounding: str, layers, max_depth: float, plot_id: str = None, test_range:tuple=(10, 10000, 30), layer_type:str = 'linear', filter_times=(7, 700)):
        #todo: make plotting more efficient and neat ... (@jakob)
        #computing relevant values for plotting
        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        roughness_values = []
        rms_values = []
        for lam in lambda_values:
            self.data_inversion(subset=[sounding], lam=lam, layer_type=layer_type, layers=layers, verbose=False,
                                max_depth=max_depth, filter_times=filter_times, noise_floor=0.025, start_model=None)

            inversion_dict = self._data_inverted.get(sounding, {}).get(f'{lam}_{filter_times[0]}_{filter_times[1]}')
            rms_values.append(inversion_dict.get('metadata').get('absrms')) #todo: change labels
            roughness_values.append(inversion_dict.get('metadata').get('phi_model'))

        #computing inflection
        inflection_lambda_index, first_diff_lam, second_diff_lam = self._find_inflection_index(lambda_values,
                                                                                               rms_values)
        inflection_roughness_index, first_diff_phi, second_diff_phi  = self._find_inflection_index(roughness_values,
                                                                                                   rms_values)
        inflection_lam_phi_index, first_diff_lam_phi, second_diff_lam_phi = self._find_inflection_index(lambda_values,
                                                                                                        roughness_values)


        def plot_analysis(sounding:str, plot_id:str, lambda_values:np.array, roughness_values:list, rms_values:list, test_range:tuple=(10, 10000, 30), filter_times=(7, 700)):
            fig, axs = plt.subplots(1, 3, figsize=(24, 8))
            ax1, ax2, ax3 = axs[0], axs[1], axs[2]
            fig.suptitle(f'Analysis of Inversion Results\n Lambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}', fontsize=14)
            ax1.set_title(f'Data plotting, Inflection at {lambda_values[inflection_lambda_index]}')
            ax1.set_ylabel('relRMS (%)')
            ax1.set_xlabel('Lambda')
            ax1.scatter(lambda_values, rms_values)
            ax1.axvline(x=lambda_values[inflection_lambda_index], color='r', linestyle='--')
            ax1.set_ylim(0, max(rms_values) * 1.1)


            ax2.set_title('First Derivative')
            ax2.set_ylabel('first derivative')
            ax2.set_xlabel('Lambda')
            ax2.scatter(lambda_values[:-1], first_diff_lam)
            ax2.axvline(x=lambda_values[inflection_lambda_index], color='r', linestyle='--')
            ax2.axvline(x=self._find_inflection_point(lambda_values, rms_values)[0], color='b', linestyle='--')
            ax2.axhline(y=first_diff_lam[inflection_lambda_index], color='r', linestyle='--')
            ax2.plot(lambda_values[inflection_lambda_index], first_diff_lam[inflection_lambda_index], marker='o', color='g')

            ax3.set_title('Second Derivative')
            ax3.set_ylabel('second derivative')
            ax3.set_xlabel('Lambda')
            ax3.axhline(y=0, color='r', linestyle='-')
            ax3.scatter(lambda_values[:-2], second_diff_lam)
            ax3.axvline(x=lambda_values[inflection_lambda_index], color='r', linestyle='--')
            ax3.axvline(x=self._find_inflection_point(lambda_values, rms_values)[0], color='b', linestyle='--')

            fig.tight_layout()
            fig.show()
            fig.savefig(self._folder_structure.get('data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_lambda_rrms_analysis.png')

            fig2, axs2 = plt.subplots(1, 3, figsize=(24, 8))
            ax4, ax5, ax6 = axs2[0], axs2[1], axs2[2]
            fig2.suptitle(
                f'Analysis of Inversion Results\nLambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}, roughness and relRMS plotted',
                fontsize=14)
            ax4.set_title('Data plotting')
            ax4.set_xlabel('roughness')
            ax4.set_ylabel('relRMS (%)')
            ax4.scatter(roughness_values, rms_values)
            ax4.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax4.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax4.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax4.set_ylim(0, max(rms_values) * 1.1)

            ax5.set_title('Data plotting')
            ax5.set_xlabel('roughness')
            ax5.set_ylabel('first diff')
            ax5.scatter(roughness_values[:-1],first_diff_phi)
            ax5.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax5.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax5.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')

            ax6.set_title('Data plotting')
            ax6.set_xlabel('roughness')
            ax6.set_ylabel('second diff')
            ax6.scatter(roughness_values[:-2], second_diff_phi)
            ax6.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax6.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax6.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax6.axhline(y=0, color='r', linestyle='-')

            fig2.tight_layout()
            fig2.show()
            fig2.savefig(self._folder_structure.get('data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_phi_rrms_analysis.png')

            fig3, axs3 = plt.subplots(1, 3, figsize=(24, 8))
            ax7, ax8, ax9 = axs3[0], axs3[1], axs3[2]
            fig3.suptitle(f'Analysis of Inversion Results\nLambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}, lambda and roughness plotted', fontsize=14)
            ax7.set_title('Data plotting')
            ax7.set_ylabel('Roughness')
            ax7.set_xlabel('Lambda')
            ax7.scatter(lambda_values, roughness_values)
            ax7.axvline(x=lambda_values[inflection_lam_phi_index], color='r', linestyle='--')
            ax7.set_ylim(0, max(roughness_values) * 1.1)

            ax8.set_title('First Derivative')
            ax8.set_ylabel('first derivative')
            ax8.set_xlabel('Lambda')
            ax8.scatter(lambda_values[:-1], first_diff_lam_phi)
            ax8.axvline(x=lambda_values[inflection_lam_phi_index], color='r', linestyle='--')
            ax8.axvline(x=self._find_inflection_point(lambda_values, roughness_values)[0], color='b', linestyle='--')
            ax8.axhline(y=first_diff_lam[inflection_lam_phi_index], color='r', linestyle='--')
            ax8.plot(lambda_values[inflection_lam_phi_index], first_diff_lam[inflection_lam_phi_index], marker='o', color='g')

            ax9.set_title('Second Derivative')
            ax9.set_ylabel('second derivative')
            ax9.set_xlabel('Lambda')
            ax9.axhline(y=0, color='r', linestyle='-')
            ax9.scatter(lambda_values[:-2], second_diff_lam_phi)
            ax9.axvline(x=lambda_values[inflection_lam_phi_index], color='r', linestyle='--')
            ax9.axvline(x=self._find_inflection_point(lambda_values, roughness_values)[0], color='b', linestyle='--')

            fig3.tight_layout()
            fig3.show()
            fig3.savefig(self._folder_structure.get('data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_lambda_phi_analysis.png')

            fig4, axs4 = plt.subplots(1, 3, figsize=(24, 8))
            ax10, ax11, ax12 = axs4[0], axs4[1], axs4[2]
            fig4.suptitle(
                f'Analysis of Inversion Results\nLambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}, roughness and relRMS plotted',
                fontsize=14)
            ax10.set_title('Data plotting')
            ax10.set_xlabel('roughness')
            ax10.set_ylabel('relRMS (%)')
            ax10.scatter(roughness_values, rms_values, c='b')
            ax10.scatter(roughness_values, np.negative(rms_values), c='r')
            ax10.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax10.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax10.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax10.loglog()

            ax11.set_title('Data plotting')
            ax11.set_xlabel('roughness')
            ax11.set_ylabel('first diff')
            ax11.scatter(roughness_values[:-1], first_diff_phi, c='b')
            ax11.scatter(roughness_values[:-1], np.negative(first_diff_phi), c='r')
            ax11.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax11.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax11.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax11.loglog()

            ax12.set_title('Data plotting')
            ax12.set_xlabel('roughness')
            ax12.set_ylabel('second diff')
            ax12.scatter(roughness_values[:-2], second_diff_phi, c='b')
            ax12.scatter(roughness_values[:-2], np.negative(second_diff_phi), c='r')
            ax12.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax12.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax12.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax12.axhline(y=0, color='r', linestyle='-')
            ax12.loglog()

            fig4.tight_layout()
            fig4.show()
            fig4.savefig(self._folder_structure.get(
                'data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_phi_rms_log_analysis.png')


        plot_analysis(sounding, plot_id, lambda_values, roughness_values, rms_values, test_range=test_range, filter_times=filter_times)

    def analyse_inversion2(self, sounding: str,
                           layers,
                           max_depth: float,
                           test_range:tuple=(10, 10000, 30),
                           layer_type:str = 'linear',
                           filter_times=(7, 700)):

        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        roughness_values = []
        rms_values = []

        for lam in lambda_values:
            self.data_inversion(subset=[sounding], lam=lam, layer_type=layer_type, layers=layers,
                                verbose=False, max_depth=max_depth, filter_times=filter_times,
                                noise_floor=0.025, start_model=None)

            inversion_dict = self._data_inverted.get(sounding, {}).get(f'{lam}_{filter_times[0]}_{filter_times[1]}')
            rms_values.append(inversion_dict.get('metadata').get('absrms'))
            roughness_values.append(inversion_dict.get('metadata').get('phi_model'))


        roughness_values = np.array(roughness_values)
        rms_values = np.array(rms_values)
        lambda_values = np.array(lambda_values)

        # roughness_values = np.log10(roughness_values)
        # rms_values = np.log10(rms_values)

        # sorted_indices = np.argsort(roughness_values)
        # roughness_values = roughness_values[sorted_indices]
        # rms_values = rms_values[sorted_indices]
        # lambda_values = lambda_values[sorted_indices]

        # cs = CubicSpline(roughness_values, rms_values)
        # cs_d = cs.derivative(nu=1)
        # cs_dd = cs.derivative(nu=2)
        #
        # first_derivative = cs_d(roughness_values)
        # second_derivative = cs_dd(roughness_values)
        #
        # curvature_values = second_derivative / (1 + first_derivative ** 2) ** (3 / 2)
        # max_curvature_index = np.argmax(curvature_values)
        # opt_lambda = lambda_values[max_curvature_index]
        #
        # max_curvature_roughness = roughness_values[max_curvature_index]
        #
        # x_new = np.linspace(roughness_values.min(), roughness_values.max(), 500)
        # y_new = cs(x_new)
        # curvature_new = cs_dd(x_new) / (1 + cs_d(x_new) ** 2) ** (3 / 2)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(roughness_values, rms_values, 'o', label='Original data')
        for i in np.linspace(0, len(lambda_values) - 1, 5).astype(int):
            ax[0].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], rms_values[i]), fontsize=8, ha='right',
                           textcoords="offset points", xytext=(10, 10))
        # ax[0].plot(x_new, y_new, '-', label='Cubic spline fit')
        # ax[0].vlines(max_curvature_roughness, rms_values.min(), rms_values.max(), colors='r', linestyles='--',
        #              label='Optimum at lambda = {:.2f}'.format(opt_lambda))
        ax[0].set_xlabel('log10(roughness)')
        ax[0].set_ylabel('log10(rms)')
        ax[0].set_title('RMS/Smoothness curve')
        ax[0].legend()

        # ax[1].plot(roughness_values, curvature_values, 'o', label='Curvature')
        # for i in np.linspace(0, len(lambda_values) - 1, 5).astype(int):
        #     ax[1].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], curvature_values[i]), fontsize=8, ha='right',
        #                    textcoords="offset points", xytext=(10, 10))
        # ax[1].plot(x_new, curvature_new, '-', label='Curvature fit')
        # ax[1].vlines(max_curvature_roughness, curvature_values.min(), curvature_values.max(), colors='r', linestyles='--',
        #              label='Max curvature at lambda = {:.2f}'.format(opt_lambda))
        ax[1].set_xlabel('log10(roughness)')
        ax[1].set_ylabel('Curvature')
        ax[1].set_title('Curvature of the RMS/Smoothness curve')

        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.tight_layout()
        fig.show()
        fig.savefig(self._folder_structure.get(
            'data_inversion_analysis') / f'lambda_analysis_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png')

        return None #opt_lambda

    def analyse_inversion_plot(self, sounding: str,
                           layers,
                           max_depth: float,
                           test_range:tuple=(10, 10000, 30),
                           layer_type:str = 'linear',
                           filter_times=(7, 700),
                           noise_floor: [int, float] = 0.025,
                           unit: str = 'rhoa'):

        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])

        nrows = np.ceil(len(lambda_values) / 5).astype(int)

        fig1, ax1 = plt.subplots(nrows, 5, figsize=(15, 3 * nrows))
        ax1 = ax1.ravel()

        fig2, ax2 = plt.subplots(nrows, 5, figsize=(15, 3 * nrows))
        ax2 = ax2.ravel()

        for i, lam in enumerate(lambda_values):
            self.data_inversion(subset=[sounding], lam=lam, layer_type=layer_type, layers=layers,
                                verbose=False, max_depth=max_depth, filter_times=filter_times,
                                noise_floor=0.025, start_model=None)

            inv_name = f'{lam}_{filter_times[0]}_{filter_times[1]}'
            filter_name = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
            inverted_data = self._data_inverted.get(sounding, {}).get(inv_name)
            filtered = self._data_filtered.get(sounding, {}).get(filter_name)

            filtered_data = filtered.get('data')
            inversion_data = inverted_data.get('data')

            obs_unit = filtered_data[unit]
            response_unit = inversion_data[unit].dropna()
            resp_sgnl = inversion_data['E/I[V/A]'].dropna()

            if unit == 'rhoa':
                unit_label_ax = r'$\rho_a$ [$\Omega$m]'
            else:
                unit_label_ax = r'$\sigma_a$ [S/m]'

            ax1[i].loglog(filtered_data['Time'], resp_sgnl, '-k', label='inversion', zorder=3)
            ax1[i].plot(filtered_data['Time'], filtered_data['E/I[V/A]'], marker='v', label='observed',
                       zorder=2)
            ax1[i].plot(filtered_data['Time'], filtered_data['Err[V/A]'], label='error', zorder=1, alpha=0.4,
                       linestyle='dashed')
            ax1[i].set_xlabel('time [s]', fontsize=16)
            ax1[i].set_ylabel(r'$\partial B_z/\partial t$ [V/m²]', fontsize=16)
            ax1[i].grid(True, which="both", alpha=.3)

            ax2[i].plot(filtered_data['Time'], response_unit, '-k', label='inversion', zorder=3)
            ax2[i].plot(filtered_data['Time'], obs_unit, marker='v', label='observed', zorder=2)
            ax2[i].set_title(f'{lam: .2f}', fontweight='bold', fontsize=16)
            ax2[i].set_xlabel('time [s]', fontsize=16)
            ax2[i].set_ylabel(unit_label_ax, fontsize=16)
            ax2[i].set_xscale('log')
            ax2[i].yaxis.tick_right()
            ax2[i].yaxis.set_label_position("right")
            ax2[i].grid(True, which="both", alpha=.3)

        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig1.tight_layout()
        fig1.show()
        fig1.savefig(self._folder_structure.get(
            'data_inversion_analysis') / f'lambda_analysis_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}_signal.png')

        fig2.tight_layout()
        fig2.show()
        fig2.savefig(self._folder_structure.get(
            'data_inversion_analysis') / f'lambda_analysis_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}_{unit}.png')

    #todo: implement the 2D-parts (@jakob)
    def inversion_plot_2D_section(self, unit='rhoa', lam=600, lay_thk=3, save=True, max_depth=50):
        fig, ax = plt.subplots(figsize=(10, 4))
        if unit == 'rhoa':
            unit_label = r'$\rho$ [$\Omega$m]'
        elif unit == 'sigma_a':
            unit_label = r'$\sigma$ [mS/m]'
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        all_xcoords = []
        all_ycoords = []
        all_values = []
        if self.inversion_results is not None:
            for inv_res, lmnt in zip(self.inversion_results, self.list_files):
                thks, res_mdld = inv_res
                xcoord = np.full(len(res_mdld), lmnt.posX)
                ycoord = np.arange(thks[0], thks[0] + len(res_mdld) * thks[0], thks[0])
                all_xcoords.append(xcoord)
                all_ycoords.append(ycoord)
                all_values.append(res_mdld)
        else:
            for lmnt in self.list_files:
                _, thks, res_mdld, _, _, _, _, _, _, _ = lmnt.inversion(lam=lam, lay_thk=lay_thk)
                xcoord = np.full(len(res_mdld), lmnt.posX)
                ycoord = np.arange(thks[0], thks[0] + len(res_mdld) * thks[0], thks[0])
                all_xcoords.append(xcoord)
                all_ycoords.append(ycoord)
                all_values.append(res_mdld)

        # Concatenate all x, y, and value arrays
        xcoords = np.concatenate(all_xcoords)
        ycoords = np.concatenate(all_ycoords)
        values = np.concatenate(all_values)
        unit_values = values if unit == 'rhoa' else 1000 / values

        sc = ax.scatter(xcoords, ycoords, c=unit_values, cmap='viridis', marker='o',
                        s=60)  # You can choose any colormap you like
        ax.invert_yaxis()
        ax.set_xlabel('Distance along profile [m]', fontsize=16)
        ax.set_ylabel('Depth [m]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(max_depth + 2)

        # Customize colorbar to match the data values
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_ticks(np.linspace(np.min(unit_values), np.max(unit_values), num=6),
                       fontsize=14)  # Set ticks according to data range
        cbar.set_label(unit_label, fontsize=16)  # Set colorbar label
        fig.suptitle('Lambda = {:<8.0f} Layer Thickness = {:<.2f}m'.format(lam, lay_thk), fontsize=20,
                     fontweight='bold')
        plt.tight_layout()

        if save:
            fig.savefig(self.path_plot_2D + 'inversion_2Dsection_{}.png'.format(unit))

        self.section_coords = (xcoords, ycoords, values)
        return (xcoords, ycoords, values)

    def interpolation(self, lam=600, lay_thk=3):
        if self.section_coords is not None:
            xcoords_concat, ycoords_concat, values_concat = self.section_coords
        else:
            xcoords_concat, ycoords_concat, values_concat = self.inversion_plot_2D_section(lam=lam, lay_thk=lay_thk,
                                                                                           save=False)

        # Define grid for interpolation
        xi = np.linspace(xcoords_concat.min(), xcoords_concat.max(), 140)
        yi = np.linspace(ycoords_concat.min(), ycoords_concat.max(), 75)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate values
        zi = griddata((xcoords_concat, ycoords_concat), values_concat, (xi, yi),
                      method='linear')  # linear, nearest, cubic

        self.interpolation_res = (xi, yi, zi, xcoords_concat, ycoords_concat, values_concat)
        return (xi, yi, zi, xcoords_concat, ycoords_concat, values_concat)

    def plot_inversion_interpolated(self, unit='rhoa', lam=600, lay_thk=4.5, save=True, max_depth=50):
        fig, ax = plt.subplots(figsize=(10, 4))
        if unit == 'rhoa':
            unit_label = r'$\rho$ [$\Omega$m]'
        elif unit == 'sigma_a':
            unit_label = r'$\sigma$ [mS/m]'
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        if self.interpolation_res is not None:
            xi, yi, zi, _, _, _ = self.interpolation_res
            value_i = zi if unit == 'rhoa' else 1000 / zi
        else:
            xi, yi, zi, _, _, _ = self.interpolation(lam=lam, lay_thk=lay_thk)
            value_i = zi if unit == 'rhoa' else 1000 / zi

        sc = ax.scatter(xi, yi, c=value_i, cmap='viridis', label='Interpolated Points', marker='s', s=7)
        ax.invert_yaxis()
        ax.set_xlabel('Distance along profile [m]', fontsize=16)
        ax.set_ylabel('Depth [m]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(max_depth)

        # Customize colorbar to match the data values
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_ticks(np.linspace(np.min(value_i), np.max(value_i), num=6),
                       fontsize=14)  # Set ticks according to data range
        cbar.set_label(unit_label, fontsize=14)  # Set colorbar label
        fig.suptitle('Lambda = {:<8.0f} Layer Thickness = {:<.2f}m'.format(lam, lay_thk), fontsize=20,
                     fontweight='bold')
        plt.tight_layout()

        if save:
            fig.savefig(self.path_plot_2D + 'inversion_2Dsection_interpolated_{}.png'.format(unit))
        return fig