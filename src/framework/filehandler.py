# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:38:41 2024

@author: peter
"""

#%% Import modules

import re
import yaml
from pathlib import Path
import logging
import pandas as pd
from gp_package.framework.coordinatehandler import CoordinateHandler
from tqdm import tqdm

#%% Aufgaben

#todo: check for duplicates
#todo: deal with ert-naming
#todo: deal with roll alongs -> merging
#todo: change dump function...filepath instead of directory_key


#todo: add create_preproc
#todo: gather_files -> search subdirs (needed?)
#todo: does it make sense to redo the whole class? -> GPfile for reading one file, GPfolder for creating the folder structure

#%% FileHandler class

class TestFileHandler(CoordinateHandler):
    def __init__(self, basedir: [str, Path], dir_template: [str, Path], template_path: [str, Path] = None):
        self.templates_dir = {}
        self.templates_file = {}
        self.folder_structure = None
        self.sub_structure = False
        self.paths_dict = {}
        self._coords = False

        self.coordinates_dict = {}
        self.preproc_dict = {}
        self.data_dict = {}

        self.basedir = Path(basedir)
        default_template_path = Path(__file__).parents[1] / 'templates'
        self.template_path = Path(template_path) if template_path is not None else default_template_path
        self.create_folders(template=dir_template)

        self.logger = self._setup_logger()
        super().__init__(basedir=basedir, log_filename=self.paths_dict.get('log'))
        self.logger.info('Setting up FileHandler')
        self.logger.info('Loaded templates')
        self.logger.info('Created folder structure')

    def _load_templates(self):
        """
        Loads templates from template_path into two dictionaries for directory structure and file structure.
        """
        if self.template_path.is_dir():
            templates_dir = [(file.name.replace('_dir.yml', ''), yaml.safe_load(file.read_text())) for file in
                             self.template_path.iterdir() if file.is_file() and file.name.endswith('_dir.yml')]
            self.templates_dir = {temp.get('name', name): temp for name, temp in templates_dir}

            templates_file = [(file.name.replace('_file.yml', ''), yaml.safe_load(file.read_text())) for file in
                              self.template_path.iterdir() if file.is_file() and file.name.endswith('file.yml')]
            if len(templates_file) == 0:
                raise FileNotFoundError('No templates for file structure found.')
            self.templates_file = {temp.get('config', {}).get('name', name): temp for name, temp in
                                   templates_file}
        else:
            raise FileNotFoundError('Path must be a directory.')

    def add_file_template(self, new_template):
        """Adds a new file template to the file template dictionary."""
        self.logger.info('add_file_templates: Adding file template')
        self._load_templates()
        new_template = Path(new_template)
        if new_template.is_file():
            new_name = new_template.name.replace('.yml', '')
            new_yaml = yaml.safe_load(new_template.read_text())
            self.templates_file[new_yaml.get('config', {}).get('prefix', new_name)] = new_template
            self.logger.info('add_file_templates: Added file template')

    def create_folders(self, template):
        self._load_templates()
        basedir = self.basedir
        template_dict = self.templates_dir.get(template)
        if template_dict is None:
            raise KeyError('create_folders: No template was found.')

        main_structure = template_dict.get('main_template')
        if main_structure is None:
            raise KeyError('create_folders: Main template was not found.')
        main_structure = {key: basedir / value for key, value in main_structure.items()}
        if main_structure.get('log') is not None:
            self.paths_dict['log'] = main_structure.get('log') / template_dict.get('log_file', 'logfile.txt')

        for value in main_structure.values():
            value.mkdir(parents=True, exist_ok=True)

        self.folder_structure = main_structure

    def create_subdirs(self, directory, template='default'):
        template_dict = self.templates_dir.get(template)
        if template_dict is None:
            self.logger.error('create_subdirs: No template was found.')
            raise KeyError

        sub_structure = template_dict.get('sub_template')
        if sub_structure is None:
            self.logger.error('create_subdirs: Sub template was not found.')
            raise KeyError
        sub_structure = {key: directory / value for key, value in sub_structure.items()}

        for value in sub_structure.values():
            value.mkdir(parents=True, exist_ok=True)

        return sub_structure

    def _move_files(self, data_raw, coords, preproc):
        if coords is not None:
            coord_path = Path(coords)
            if coord_path.exists():
                if coord_path.is_file():
                    new_coordinate_path = self.folder_structure.get('coordinates_raw') / coord_path.name
                    self.paths_dict['coordinates_raw'] = coord_path.rename(new_coordinate_path)
                else:
                    self.logger.error('_move_files: Coordinate-file path was not a file.')
                    raise ValueError('_move_files: Coordinate-file path was not a file.')
            else:
                self.logger.warning('_move_files: No coordinate-file was found.')

        if preproc is not None:
            preproc_path = Path(preproc)
            if preproc_path.exists():
                if preproc_path.is_file():
                    new_preproc_path = self.folder_structure.get('preproc') / preproc_path.name
                    self.paths_dict['preproc'] = preproc_path.rename(new_preproc_path)
                else:
                    self.logger.error('_move_files: Preprocessing-file path was not a file.')
                    raise ValueError('_move_files: Preprocessing-file path was not a file.')
            else:
                self.logger.warning('_move_files: No preprocessing-file was found.')

        if data_raw is not None:
            data_path = Path(data_raw)
            if data_path.exists():
                if data_path.is_file():
                    new_data_path = self.folder_structure.get('data_raw') / data_path.name
                    self.paths_dict['data_raw'] = data_path.rename(new_data_path)
                elif data_path.is_dir():
                    for file in data_path.iterdir():
                        new_data_path = self.folder_structure.get('data_raw') / file.name
                        _ = file.rename(new_data_path)
                    self.paths_dict['data_raw'] = self.folder_structure.get('data_raw')
                else:
                    self.logger.error('_move_files: Data path was neither a file nor a directory.')
                    raise ValueError('_move_files: Invalid data_path. Must be a file or directory.')
            else:
                self.logger.warning('_move_files: No data was found. Might have been moved already')

    def gather_files(self):
        for key, value in self.folder_structure.items():
            if self.paths_dict.get(key) is None:
                value = Path(value)
                if value.exists() and value.is_dir():
                    files = [file for file in value.iterdir() if file.is_file()]
                    if len(files) == 1:
                        self.paths_dict[key] = files[0]
                    elif len(files) > 1:
                        self.paths_dict[key] = value

    def _check_extension(self, file_path: [str, Path]):
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f'_check_extension: {file_path} does not exist.')
            raise FileNotFoundError(f'_check_extension: {file_path} does not exist.')
        else:
            if file_path.is_file():
                extensions = file_path.suffix
            elif file_path.is_dir():
                extensions = {file.suffix for file in file_path.iterdir() if file.is_file()}
                if len(extensions) != 1:
                    self.logger.error(f'_load_filepaths: Files have different extensions: {extensions}')
                    raise ValueError('All files must have the same extension.')
                else:
                    extensions = next(iter(extensions))
            else:
                self.logger.error(f'_check_extension: {file_path} is neither a file nor a directory.')
                raise ValueError(f'_check_extension: {file_path} is neither a file nor a directory.')
        return extensions

    def choose_file_type(self, file_path, chosen_template=None):
        """Automatically or manually chooses file template for the data files."""
        suffix = self._check_extension(file_path=file_path)

        if chosen_template is not None:
            file_type = self.templates_file.get(chosen_template, None)
            if file_type is None:
                self.logger.error('choose_file_template: Chosen file type was not found')
                raise FileNotFoundError
            elif file_type.get('config').get('file_suffix') != suffix:
                self.logger.error('choose_file_template: File suffix does not match')
                raise ValueError('choose_file_template: File suffix does not match')
            else:
                return file_type
        else:
            parsing_dict = {value.get('config').get('file_suffix'): key for key, value in self.templates_file.items()}
            type_name = parsing_dict.get(suffix)
            file_type = self.templates_file.get(type_name, None)
            if file_type is None:
                self.logger.error(
                    f'choose_file_template: File type was not found in {self.template_path}. Try add_template.')
                raise FileNotFoundError
            return file_type

    def _load_preproc(self):
        self.logger.info('_load_preproc: Loading preprocessing template.')
        self.gather_files()

        preproc = self.paths_dict.get('preproc')
        if preproc is not None:
            self.preproc_dict = yaml.safe_load(preproc.read_text())
            self.logger.info('_load_preproc: Loaded preprocessing template.')
        else:
            self.logger.warning('_load_preproc: No preprocessing template was found. Please create one.')

    @staticmethod
    def create_preproc(**kwargs):
        # todo: implement create_preproc
        print('Function not implemented yet.')

    def _load_coords(self):
        self.logger.info('_load_filepaths: Loading data files')
        self._load_preproc()

        if self.preproc_dict.get('config') is None:
            self.logger.error('_load_filepaths: No preprocessing template was found. Please create one.')
            raise ValueError('_load_filepaths: No preprocessing template was found. Please create one.')

        coords = self.paths_dict.get('coordinates_raw')
        # self.coordinate_handler = CoordinateHandler(basedir=self.basedir, log_filename=self.paths_dict.get('log'))
        if coords is not None:
            sep = self.preproc_dict.get('config', {}).get('sep', ',')
            self.read_coord_file(coords=coords, sep=sep)
            self._coords = True
            self.logger.info('_load_coords: Loaded coordinate files.')
        else:
            self._coords = False

    def add_files(self, data=None, coords=None, preproc=None):
        self.logger.info('read_files: Reading files')
        self._move_files(data_raw=data, coords=coords, preproc=preproc)
        self._load_preproc()
        self.logger.info('read_files: Added files to folderstructure.')

    @staticmethod
    def _find_measurement_blocks(template_dict, lines):
        """Finds all starting indices of the measurement blocks."""
        block_indices = []
        keyword = template_dict.get('config', {}).get('block_start_pattern', None)
        buffer = template_dict.get('config', {}).get('start_buffer', 0)
        if keyword is None:
            block_indices.append(0)
        else:
            for i, line in enumerate(lines):
                if keyword in line:
                    block_indices.append(i + buffer)
        return block_indices

    def parse_dict(self, template_dict, lines, start_index):
        if not template_dict.get('type', None) == 'dict':
            self.logger.error('parse_dict: Invalid type for parse_dict.')
            raise TypeError
        dictionary = {}
        # self.logger.info('parse_dict: Parsing type is dict')
        start_with = template_dict.get('start_with', None)
        if start_with is not None:
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_with in line:
                        break
        else:
            block_start = start_index

        start_after = template_dict.get('start_after', 0)
        block_start += start_after

        template_lines = template_dict.get('lines', [])
        if not template_lines:
            self.logger.error('parse_dict: Template is missing "lines". It is needed for parsing.')
            raise KeyError
        block_end = block_start + len(template_lines)
        block_lines = lines[block_start:block_end]
        for line in block_lines:
            for entry in template_lines:
                match = re.search(entry.get('pattern'), line)
                if match:
                    for i, key in enumerate(entry.get('key', [])):
                        dictionary[key] = match.group(i + 1)

        for key, value in dictionary.items():
            try:
                dictionary[key] = pd.to_numeric(value, errors='ignore')
            except Exception as e:
                self.logger.warning(f'parse_dict: Unexpected Error {e} occurred.')
                continue

        return dictionary

    def parse_dataframe(self, template_dict, lines, start_index):
        if not template_dict.get('type', None) == 'dataframe':
            self.logger.error('parse_dataframe: Invalid type for parse_dataframe.')
            raise TypeError
        # self.logger.info('parse_dataframe: Parsing type is dataframe')
        start_with = template_dict.get('start_with', None)
        if start_with is not None:
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_with in line:
                        break
        else:
            block_start = start_index

        start_after = template_dict.get('start_after', 0)
        block_start += start_after

        end_with = template_dict.get('end_with', None)
        for i, line in enumerate(lines):
            if i >= block_start:
                block_end = i
                if end_with is not None:
                    if end_with in line:
                        break

        end_after = template_dict.get('end_after', 0)
        if end_with is not None:
            block_end += end_after
        block_lines = lines[block_start:block_end + 1]
        raw_columns = block_lines[0]
        raw_measurements = block_lines[1:]
        delimiter = template_dict.get('delimiter', '\t').encode().decode('unicode_escape')
        column_delimiter = template_dict.get('column_delimiter', '\t').encode().decode('unicode_escape')
        column_prefix = template_dict.get('column_prefix', '')
        columns = raw_columns.replace(column_prefix, '').strip().split(column_delimiter)
        columns = [col.strip() for col in columns]
        measurements = [line.strip().split(delimiter) for line in raw_measurements]
        measurements = [[m.strip() for m in line] for line in measurements]

        df = pd.DataFrame(measurements, columns=columns)
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore'))
        return df

    def _type_changer(self, type_string, value):
        value = value.strip()
        if type_string == 'int':
            changed = int(value)
        elif type_string == 'float':
            changed = float(value)
        elif type_string == 'bool':
            changed = bool(value)
        elif type_string == 'str':
            changed = str(value)
        else:
            self.logger.error(f'_type_changer: {type_string} is not a valid type.')
            raise TypeError(f'Invalid type string: {type_string}')
        return changed

    def parse_line(self, template_dict, lines, start_index):
        if not template_dict.get('type', None) == 'line':
            self.logger.error('parse_line: Invalid type for parse_line.')
            raise TypeError

        start_with = template_dict.get('start_with', None)
        if start_with is not None:
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_with in line:
                        break
        else:
            block_start = start_index
        start_after = template_dict.get('start_after', 0)
        block_start = block_start + start_after
        line_number = template_dict.get('lines', 1)
        subtype = template_dict.get('subtype', False)
        if line_number == 1:
            line = lines[block_start]
            if subtype:
                line = self._type_changer(subtype, line)
            return line
        else:
            line_block = lines[block_start: block_start + line_number]
            if subtype:
                line_block = [self._type_changer(subtype, x) for x in line_block]
            return line_block

    def find_parsing_type(self, template_dict, lines, start_index):
        template_type = template_dict.get('type', None)
        if template_type is None:
            self.logger.error('find_parsing_type: Template is missing "type". It is needed for parsing.')
            raise KeyError
        if template_type == 'dict':
            return self.parse_dict(template_dict, lines, start_index)
        elif template_type == 'dataframe':
            return self.parse_dataframe(template_dict, lines, start_index)
        elif template_type == 'line':
            return self.parse_line(template_dict, lines, start_index)
        else:
            self.logger.error('find_parsing_type: Template type was not recognized.')
            raise KeyError

    def _parse_file(self, file_path, file_type=None):
        file_path = Path(file_path)
        self.logger.info(f'_parse_file: Started reading {file_path.name}')

        type_file = self.choose_file_type(file_path=file_path, chosen_template=file_type)
        if type_file is None:
            self.logger.error('_parse_file: No file type was found.')
            raise KeyError('_parse_file: No file type was found.')

        file_template = type_file.get('template')
        if file_template is None:
            self.logger.error('_parse_file: Template is missing "template". It is needed for parsing.')
            raise KeyError('"template" not found in YAML-file.')

        measurements = {}
        temp_keys = []
        for line in file_template.get('metadata', {}).get('lines', []):
            for key in line.get('key', []):
                temp_keys.append(key)
        name_exists = 'name' in temp_keys

        if file_path is not None:
            data = [file_path] if file_path.is_file() else file_path.iterdir()

            for data_file in tqdm(data, desc=f'{file_path.name}', unit='file'):
                file_name = data_file.stem
                with open(data_file, 'r') as file_:
                    lines = file_.readlines()

                block_start_indices = self._find_measurement_blocks(template_dict=type_file, lines=lines)
                multiple_blocks = len(block_start_indices) > 1

                if multiple_blocks and not name_exists:
                    multi_block_dict = {key: [] for key in file_template.keys()}

                for start_index in block_start_indices:
                    block_dict = {}
                    for temp_name, temp in file_template.items():
                        # self.logger.info(f'read_files: Parsing {temp_name}')
                        if multiple_blocks and not name_exists:
                            multi_block_dict[temp_name].append(self.find_parsing_type(temp, lines, start_index))
                        else:
                            block_dict[temp_name] = self.find_parsing_type(temp, lines, start_index)
                    if not multiple_blocks or name_exists:
                        measurement_name = block_dict.get('metadata', {}).get('name', file_name)
                        measurements[measurement_name] = block_dict
                        # self.logger.info(f'read_files: Added {measurement_name} to measurements')
                if multiple_blocks and not name_exists:
                    measurements[file_name] = multi_block_dict
                    # self.logger.info(f'read_files: Added {file_name} to measurements')
        self.logger.info(f'Added {file_path.name} to data_dir')
        return measurements

    def _get_coords(self):
        self._load_coords()
        if self._coords:
            if self.preproc_dict.get('config') is not None:
                config = self.preproc_dict.get('config')
                rename_cols = self.preproc_dict.get('rename_columns')
                rename_points = self.preproc_dict.get('rename_points')

                sep = config.get('sep', ',')
                if rename_cols is not None:
                    self.rename_columns(rename_cols)
                if self.check_cols():
                    raise KeyError('_get_coords: Necessary columns not found.')
                if rename_points is not None:
                    self.rename_points(rename_points)

                reproj_key = config.get('reprojection_key')
                if reproj_key is None:
                    crs_start = config.get('crs_initial')
                    crs_end = config.get('crs_to')
                    if crs_start is None or crs_end is None:
                        reproj_key = 'wgs84_utm33n'
                        self.logger.warning(f'_get_coords:No Reprojection key found: {reproj_key} was used.')
                        self.reproject(reproj_key=reproj_key)
                    else:
                        self.custom_reprojection(from_crs=crs_start, to_crs=crs_end)
                else:
                    self.reproject(reproj_key=reproj_key)

                self.interpolate_points()

                naming_pattern = config.get('naming_pattern', r'([A-Za-z]+)(\d*)_(\d*)')
                self.sort_points(pattern=naming_pattern)
                self.coordinates_dict = self.extract_coords()
                proc_coords_path = Path(self.folder_structure.get('coordinates_proc')) / self.paths_dict.get(
                    'coordinates_raw').name.replace('.', '_proc.')
                self.to_file(path=proc_coords_path, sep=sep)
                self.logger.info('_get_coords: Coordinates extracted.')

        else:
            self.logger.warning('_get_coords: No _coordinates_proc file was found. Create dummy-_coordinates_proc file.')

    def _add_coords(self, template_key):
        self._get_coords()
        data_dir = self.data_dict.get('data_raw')
        template_dict = self.templates_file.get(template_key)
        if template_dict is None:
            self.logger.error('_add_coords: No template was found.')
            return

        metadata = template_dict.get('template', {}).get('metadata') is not None
        coords = template_dict.get('template', {}).get('_coordinates_proc') is not None
        num_elec = template_dict.get('template', {}).get('number_electrodes') is not None
        num_meas = template_dict.get('template', {}).get('number_measurements') is not None

        # todo: work with ert (roll alongs)
        for key, value in tqdm(data_dir.items(), desc='Adding _coordinates_proc'):
            if metadata:
                new_coords = self.coordinates_dict.get(key, {})
                data_dir[key]['metadata']['x'] = new_coords.get('x', 0)
                data_dir[key]['metadata']['y'] = new_coords.get('y', 0)
                data_dir[key]['metadata']['z'] = new_coords.get('z', 0)

            if num_meas:
                data_dir[key]['number_measurements'] = data_dir.get('data', pd.DataFrame()).shape[0]
            if num_elec:
                data_dir[key]['number_electrodes'] = data_dir.get('data', pd.DataFrame()).iloc[:, :4].max().max()
            if coords:
                new_coords = self.coordinates_dict.get(key, None)
                if new_coords is None:
                    elec_num = data_dir.get('data', pd.DataFrame()).iloc[:, :4].max().max()
                    if pd.isna(elec_num):
                        elec_num = 72
                    new_coords = self.create_dummy(elec_sep=1, elec_num=elec_num)

                data_dir[key]['_coordinates_proc'] = new_coords

        self.data_dict['data_preproc'] = data_dir

    def load_data(self, template):
        self.gather_files()
        raw = self.paths_dict.get('data_raw')
        preprocessed = self.paths_dict.get('data_preproc')
        filtered = self.paths_dict.get('data_filtered')

        # print(filtered)
        # raise ValueError('stop here')

        if filtered is not None:
            self.data_dict['data_filtered'] = self._parse_file(file_path=filtered)

        if preprocessed is not None:
            self.data_dict['data_preproc'] = self._parse_file(file_path=preprocessed)

        if raw is not None:
            self.data_dict['data_raw'] = self._parse_file(file_path=raw)
            if preprocessed is None:
                self._add_coords(template_key=template)
                self.dump(directory_key='data_preproc', template_key=template, suffix='_proc')

    @staticmethod
    def write_dict(template_dict, data, file_):
        template_lines = template_dict.get('lines', [])
        for line in template_lines:
            line_keys = line.get('key', [])
            line_values = {key: data.get(key) for key in line_keys}
            write_line = line.get('output_format', '').encode().decode('unicode_escape').format(**line_values)
            file_.write(write_line + '\n')

    @staticmethod
    def write_dataframe(template_dict, data, file_):
        delimiter = template_dict.get('delimiter', '\t').encode().decode('unicode_escape')
        column_delimiter = template_dict.get('column_delimiter', delimiter).encode().decode('unicode_escape')
        column_prefix = template_dict.get('column_prefix', '')
        columns = column_delimiter.join(data.columns)
        file_.write(column_prefix + columns + '\n')

        for index, row in data.iterrows():
            file_.write(f'{delimiter.join(map(str, row))}\n')

    def write_line(self, template_dict, data, file_):
        line_amount = template_dict.get('lines', 1)
        if line_amount == 1:
            file_.write(f'{data}\n')
        else:
            if not len(data) == line_amount:
                self.logger.error(
                    f'write_line: Input dictionary has {len(data)} lines while template allows {line_amount} lines.')
                raise ValueError(f'Input dictionary has {len(data)} lines ({line_amount} needed).')
            for line in data:
                file_.write(f'{line}\n')

    def find_writing_type(self, template_dict, data, file_):
        template_type = template_dict.get('type', None)
        if template_type is None:
            self.logger.error('find_writing_type: Template is missing "type". It is needed for parsing.')
            raise KeyError
        if template_type == 'dict':
            self.write_dict(template_dict, data, file_)
        elif template_type == 'dataframe':
            self.write_dataframe(template_dict, data, file_)
        elif template_type == 'line':
            self.write_line(template_dict, data, file_)
        else:
            self.logger.error('find_writing_type: Template type was not recognized.')
            raise KeyError

    def dump(self, directory_key, template_key, suffix=''):
        template_dict = self.templates_file.get(template_key)
        if template_dict is None:
            self.logger.error('dump: No template was found.')
            return
        file_suffix = template_dict.get('config', {}).get('file_suffix', '')
        multi_file = self.paths_dict.get('data_raw').is_dir()
        data_dict = self.data_dict.get(directory_key)
        target_folder = Path(self.folder_structure.get(directory_key))
        if data_dict is None:
            self.logger.error('dump: No data was found.')
            return

        if not multi_file:
            filename = self.paths_dict.get('data_raw').stem + suffix + file_suffix
            filepath = target_folder / filename
            with open(filepath, 'w') as file_:
                for data in data_dict.values():
                    for key, value in template_dict.get('template').items():
                        data_block = data.get(key)
                        self.find_writing_type(value, data_block, file_)
        else:
            for name, data in data_dict.items():
                filename = self.paths_dict.get('data_raw').stem + suffix + name
                filepath = target_folder / filename
                with open(filepath, 'w') as file_:
                    for key, value in template_dict.get('template').items():
                        data_block = data.get(key)
                        self.find_writing_type(value, data_block, file_)

        self.logger.info(f'dump: Dumped {directory_key} to {target_folder}')

    def close(self):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
                self.logger.info("File handler closed and removed.")
                break

        self.coordinates = None
        self.extracted_coords = None
        self.paths_dict = {}
        self._coords = False
        self.coordinates_dict = {}
        self.preproc_dict = {}
        self.data_dict = {}


#%% NewHandler class

class FileHandler(CoordinateHandler):
    def __init__(self, basedir: [str, Path], dir_template: [str, Path], template_path: [str, Path] = None):
        self.templates_dir = {}
        self.templates_file = {}
        self.folder_structure = None
        self.sub_structure = False
        self.paths_dict = {}
        self._coords = False

        self.coordinates_dict = {}
        self.preproc_dict = {}
        self.data_dict = {}

        self.basedir = Path(basedir)
        default_template_path = Path(__file__).parents[1] / 'templates'
        self.template_path = Path(template_path) if template_path is not None else default_template_path
        print(self.template_path)
        self.create_folders(template=dir_template)

        self.logger = self._setup_logger()
        super().__init__(basedir=basedir, log_filename=self.paths_dict.get('log'))
        self.logger.info('Setting up FileHandler')
        self.logger.info('Loaded templates')
        self.logger.info('Created folder structure')

    def _load_templates(self):
        """
        Loads templates from template_path into two dictionaries for directory structure and file structure.
        """
        if self.template_path.is_dir():
            templates_dir = [(file.name.replace('_dir.yml', ''), yaml.safe_load(file.read_text())) for file in
                             self.template_path.iterdir() if file.is_file() and file.name.endswith('_dir.yml')]
            self.templates_dir = {temp.get('name', name): temp for name, temp in templates_dir}

            templates_file = [(file.name.replace('_file.yml', ''), yaml.safe_load(file.read_text())) for file in
                              self.template_path.iterdir() if file.is_file() and file.name.endswith('file.yml')]
            if len(templates_file) == 0:
                raise FileNotFoundError('No templates for file structure found.')
            self.templates_file = {temp.get('config', {}).get('name', name): temp for name, temp in
                                   templates_file}
        else:
            raise FileNotFoundError('Path must be a directory.')

    def add_file_template(self, new_template):
        """Adds a new file template to the file template dictionary."""
        self.logger.info('add_file_templates: Adding file template')
        self._load_templates()
        new_template = Path(new_template)
        if new_template.is_file():
            new_name = new_template.name.replace('.yml', '')
            new_yaml = yaml.safe_load(new_template.read_text())
            self.templates_file[new_yaml.get('config', {}).get('prefix', new_name)] = new_template
            self.logger.info('add_file_templates: Added file template')

    def create_folders(self, template):
        self._load_templates()
        print(self.templates_dir)
        basedir = self.basedir
        template_dict = self.templates_dir.get(template)
        if template_dict is None:
            raise KeyError('create_folders: No template was found.')

        main_structure = template_dict.get('main_template')
        if main_structure is None:
            raise KeyError('create_folders: Main template was not found.')
        main_structure = {key: basedir / value for key, value in main_structure.items()}
        if main_structure.get('log') is not None:
            log_path = main_structure.pop('log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.paths_dict['log'] = log_path
        else:
            log_path = None

        for value in main_structure.values():
            value.mkdir(parents=True, exist_ok=True)

        self.folder_structure = main_structure

        if log_path is not None:
            self.folder_structure['log'] = log_path

    def create_subdirs(self, directory, template='default'):
        template_dict = self.templates_dir.get(template)
        if template_dict is None:
            self.logger.error('create_subdirs: No template was found.')
            raise KeyError

        sub_structure = template_dict.get('sub_template')
        if sub_structure is None:
            self.logger.error('create_subdirs: Sub template was not found.')
            raise KeyError
        sub_structure = {key: directory / value for key, value in sub_structure.items()}

        for value in sub_structure.values():
            value.mkdir(parents=True, exist_ok=True)

        return sub_structure

    def _move_files(self, data_raw, coords, preproc):
        if coords is not None:
            coord_path = Path(coords)
            if coord_path.exists():
                if coord_path.is_file():
                    new_coordinate_path = self.folder_structure.get('coordinates_raw') / coord_path.name
                    self.paths_dict['coordinates_raw'] = coord_path.rename(new_coordinate_path)
                else:
                    self.logger.error('_move_files: Coordinate-file path was not a file.')
                    raise ValueError('_move_files: Coordinate-file path was not a file.')
            else:
                self.logger.warning('_move_files: No coordinate-file was found.')

        if preproc is not None:
            preproc_path = Path(preproc)
            if preproc_path.exists():
                if preproc_path.is_file():
                    new_preproc_path = self.folder_structure.get('preproc') / preproc_path.name
                    self.paths_dict['preproc'] = preproc_path.rename(new_preproc_path)
                else:
                    self.logger.error('_move_files: Preprocessing-file path was not a file.')
                    raise ValueError('_move_files: Preprocessing-file path was not a file.')
            else:
                self.logger.warning('_move_files: No preprocessing-file was found.')

        if data_raw is not None:
            data_path = Path(data_raw)
            if data_path.exists():
                if data_path.is_file():
                    new_data_path = self.folder_structure.get('data_raw') / data_path.name
                    self.paths_dict['data_raw'] = data_path.rename(new_data_path)
                elif data_path.is_dir():
                    for file in data_path.iterdir():
                        new_data_path = self.folder_structure.get('data_raw') / file.name
                        _ = file.rename(new_data_path)
                    self.paths_dict['data_raw'] = self.folder_structure.get('data_raw')
                else:
                    self.logger.error('_move_files: Data path was neither a file nor a directory.')
                    raise ValueError('_move_files: Invalid data_path. Must be a file or directory.')
            else:
                self.logger.warning('_move_files: No data was found. Might have been moved already')

    def gather_files(self):
        for key, value in self.folder_structure.items():
            if self.paths_dict.get(key) is None:
                value = Path(value)
                if value.exists() and value.is_dir():
                    files = [file for file in value.iterdir() if file.is_file()]
                    if len(files) == 1:
                        self.paths_dict[key] = files[0]
                    elif len(files) > 1:
                        self.paths_dict[key] = value

    def _check_extension(self, file_path: [str, Path]):
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f'_check_extension: {file_path} does not exist.')
            raise FileNotFoundError(f'_check_extension: {file_path} does not exist.')
        else:
            if file_path.is_file():
                extensions = file_path.suffix
            elif file_path.is_dir():
                extensions = {file.suffix for file in file_path.iterdir() if file.is_file()}
                if len(extensions) != 1:
                    self.logger.error(f'_load_filepaths: Files have different extensions: {extensions}')
                    raise ValueError('All files must have the same extension.')
                else:
                    extensions = next(iter(extensions))
            else:
                self.logger.error(f'_check_extension: {file_path} is neither a file nor a directory.')
                raise ValueError(f'_check_extension: {file_path} is neither a file nor a directory.')
        return extensions

    def choose_file_type(self, file_path, chosen_template=None):
        """Automatically or manually chooses file template for the data files."""
        suffix = self._check_extension(file_path=file_path)

        if chosen_template is not None:
            file_type = self.templates_file.get(chosen_template, None)
            if file_type is None:
                self.logger.error('choose_file_template: Chosen file type was not found')
                raise FileNotFoundError
            elif file_type.get('config').get('file_suffix') != suffix:
                self.logger.error('choose_file_template: File suffix does not match')
                raise ValueError('choose_file_template: File suffix does not match')
            else:
                return file_type
        else:
            parsing_dict = {value.get('config').get('file_suffix'): key for key, value in self.templates_file.items()}
            type_name = parsing_dict.get(suffix)
            file_type = self.templates_file.get(type_name, None)
            if file_type is None:
                self.logger.error(f'choose_file_template: File type was not found in {self.template_path}. Try add_template.')
                raise FileNotFoundError
            return file_type

    def _load_preproc(self):
        self.logger.info('_load_preproc: Loading preprocessing template.')
        self.gather_files()

        preproc = self.paths_dict.get('preproc')
        if preproc is not None:
            self.preproc_dict = yaml.safe_load(preproc.read_text())
            self.logger.info('_load_preproc: Loaded preprocessing template.')
        else:
            self.logger.warning('_load_preproc: No preprocessing template was found. Please create one.')

    @staticmethod
    def create_preproc(**kwargs):
        #todo: implement create_preproc
        print('Function not implemented yet.')

    def _load_coords(self):
        self.logger.info('_load_filepaths: Loading data files')
        self._load_preproc()

        if self.preproc_dict.get('config') is None:
            self.logger.error('_load_filepaths: No preprocessing template was found. Please create one.')
            raise ValueError('_load_filepaths: No preprocessing template was found. Please create one.')

        coords = self.paths_dict.get('coordinates_raw')
        # self.coordinate_handler = CoordinateHandler(basedir=self.basedir, log_filename=self.paths_dict.get('log'))
        if coords is not None:
            sep = self.preproc_dict.get('config', {}).get('sep', ',')
            self.read_coord_file(coords=coords, sep=sep)
            self._coords = True
            self.logger.info('_load_coords: Loaded coordinate files.')
        else:
            self._coords = False

    def add_files(self, data=None, coords=None, preproc=None):
        self.logger.info('read_files: Reading files')
        self._move_files(data_raw=data, coords=coords, preproc=preproc)
        self._load_preproc()
        self.logger.info('read_files: Added files to folderstructure.')

    @staticmethod
    def _find_measurement_blocks(template_dict, lines):
        """Finds all starting indices of the measurement blocks."""
        block_indices = []
        keyword = template_dict.get('config', {}).get('block_start_pattern', None)
        buffer = template_dict.get('config', {}).get('start_buffer', 0)
        if keyword is None:
            block_indices.append(0)
        else:
            for i, line in enumerate(lines):
                    if keyword in line:
                        block_indices.append(i+buffer)
        return block_indices

    def parse_dict(self,template_dict, lines, start_index):
        if not template_dict.get('type', None) == 'dict':
            self.logger.error('parse_dict: Invalid type for parse_dict.')
            raise TypeError
        dictionary = {}
        # self.logger.info('parse_dict: Parsing type is dict')
        start_with = template_dict.get('start_with', None)
        if start_with is not None:
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_with in line:
                        break
        else:
            block_start = start_index

        start_after = template_dict.get('start_after', 0)
        block_start += start_after

        template_lines = template_dict.get('lines', [])
        if not template_lines:
            self.logger.error('parse_dict: Template is missing "lines". It is needed for parsing.')
            raise KeyError
        block_end = block_start + len(template_lines)
        block_lines = lines[block_start:block_end]
        for line in block_lines:
            for entry in template_lines:
                match = re.search(entry.get('pattern'), line)
                if match:
                    for i, key in enumerate(entry.get('key', [])):
                        dictionary[key] = match.group(i+1)

        for key, value in dictionary.items():
            try:
                dictionary[key] = pd.to_numeric(value, errors='ignore')
            except Exception as e:
                self.logger.warning(f'parse_dict: Unexpected Error {e} occurred.')
                continue

        return dictionary

    def parse_dataframe(self, template_dict, lines, start_index):
        if not template_dict.get('type', None) == 'dataframe':
            self.logger.error('parse_dataframe: Invalid type for parse_dataframe.')
            raise TypeError
        # self.logger.info('parse_dataframe: Parsing type is dataframe')
        start_with = template_dict.get('start_with', None)
        if start_with is not None:
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_with in line:
                        break
        else:
            block_start = start_index

        start_after = template_dict.get('start_after', 0)
        block_start += start_after

        end_with = template_dict.get('end_with', None)
        for i, line in enumerate(lines):
            if i >= block_start:
                block_end = i
                if end_with is not None:
                    if end_with in line:
                        break


        end_after = template_dict.get('end_after', 0)
        if end_with is not None:
            block_end += end_after
        block_lines = lines[block_start:block_end+1]
        raw_columns = block_lines[0]
        raw_measurements = block_lines[1:]
        delimiter = template_dict.get('delimiter', '\t').encode().decode('unicode_escape')
        column_delimiter = template_dict.get('column_delimiter', '\t').encode().decode('unicode_escape')
        column_prefix = template_dict.get('column_prefix', '')
        columns = raw_columns.replace(column_prefix, '').strip().split(column_delimiter)
        columns = [col.strip() for col in columns]
        measurements = [line.strip().split(delimiter) for line in raw_measurements]
        measurements = [[m.strip() for m in line] for line in measurements]

        df = pd.DataFrame(measurements, columns=columns)
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore'))
        return df

    def _type_changer(self, type_string, value):
        value = value.strip()
        if type_string == 'int':
            changed = int(value)
        elif type_string == 'float':
            changed = float(value)
        elif type_string == 'bool':
            changed = bool(value)
        elif type_string == 'str':
            changed = str(value)
        else:
            self.logger.error(f'_type_changer: {type_string} is not a valid type.')
            raise TypeError(f'Invalid type string: {type_string}')
        return changed

    def parse_line(self, template_dict, lines, start_index):
        if not template_dict.get('type', None) == 'line':
            self.logger.error('parse_line: Invalid type for parse_line.')
            raise TypeError

        start_with = template_dict.get('start_with', None)
        if start_with is not None:
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_with in line:
                        break
        else:
            block_start = start_index
        start_after = template_dict.get('start_after', 0)
        block_start = block_start + start_after
        line_number = template_dict.get('lines', 1)
        subtype = template_dict.get('subtype', False)
        if line_number == 1:
            line = lines[block_start]
            if subtype:
                line = self._type_changer(subtype, line)
            return line
        else:
            line_block = lines[block_start: block_start + line_number]
            if subtype:
                line_block = [self._type_changer(subtype, x) for x in line_block]
            return line_block

    def find_parsing_type(self, template_dict, lines, start_index):
        template_type = template_dict.get('type', None)
        if template_type is None:
            self.logger.error('find_parsing_type: Template is missing "type". It is needed for parsing.')
            raise KeyError
        if template_type == 'dict':
            return self.parse_dict(template_dict, lines, start_index)
        elif template_type == 'dataframe':
            return self.parse_dataframe(template_dict, lines, start_index)
        elif template_type == 'line':
            return self.parse_line(template_dict, lines, start_index)
        else:
            self.logger.error('find_parsing_type: Template type was not recognized.')
            raise KeyError

    def _parse_file(self, file_path, file_type=None):
        file_path = Path(file_path)
        self.logger.info(f'_parse_file: Started reading {file_path.name}')

        type_file = self.choose_file_type(file_path=file_path, chosen_template=file_type)
        if type_file is None:
            self.logger.error('_parse_file: No file type was found.')
            raise KeyError('_parse_file: No file type was found.')

        file_template = type_file.get('template')
        if file_template is None:
            self.logger.error('_parse_file: Template is missing "template". It is needed for parsing.')
            raise KeyError('"template" not found in YAML-file.')

        measurements = {}
        temp_keys = []
        for line in file_template.get('metadata', {}).get('lines', []):
            for key in line.get('key', []):
                temp_keys.append(key)
        name_exists = 'name' in temp_keys

        if file_path is not None:
            data = [file_path] if file_path.is_file() else file_path.iterdir()

            for data_file in tqdm(data, desc=f'{file_path.name}', unit='file'):
                file_name = data_file.stem
                with open(data_file, 'r') as file_:
                    lines = file_.readlines()

                block_start_indices = self._find_measurement_blocks(template_dict=type_file, lines=lines)
                multiple_blocks = len(block_start_indices) > 1

                if multiple_blocks and not name_exists:
                    multi_block_dict = {key: [] for key in file_template.keys()}

                for start_index in block_start_indices:
                    block_dict = {}
                    for temp_name, temp in file_template.items():
                        # self.logger.info(f'read_files: Parsing {temp_name}')
                        if multiple_blocks and not name_exists:
                            multi_block_dict[temp_name].append(self.find_parsing_type(temp, lines, start_index))
                        else:
                            block_dict[temp_name] = self.find_parsing_type(temp, lines, start_index)
                    if not multiple_blocks or name_exists:
                        measurement_name = block_dict.get('metadata', {}).get('name', file_name)
                        measurements[measurement_name] = block_dict
                        # self.logger.info(f'read_files: Added {measurement_name} to measurements')
                if multiple_blocks and not name_exists:
                    measurements[file_name] = multi_block_dict
                    # self.logger.info(f'read_files: Added {file_name} to measurements')
        self.logger.info(f'Added {file_path.name} to data_dir')
        return measurements

    def _get_coords(self):
        self._load_coords()
        if self._coords:
            if self.preproc_dict.get('config') is not None:
                config = self.preproc_dict.get('config')
                rename_cols = self.preproc_dict.get('rename_columns')
                rename_points = self.preproc_dict.get('rename_points')

                sep = config.get('sep', ',')
                if rename_cols is not None:
                    self.rename_columns(rename_cols)
                if self.check_cols():
                    raise KeyError('_get_coords: Necessary columns not found.')
                if rename_points is not None:
                    self.rename_points(rename_points)

                reproj_key = config.get('reprojection_key')
                if reproj_key is None:
                    crs_start = config.get('crs_initial')
                    crs_end = config.get('crs_to')
                    if crs_start is None or crs_end is None:
                        reproj_key = 'wgs84_utm33n'
                        self.logger.warning(f'_get_coords:No Reprojection key found: {reproj_key} was used.')
                        self.reproject(reproj_key=reproj_key)
                    else:
                        self.custom_reprojection(from_crs=crs_start, to_crs=crs_end)
                else:
                    self.reproject(reproj_key=reproj_key)

                self.interpolate_points()

                naming_pattern = config.get('naming_pattern', r'([A-Za-z]+)(\d*)_(\d*)')
                self.sort_points(pattern=naming_pattern)
                self.coordinates_dict = self.extract_coords()
                proc_coords_path = Path(self.folder_structure.get('coordinates_proc')) / self.paths_dict.get('coordinates_raw').name.replace('.', '_proc.')
                self.to_file(path=proc_coords_path, sep=sep)
                self.logger.info('_get_coords: Coordinates extracted.')

        else:
            self.logger.warning('_get_coords: No _coordinates_proc file was found. Create dummy-_coordinates_proc file.')

    def _add_coords(self, template_key):
        self._get_coords()
        data_dir = self.data_dict.get('data_raw')
        template_dict = self.templates_file.get(template_key)
        if template_dict is None:
            self.logger.error('_add_coords: No template was found.')
            return

        metadata = template_dict.get('template', {}).get('metadata') is not None
        coords = template_dict.get('template', {}).get('_coordinates_proc') is not None
        num_elec = template_dict.get('template', {}).get('number_electrodes') is not None
        num_meas = template_dict.get('template', {}).get('number_measurements') is not None

        #todo: work with ert (roll alongs)
        for key, value in tqdm(data_dir.items(), desc='Adding _coordinates_proc'):
            if metadata:
                new_coords = self.coordinates_dict.get(key, {})
                data_dir[key]['metadata']['x'] = new_coords.get('x', 0)
                data_dir[key]['metadata']['y'] = new_coords.get('y', 0)
                data_dir[key]['metadata']['z'] = new_coords.get('z', 0)

            if num_meas:
                data_dir[key]['number_measurements'] = data_dir.get('data', pd.DataFrame()).shape[0]
            if num_elec:
                data_dir[key]['number_electrodes'] = data_dir.get('data', pd.DataFrame()).iloc[:, :4].max().max()
            if coords:
                new_coords = self.coordinates_dict.get(key, None)
                if new_coords is None:
                    elec_num = data_dir.get('data', pd.DataFrame()).iloc[:, :4].max().max()
                    if pd.isna(elec_num):
                        elec_num = 72
                    new_coords = self.create_dummy(elec_sep=1, elec_num=elec_num)

                data_dir[key]['_coordinates_proc'] = new_coords

        self.data_dict['data_preproc'] = data_dir

    def load_data(self, template):
        self.gather_files()
        raw = self.paths_dict.get('data_raw')
        preprocessed = self.paths_dict.get('data_preproc')
        filtered = self.paths_dict.get('data_filtered')

        # print(filtered)
        # raise ValueError('stop here')

        if filtered is not None:
            self.data_dict['data_filtered'] = self._parse_file(file_path=filtered)

        if preprocessed is not None:
            self.data_dict['data_preproc'] = self._parse_file(file_path=preprocessed)

        if raw is not None:
            self.data_dict['data_raw'] = self._parse_file(file_path=raw)
            if preprocessed is None:
                self._add_coords(template_key=template)
                self.dump(directory_key= 'data_preproc', template_key=template, suffix='_proc')

    @staticmethod
    def write_dict(template_dict, data, file_):
        template_lines = template_dict.get('lines', [])
        for line in template_lines:
            line_keys = line.get('key', [])
            line_values = {key: data.get(key) for key in line_keys}
            write_line = line.get('output_format', '').encode().decode('unicode_escape').format(**line_values)
            file_.write(write_line + '\n')

    @staticmethod
    def write_dataframe(template_dict, data, file_):
        delimiter = template_dict.get('delimiter', '\t').encode().decode('unicode_escape')
        column_delimiter = template_dict.get('column_delimiter', delimiter).encode().decode('unicode_escape')
        column_prefix = template_dict.get('column_prefix', '')
        columns = column_delimiter.join(data.columns)
        file_.write(column_prefix + columns + '\n')

        for index, row in data.iterrows():
            file_.write(f'{delimiter.join(map(str, row))}\n')

    def write_line(self, template_dict, data, file_):
        line_amount = template_dict.get('lines', 1)
        if line_amount == 1:
            file_.write(f'{data}\n')
        else:
            if not len(data) == line_amount:
                self.logger.error(f'write_line: Input dictionary has {len(data)} lines while template allows {line_amount} lines.')
                raise ValueError(f'Input dictionary has {len(data)} lines ({line_amount} needed).')
            for line in data:
                file_.write(f'{line}\n')

    def find_writing_type(self, template_dict, data, file_):
        template_type = template_dict.get('type', None)
        if template_type is None:
            self.logger.error('find_writing_type: Template is missing "type". It is needed for parsing.')
            raise KeyError
        if template_type == 'dict':
            self.write_dict(template_dict, data, file_)
        elif template_type == 'dataframe':
            self.write_dataframe(template_dict, data, file_)
        elif template_type == 'line':
            self.write_line(template_dict, data, file_)
        else:
            self.logger.error('find_writing_type: Template type was not recognized.')
            raise KeyError

    def dump(self, directory_key, template_key, suffix=''):
        template_dict = self.templates_file.get(template_key)
        if template_dict is None:
            self.logger.error('dump: No template was found.')
            return
        file_suffix = template_dict.get('config', {}).get('file_suffix', '')
        multi_file = self.paths_dict.get('data_raw').is_dir()
        data_dict = self.data_dict.get(directory_key)
        target_folder = Path(self.folder_structure.get(directory_key))
        if data_dict is None:
            self.logger.error('dump: No data was found.')
            return

        if not multi_file:
            filename = self.paths_dict.get('data_raw').stem + suffix + file_suffix
            filepath = target_folder / filename
            with open(filepath, 'w') as file_:
                for data in data_dict.values():
                    for key, value in template_dict.get('template').items():
                        data_block = data.get(key)
                        self.find_writing_type(value, data_block, file_)
        else:
            for name, data in data_dict.items():
                filename = self.paths_dict.get('data_raw').stem + suffix + name
                filepath = target_folder / filename
                with open(filepath, 'w') as file_:
                    for key, value in template_dict.get('template').items():
                        data_block = data.get(key)
                        self.find_writing_type(value, data_block, file_)

        self.logger.info(f'dump: Dumped {directory_key} to {target_folder}')

    def close(self):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
                self.logger.info("File handler closed and removed.")
                break

        self.coordinates = None
        self.extracted_coords = None
        self.paths_dict = {}
        self._coords = False
        self.coordinates_dict = {}
        self.preproc_dict = {}
        self.data_dict = {}

