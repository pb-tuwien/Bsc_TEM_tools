from scipy.constants import survey_footfrom Demos.win32console_demo import coord

# SurveyTEM Package

## Overview

The `SurveyTEM` package is designed for processing and analyzing Transient Electromagnetic (TEM) survey data. It provides functionalities for reading, preprocessing, filtering, inverting, and visualizing TEM data.

## Features

- **Data Reading**: Read raw TEM data from files.
- **Data Preprocessing**: Add coordinates, normalize data, and calculate apparent resistivity and conductivity.
- **Data Filtering**: Filter data based on time range and noise floor.
- **Data Inversion**: Perform inversion to obtain subsurface resistivity models.
- **Data Forward Modeling**: Generate forward models for TEM data.
- **Visualization**: Plot raw, filtered, and inverted data.

## Installation

To install the package, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Importing the Package

```python
import src.tem.survey_tem as st
```

### First Setup

For the first time it takes some extra steps to set up the project directory. 
The project directory is the directory where all the data files are stored.
Additionally, some pre-processing steps are performed to prepare the data for further analysis.
The following code snippet shows how to set up the project directory:

```python
import src.tem.survey_tem as st

coordinates = 'path/to/coordinates.csv'
tem_data = 'path/to/tem_data.tem'
project_directory = 'path/to/project'

# If sounding have a different name in the coordinate file to the tem data file:
rename_points = {'old_name': 'new_name', 'old_part': 'new_part'}
# Renames either a point or part of a point name (like - to _)

# If one coordinate point is the location for several TEM soundings:
parse_points = {'sounding_name_1': 'coordinate_name', 'sounding_name_2': 'coordinate_name'}

survey = st.SurveyTEM(project_directory)
survey.coords_read(coords=coordinates, sep=',')
survey.coords_rename_points(rename_dict=rename_points)
survey.coords_sort_points()
survey.coords_reproject()
survey.coords_extract_save()
survey.data_read(data=tem_data)
survey.data_preprocess(parsing_dict=parse_points)
```

### Second Setup

After the project directory is created and the data is preprocessed, 
the following code snippet shows how to load the project directory:

```python
import src.tem.survey_tem as st

project_directory = 'path/to/project'
parse_points = {'sounding_name_1': 'coordinate_name', 'sounding_name_2': 'coordinate_name'}

survey = st.SurveyTEM(project_directory)
survey.data_read()
survey.data_preprocess(parsing_dict=parse_points)
```

### Filtering Data

The following code snippet shows how to filter the data based on a time range:

```python
filter_times = (10, 500) # Time range in microseconds
survey.data_filter(filter_times=filter_times)
```
All functions later on will use this `data_filter()` internally, so it is not necessary to call it before other functions.

### First-Look Plot

Visualize the raw versus filtered data:

```python
# If no subset is provided, all soundings will be plotted
subset = ['sounding_name_1', 'sounding_name_2']
survey.plot_raw_filtered(subset=subset, filter_times=(7, 700), legend=True)
```

### Inversion

Inversion results can be calculated using the `data_inversion()` function. 
As it is also used within `plot_inversion()`, which also plots the inversion results,
the code snippet below shows how to plot the inversion results:

```python
survey.plot_inversion(subset=['sounding_name_1'], lam=600, filter_times=(7, 700),
                      layer_type='linear', layers=1, max_depth=30)
```
This plots the inversion results for `sounding_name_1` with a regularization parameter of 600,
a time range of 7 to 700 microseconds, a linear layer type with a layer thickness of 1 meter, 
and a maximum depth of 30 meters.

### Forward Modeling

It is also possible to simulate TEM data using the forward modeling function:

```python
import numpy as np

depth_vector = np.array([0, 5, 10, 15, 20, 25, 30])
model_vector = np.array([10, 10, 10, 100, 100, 10, 10])
survey.plot_forward_model(subset=['sounding_name_2'], filter_times=(1, 1000), 
                          max_depth=30, layer_type='custom', layers=depth_vector, 
                          model=model_vector)
```
This plots the simulated response between 1 and 1000 microseconds 
for the measurement configuration of `sounding_name_2`.
The provided model of the subsurface consists of 3 Layers with resistivities:
- 10 Ohm\*m (between 0 and 15 meters)
- 100 Ohm\*m (between 15 and 25 meters)
- 10 Ohm\*m (between 25 and 30 meters)

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. 