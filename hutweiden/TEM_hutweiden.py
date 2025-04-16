#%% Import class
import os
os.chdir('C:/Users/jakob/Documents/Meine Ordner/TU/Bachelorarbeit/Bsc_TEM_tools')
import TEM_tools.tem.survey_tem as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygimli.viewer.mpl as pgv

#%% Test Survey

tem_data = 'C:/Users/jakob/Documents/Meine Ordner/TU/Bachelorarbeit/Bsc_TEM_tools/hutweiden/data/20241008_Hutweidelacke_txt.tem'
tem_coords = 'C:/Users/jakob/Documents/Meine Ordner/TU/Bachelorarbeit/Bsc_TEM_tools/hutweiden/data/20240917_tem_hutweiden_coords.csv'

# rename_points = {'M11': 'M011', 'M12': 'M012', 'M13': 'M013', 'M14': 'M014',
#                         'M15z': 'M015', 'M16': 'M016', 'M17': 'M017', 'M18': 'M018',
#                         'M19': 'M019', 'M20': 'M020', 'M21': 'M021', 'M22': 'M022',
#                         'M23': 'M023', 'M24': 'M024', 'M25': 'M025', 'M26': 'M026',
#                         'M27': 'M027', 'M28': 'M028', 'M29': 'M029', 'M30': 'M030',
#                         'M31': 'M031', 'M32': 'M032', 'M33': 'M033', 'M34': 'M034',
#                         'M35': 'M035', 'M36': 'M036', 'M37': 'M037', 'M38': 'M038',
#                         'M39': 'M039', 'M40': 'M040', 'M41': 'M041', 'M42': 'M042',
#                         'M43': 'M043', 'M44': 'M044', 'M45': 'M045'}
parsing_coords = {'H_test': 'Htest'}
#%%
survey = st.SurveyTEM('hutweiden/dir')
#%%
survey.coords_read(coords=tem_coords)
# survey.coords_rename_points(rename_dict=rename_points)
survey.coords_sort_points()
survey.coords_reproject()
survey.coords_extract_save()
survey.data_read(data=tem_data)
survey.coords_read()
#%%

survey.data_read()
survey.data_preprocess(parsing_dict=parsing_coords)
#%%


lake = [51, 43, 50, 10, 12, 49, 17, 48, 46, 45, 32, 35, 37, 30, 47, 44]
lake.sort()

name_lake = ['H'+str(i).zfill(3) for i in lake]
survey.plot_raw_filtered(subset=name_lake, filter_times=(10, 80), legend=True)
#%% DICT

survey.plot_inversion(subset='H050', max_depth=10, lam=183, layer_type='log', layers=40, filter_times=(10,80))
survey.plot_inversion(subset='H050', max_depth=10, lam=183, layer_type='linear', layers=0.5, filter_times=(10,80))
survey.plot_inversion(subset='H050', max_depth=10, lam=183, layer_type='dict', layers={0:1, 5:1.5, 15:2}, filter_times=(10,80))


#%%
survey.analyse_inversion_gradient_curvature(sounding='H050',
                            layer_type='dict',
                             layers={0:1, 5:1.5, 15:2},
                             max_depth=10,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80))
#%%
survey.analyse_inversion_golden_section(sounding='H050',
                                        layer_type='dict',
                                        layers={0:1, 5:1.5, 15:2},
                                        max_depth=10,
                                        test_range=(100, 400),
                                        filter_times=(10, 80))
#%%
_ = survey.l_curve_plot(sounding='H050',
                    layer_type='dict',
                    layers={0:1, 5:1.5, 15:2},
                    max_depth=10,
                    test_range=(10, 1000, 20),
                    filter_times=(10, 80))
#%%
survey.optimised_inversion_plot(sounding='H050',
                                layer_type='dict',
                                layers={0:1, 5:1.5, 15:2},
                                max_depth=10,
                                test_range=(10, 1000, 20),
                                filter_times=(10, 80),
                                lam=183,
                                fname=False)
#%% LINEAR



survey.plot_inversion(subset=['H050'], max_depth=20, fname=False)
#%%
survey.lambda_analysis_comparison(sounding='H050',
                             layer_type='dict',
                             layers={0:1, 5:1.5, 15:2},
                             max_depth=10,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80), fname=False)

#%%
survey.plot_inversion(subset='H050', max_depth=20, lam=200, layer_type='linear', layers=.5, filter_times=(10,80))
#%%
survey.analyse_inversion_gradient_curvature(sounding='H050',
                            layer_type='linear',
                             layers=.5,
                             max_depth=20,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80))
#%%
survey.analyse_inversion_golden_section(sounding='H050',
                                        layer_type='linear',
                                        layers=.5,
                                        max_depth=20,
                                        test_range=(100, 400),
                                        filter_times=(10, 80))
#%%
_ = survey.l_curve_plot(sounding='H050',
                    layer_type='linear',
                    layers=.5,
                    max_depth=20,
                    test_range=(10, 1000, 20),
                    filter_times=(10, 80))
#%%
survey.optimised_inversion_plot(sounding='H050',
                                layer_type='linear',
                                layers=.5,
                                max_depth=20,
                                test_range=(10, 1000, 20),
                                filter_times=(10, 80),
                                lam=89,
                                fname=False)

#%%
survey.lambda_analysis_comparison(sounding='H050',
                             layer_type='linear',
                             layers=.5,
                             max_depth=20,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80), fname=False)

#%% LOG


survey.plot_inversion(subset=['H050'], max_depth=20, fname=False)
#%%
survey.lambda_analysis_comparison(sounding='H050',
                             layer_type='log',
                             layers=40,
                             max_depth=20,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80), fname=False)

#%%
survey.plot_inversion(subset='H050', max_depth=20, lam=200, layer_type='log', layers=40, filter_times=(10,80))
#%%
survey.analyse_inversion_gradient_curvature(sounding='H050',
                            layer_type='log',
                             layers=40,
                             max_depth=20,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80))
#%%
survey.analyse_inversion_golden_section(sounding='H050',
                                        layer_type='log',
                                        layers=40,
                                        max_depth=20,
                                        test_range=(100, 400),
                                        filter_times=(10, 80))
#%%
_ = survey.l_curve_plot(sounding='H050',
                    layer_type='log',
                    layers=40,
                    max_depth=20,
                    test_range=(10, 1000, 20),
                    filter_times=(10, 80))
#%%
survey.optimised_inversion_plot(sounding='H050',
                                layer_type='log',
                                layers=40,
                                max_depth=20,
                                test_range=(10, 1000, 20),
                                filter_times=(10, 80),
                                lam=162,
                                fname=False)

#%%
survey.lambda_analysis_comparison(sounding='H050',
                             layer_type='log',
                             layers=40,
                             max_depth=20,
                             test_range=(10, 1000, 20),
                             filter_times=(10, 80), fname=False)

#%%% L-Curves
for sounding in ['H'+str(i).zfill(3) for i in range(1, 56)]:
    _ = survey.l_curve_plot(sounding=sounding,
                    layer_type='dict',
                    layers={0:1, 5:1.5, 15:2},
                    max_depth=10,
                    test_range=(10, 1000, 20),
                    filter_times=(10, 80))



#%%

intersections = pd.read_csv('C:/Users/jakob/Documents/Meine Ordner/TU/Bachelorarbeit/Bsc_TEM_tools/hutweiden/dir/TEM-coordinates/01-raw/TEM_intersections_buffered.csv')   
inversion_soundings = []

intersections = intersections[["Name", "Longitude", "Latitude", "Ellipsoida"]]

line = [[640130.3333035399, 5292391.683254148], [639965.9683217484, 5292105.370060059]]

#%%
from pyproj import Transformer

# Define the EPSG codes
source_epsg = "EPSG:4326"  # Replace with your source EPSG code
target_epsg = "EPSG:32633"  # Replace with your target local EPSG code

# Create a transformer object
transformer = Transformer.from_crs(source_epsg, target_epsg, always_xy=True)

# Example: Convert coordinates
intersections["Local_X"], intersections["Local_Y"] = transformer.transform(
    intersections["Longitude"].values, intersections["Latitude"].values
)
#%%
# Define the start and end points of the line
line_start = np.array(line[0])
line_end = np.array(line[1])

# Define the points in proximity as a pandas DataFrame
# Example DataFrame
# data = {
#     "Point": [1, 2, 3],
#     "X": [640100, 640050, 640000],
#     "Y": [5292300, 5292200, 5292100],
# }
# points_df = pd.DataFrame(data)

# Calculate the direction vector of the line
line_vector = line_end - line_start
line_length = np.linalg.norm(line_vector)  # Length of the line
line_unit_vector = line_vector / line_length  # Unit vector of the line

# Function to project a point onto the line and calculate the distance from the start
def project_point(point):
    point_vector = np.array([point["Local_X"], point["Local_Y"]]) - line_start
    projection_length = np.dot(point_vector, line_unit_vector)  # Scalar projection
    return projection_length

# Apply the projection function to each point
intersections["Distance"] = intersections.apply(project_point, axis=1)

# Sort the points by their distance along the line
intersections = intersections.sort_values(by="Distance").reset_index(drop=True)

# Extract the list of distances and point numbers in the correct order
distances = intersections["Distance"].tolist()
point_numbers = intersections["Name"].tolist()

# Output the results
print("Distances:", distances)
print("Point Numbers:", point_numbers)


#%%

def inversion_plot_2D_section(lam=183, lay_thk={0:1, 5:1.5, 15:2}):
    fig, ax = plt.subplots(figsize=(10, 4))
    filter_times = [10,80]
    noise_floor = 0.025
    thicknesses, resistivities, resistivities2d = [], [], []
    for lmnt in intersections['Name']:
        survey.data_inversion(subset = [lmnt], layer_type='dict', 
                              lam=lam, layers=lay_thk, filter_times=filter_times, 
                              noise_floor=noise_floor, max_depth=10)
        inv_name = f'{lam}_{filter_times[0]}_{filter_times[1]}'
        inverted_data = survey.data_inverted().get(lmnt, {}).get(inv_name)
        inversion_data = inverted_data.get('data')
        thks = inversion_data['modelled_thickness'].dropna()
        model_unit = inversion_data['resistivity_model'].dropna()
        resistivities2d.append([res + intersections[intersections['Name']==lmnt]['Distance'] for res in model_unit])
        resistivities.append(model_unit)
        thicknesses.append(thks)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plotting the data
    # Customize colorbar to match the data values
    model_unit_list, dist_list, thks_list = [], [], []
    thks = [1, 2, 3, 4, 5, 6.5, 8.0, 9.5, 11.0]
    for sounding, dist in zip(range(11), distances):
        model_unit = resistivities[sounding]
        dist = [dist for _ in range(len(thks))]
        for i, j, k in zip(model_unit, dist, thks):
            model_unit_list.append(i)
            dist_list.append(j)
            thks_list.append(k)

        
    sc = ax.scatter(dist_list,thks_list, c=model_unit_list, cmap='viridis', s=150, label='pyGIMLI')
    
    cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_ticks(np.linspace(np.min(unit_values), np.max(unit_values), num=6),
    #                 fontsize=14)  # Set ticks according to data range
    cbar.set_label(r'$\rho$ ($\Omega$m)', fontsize=16)  # Set colorbar label
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"2D Crosssection with Multiple Soundings\n$\lambda$ = 183", fontsize=22, fontweight='bold')
    ax.set_ylim(-2, 12)
    for x_val, y_val, label in zip(distances, [0 for _ in range(len(distances))], intersections['Name']):
        ax.text(x_val, y_val, label, fontsize=10, color="black", ha="center", va="bottom", rotation=45)
   
    ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
    fig.savefig('inversion_2Dsection_scatter_{}.png'.format('rhoa'))


# %%
inversion_plot_2D_section()
# %%
