#%% Import class
import os
os.chdir('C:/Users/jakob/Documents/Meine Ordner/TU/Bachelorarbeit/Bsc_TEM_tools')
import src.tem.survey_tem as st
import numpy as np

#%% Test Survey

tem_data = 'C:/Users/jakob/Documents/Meine Ordner/TU/Bachelorarbeit/Bsc_TEM_tools/hutweiden/data/20241008_tem_hutweiden_data.tem'
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
survey.data_read()
#%%

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
                             layer_type='linear',
                             layers=.5,
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