#%%
from pathlib import Path
import gp_package.framework.filehandler as fh
import gp_package.tem.survey_tem as st
import shutil

ohm_template = 'src/templates/ert'
tem_template = 'src/templates/tem'

tem_data = 'test/test_data/tem_testfile.tem'
tem_coords = 'test/test_data/tem_testcoords.csv'
tem_preproc = 'test/test_data/tem_testpreproc.yml'
ohm_data = 'test/test_data/ohm_testfile.ohm'
syscal_data = 'test/test_data/syscal_testfile.dat'
syscal_coords = 'test/test_data/syscal_testcoords.csv'
syscal_preproc = 'test/test_data/syscal_testpreproc.yml'

#%% Test NewHandler: TEM

new_tem = fh.FileHandler('test/test_data/tem', tem_template)
new_tem.add_files(data=tem_data, coords=tem_coords, preproc=tem_preproc)
new_tem.load_data('tem')

def return_folder_structure(handler_class):
    coords = handler_class.paths_dict.get('coordinates_raw')
    data = handler_class.paths_dict.get('data_raw')
    preproc = handler_class.paths_dict.get('preproc')

    assert coords.exists()
    assert data.exists()
    assert preproc.exists()

    coords.rename(Path(tem_coords))
    data.rename(Path(tem_data))
    preproc.rename(Path(tem_preproc))
    new_tem.close()

    shutil.rmtree('test/test_data/tem')
    print('Checked tem folder structure and returned to original layout.')


#%% Hutweiden_lacke

# hut_tem_coords = 'data/hutweiden/tem/20241008/20240917_tem_hutweiden_coords.csv'
# hut_tem_data = 'data/hutweiden/tem/20241008/20241008_tem_hutweiden_data.tem'
# hut_tem_preproc = 'data/hutweiden/tem/20241008/20241008_tem_hutweiden_preproc.yml'
#
# survey_hut = st.SurveyTEM('data/hutweiden/tem/20241008')
# survey_hut.read_data()
# survey_hut.load_data()
# survey_hut.filter_data(filtertimes=(10, 80))
# survey_hut.plot_raw_filtered(legend=False, filter_times=(10, 80))
# survey_hut.plot_inversion(subset=['H010'], max_depth=30, layers=1, filter_times=(10, 80))

#%% Testing

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

lambda_values = np.logspace(np.log10(10), np.log10(5000), 20)
roughness_values = []
rms_values = []
for lam in lambda_values:
    survey_marten.data_inversion(subset=['M043'], lam=lam, layer_type='dict', layers={0:1, 5:1.5, 15:2}, verbose=False,
                        max_depth=30, filter_times=(10, 110), noise_floor=0.025, start_model=None)

    inversion_dict = survey_marten._data_inverted.get('M043', {}).get(f'{lam}_{10}_{110}')
    rms_values.append(inversion_dict.get('metadata').get('absrms')) #todo: change labels
    roughness_values.append(inversion_dict.get('metadata').get('phi_model'))

# Example data
roughness_values = np.array(roughness_values)  # Replace with your roughness values
rms_values = np.array(rms_values)  # Replace with your RMS values
lambda_values = np.array(lambda_values)  # Replace with your lambda values

roughness_values = np.log10(roughness_values)
rms_values = np.log10(rms_values)

sorted_indices = np.argsort(roughness_values)
roughness_values = roughness_values[sorted_indices]
rms_values = rms_values[sorted_indices]
lambda_values = lambda_values[sorted_indices]

# Fit cubic spline
cs = CubicSpline(roughness_values, rms_values)

# Compute the first and second derivatives of the spline
cs_d = cs.derivative(nu=1)
cs_dd = cs.derivative(nu=2)

# Evaluate the first and second derivatives at the original roughness values
first_derivative = cs_d(roughness_values)
second_derivative = cs_dd(roughness_values)

# Calculate the curvature kappa
curvature_values = second_derivative / (1 + first_derivative**2)**(3/2)
max_curvature_index = np.argmax(curvature_values)
opt_lambda = lambda_values[max_curvature_index]

max_curv_roughness = roughness_values[max_curvature_index]

vmax = max(curvature_values.max(), rms_values.max())
vmin = min(curvature_values.min(), rms_values.min())

# Print the curvature values
# for roughness, rms, curvature in zip(roughness_values, rms_values, curvature_values):
#     print(f'Roughness: {roughness}, RMS: {rms}, Curvature: {curvature}')

# Plot the original data, the fitted spline, and the curvature
x_new = np.linspace(roughness_values.min(), roughness_values.max(), 500)
y_new = cs(x_new)
curvature_new = cs_dd(x_new) / (1 + cs_d(x_new)**2)**(3/2)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(roughness_values, rms_values, 'o', label='Original data')
ax[1].plot(roughness_values, curvature_values, 'd', label='Curvature')
ax[0].plot(x_new, y_new, '-', label='Cubic spline fit')
ax[0].vlines(max_curv_roughness, rms_values.min(), rms_values.max(), colors='r', linestyles='--', label='Max curvature at lambda = {:.2f}'.format(opt_lambda))
ax[1].vlines(max_curv_roughness, curvature_values.min(), curvature_values.max(), colors='r', linestyles='--', label='Max curvature at lambda = {:.2f}'.format(opt_lambda))
ax[1].plot(x_new, curvature_new, '-', label='Curvature')
# plt.xlabel('Roughness')
# plt.ylabel('RMS / Curvature')
ax[0].legend()
ax[1].legend()
fig.show()