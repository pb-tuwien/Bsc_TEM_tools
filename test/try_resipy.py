#%%
from resipy import Project
from pathlib import Path
from gp_package.core.gp_file import GPfile
from gp_package.core.gp_coords import GPcoords
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

test_data_dat = Path('data/hutweiden/ert/20240917/20240917_ert_hutweiden_data/P1_1.dat')
test_data_coords = Path('data/hutweiden/ert/20240917/20240917_ert_hutweiden_coords.csv')
test_data_ohm = Path('test/test_data/ohm_testfile.ohm')
rename_coords = {'L2': 'P1'}

#%%
test_data = GPfile().read(test_data_dat)
p1_data = test_data['P1_1'].get('data')
p1_data.rename(columns={'A': 'a', 'B': 'b', 'M': 'm', 'N': 'n', 'R(omega}': 'r'}, inplace=True)
p1_data = p1_data[['a', 'b', 'm', 'n', 'r']]

gp_coords = GPcoords()
gp_coords.read(test_data_coords)
gp_coords.rename_points(rename_coords)
gp_coords.sort_points()
gp_coords.interpolate_points()
gp_coords.reproject()
coord_dict = gp_coords.extract_coords()

p1_coords = coord_dict['P1']

data_dict = {'number_electrodes': len(p1_coords),
             'coordinates': p1_coords,
             'number_measurements': len(p1_data),
             'data': p1_data}

GPfile().write(data=data_dict, file_path='test/test_data/p1_testfile.ohm')





#%%


k = Project(typ='R2', dirname='test')
k.createSurvey(fname='test/test_data/p1_testfile.ohm', ftype='BERT')
fig, ax = plt.subplots()
k.showPseudo(ax=ax)
fig.show()
#%%
# fig2, ax2 = plt.subplots()
# k.showError(ax=ax2)
# fig2.show()

# k.filterUnpaired()
# k.filterElec([50])
# fig3, ax3 = plt.subplots()
# k.showPseudo(ax=ax3)
# fig3.show()

# fig4, ax4 = plt.subplots()
# k.filterRecip(percent=20)
# k.showPseudo(ax=ax4)
# fig4.show()
#
# fig5, ax5 = plt.subplots()
# k.fitErrorLin(ax=ax5)
# fig5.show()

fig6, ax6 = plt.subplots()
k.createMesh(typ='trian', show_output=False)
k.showMesh(ax=ax6)
fig6.show()


# k.err = True
k.invert()
#%%
fig7, ax7 = plt.subplots()
k.showResults(attr='Resistivity(ohm.m)', sens=False, contour=True, ax=ax7)
fig7.show()