from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from DataAnalysis import PEPS_plots as pp
from PEPS import IO_helper as IO
from PEPS.CylinderPEPS import run
import sfig

from matplotlib import rcParams
rcParams.update({'figure.figsize':(5.3, 4)})

Ls = range(2, 9)
scale = 0
L_min = min(Ls)
L_max = max(Ls)

# from PEPS.CylinderPEPS import run
# run.go_diag_sectors(8, 8, sectors=[[0,1], [1,1]])
# run.go_diag_sectors(5, 5, sectors=[[0,1], [1,1]])
# run.go_diag_sectors(5, 5, sectors=[[0,1], [1,1]], overwrite=True)

#pp.IO.go_to_data_parent('interpolatedboson/a0')
IO.go_to_data_parent('softcoreboson')
IO.add_to_path()
state = IO.get_state()

spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
pp.shift_and_scale(spectrum, scale)
spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [lambda band:band<=1], band='band')
    
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = pp.data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             markersize=8)


symmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 0, 1]))]
symmetric_inverse_corr2 = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [1, 0, 0]))]
notsymmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 1, 0]))]

symmetric_inverse_corr.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
symmetric_inverse_corr2.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
notsymmetric_inverse_corr.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')

axs = []
grid_shape = (2,3)
main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
for indx in xrange(2):
    axs.append(plt.subplot2grid(grid_shape, (indx, 2)))

spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)

import PEPS.analysis as a

symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
symdata = np.array([[L, val] for L, val in symdata])
symdata = np.array(sorted(symdata, key = lambda x:x[0]))
plt.sca(axs[1])
fit_data = symdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = symdata[:, 0][-4:]
fit_label = 'Fit'

(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
symmetric_inverse_corr2.plot('L', 'Correlation Bound', ax = axs[1], xlabel = 'L', ylabel='Correlation Length', title = 'Symmetric Correlation Length', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color='k')
txt_coords = (5, -0.3)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': \n$a-ce^{-L/b}$'+'\n a ={} \n b={} \nc={}'.format(*[IO.round_sig(p) for p in params]))
plt.xlim(3, 10)
plt.ylim(-0.4, 1.5)                                                


notsymdata = notsymmetric_inverse_corr.view_props(['L', 'Correlation Bound'])
notsymdata = np.array([[L, val] for L, val in notsymdata])
notsymdata = np.array(sorted(notsymdata, key = lambda x:x[0]))
plt.sca(axs[1])
fit_data = notsymdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-4:]
fit_label = 'Fit'
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length', title = 'Overall Correlation Length', label = 'Data', marker = '*', color = 'b')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color='b')


txt_coords = (5, 0.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': \n$a-ce^{-L/b}$'+'\n a ={} \n b={} \nc={}'.format(*[IO.round_sig(p) for p in params]))
plt.xlim(3, 10)
plt.ylim(-0.2, 4.5)

plt.legend(loc=2, frameon=False, numpoints=1)
plt.sca(axs[1])
plt.legend(loc=2, frameon=False, numpoints=1)
plt.tight_layout()
plt.show()
