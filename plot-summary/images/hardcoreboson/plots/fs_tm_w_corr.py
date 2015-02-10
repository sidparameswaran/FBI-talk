# coding: utf-8
from __future__ import division
from DataAnalysis import PEPS_plots as pp
from matplotlib import pyplot as plt
import numpy as np
pp.IO.go_to_data_parent('hardcoreboson')
pp.IO.add_to_path()
state = pp.IO.get_state()
# Ls = range(2, 9)
# scale = 0
# L_min = min(Ls)
# L_max = max(Ls)
# spectrum = load_transfer_matrix_spectrum(state, Ls)
# shift_and_scale(spectrum, scale)
# spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [lambda band:band<=1], band='band')

# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             # xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             # xlim=[-0.7, 3.2], ylim=[-0.1, 4], ls = ':',
                             # markersize=8)
Ls = range(2, 9)
scale = 0
L_min = min(Ls)
L_max = max(Ls)
spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
pp.shift_and_scale(spectrum, scale)
spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [lambda band:band<=1], band='band')

spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = pp.data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             xlim=[-0.7, 3.2], ylim=[-0.1, 4], ls = ':',
                             markersize=8)
# axs = []
# grid_shape = (2,3)
# main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
# for indx in xrange(2):
    # axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# axs = []
# grid_shape = (2,3)
# main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
# for indx in xrange(2):
    # axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
from PEPS.CylinderPEPS import run
run.go_diag_sectors(8, 8, sectors=[[0,1], [1,1]])
run.go_diag_sectors(5, 5, sectors=[[0,1], [1,1]])
run.go_diag_sectors(5, 5, sectors=[[0,1], [1,1]], overwrite=True)
30*16**3
0.5*16**3
0.5*16**3/(60)
pp.IO.go_to_data_parent('interpolatedboson/a0')
state = pp.IO.get_state()
# axs = []
# grid_shape = (2,3)
# main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
# for indx in xrange(2):
    # axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
# spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
# Ls
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum', xlabel='K', ylabel='Inverse Correlation **',
                             # xlim=[-0.7, 4.1], ylim=[-0.1, 10])
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# spec_points
# spectrum
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points
# for point in spectrum:
    # print len(spectrum[point])
    
# for point in spectrum:
    # if len(spectrum[point])==0:
        # del spectrum[point]
# spectrum
# spectrum = spectrum.filter([lambda p:0<=p['K']<=1 and 0<=p['N']<=1], [])
# spectrum
# spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
# pp.shift_and_scale(spectrum)
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum', xlabel='K', ylabel='Inverse Correlation **',
                             # xlim=[-0.7, 4.1], ylim=[-0.1, 10])
# axs = []
# grid_shape = (2,3)
# main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
# for indx in xrange(2):
    # axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# get_ipython().system(u'explorer l')
# get_ipython().system(u'explorer .')
# pp.IO.go_to_data_parent('hardcoreboson')
# state = pp.IO.get_state()
# spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# pp.shift_and_scale(spectrum)
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum', xlabel='K', ylabel='Inverse Correlation **',
                             # xlim=[-0.7, 4.1], ylim=[-0.1, 10])
axs = []
grid_shape = (2,3)
main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
for indx in xrange(2):
    axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
plt.xlim(-0.5, 2.1)
plt.ylim(-0.1, 2.1)
spec_points.plot_toolbox.spec_plot(spec_points.filter([lambda p:p['K']>=0 and p['N']>=0],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
plt.xlim(-0.5, 2.1)
plt.ylim(-0.1, 2.1)
get_ipython().system(u'explorer .')
plt.cla()
spec_points.plot_toolbox.spec_plot(spec_points.filter([lambda p:p['K']>=0 and p['N']>=0],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
plt.xlim(-0.5, 2.1)
plt.ylim(-0.1, 2.1)
plt.draw()
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             # xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             # xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             # markersize=8)
# spec_points.plot_toolbox.spec_plot(spec_points.filter([lambda p:2>=p['K']>=0 and 3>=p['N']>=0],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# plt.cla()
# spec_points.plot_toolbox.spec_plot(spec_points.filter([lambda p:2>=p['K']>=0 and 3>=p['N']>=0],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# spec_points.plot_toolbox.spec_plot(spec_points.filter([lambda p:p['K']>=0 and p['N']>=0],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# spec_points
# spectrum
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             # xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             # xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             # markersize=8)
# spec_points
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum', xlabel='K', ylabel='Inverse Correlation **',
                             # xlim=[-0.7, 4.1], ylim=[-0.1, 10])
# spec_points
# spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
# pp.shift_and_scale(spectrum)
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             # xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             # xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             # markersize=8)
# spec_points
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# plt.xlim(-0.5, 2.1)
# plt.ylim(-0.1, 2.1)
# plt.draw()
# spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [lambda band:band<=1], band='band')
# spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [])
# spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
# spec_points.plot_toolbox = pp.data.SpectraPlotTools()
# spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
# spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             # xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             # xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             # markersize=8)
# plt.cla()
# spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                # shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
# spec_points
# spectrum
# spectrum.keys()
# [p for p in spectrum]
# spectrum.params
# print spectrum.params
# type(spectrum.params)
# spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [])
# spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
# pp.shift_and_scale(spectrum)
# spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [])
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = pp.data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             markersize=8)

spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
plt.cla()
spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
symmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 0, 1]))]
symmetric_inverse_corr2 = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [1, 0, 0]))]
symmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 0, 1]))]
print symmetric_inverse_corr
print symmetric_inverse_corr2
symmetric_inverse_corr.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
symmetric_inverse_corr2.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
notsymmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 1, 0]))]
notsymmetric_inverse_corr.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
# symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
# symdata = np.array([[L, val] for L, val in symdata])
# plt.sca(axs[1])
# fit_data = symdata[:, 1]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.expic
# fit_Ls = range(4, 9)
# fit_label = 'Fit'
# txt_coords = (3.1, 0.1)
# (fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
import PEPS.analysis as a
# symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
# symdata = np.array([[L, val] for L, val in symdata])
# plt.sca(axs[1])
# fit_data = symdata[:, 1]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.expic
# fit_Ls = range(4, 9)
# fit_label = 'Fit'
# txt_coords = (3.1, 0.1)
# (fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
# fit_data
# fit_Ls
# symdata
# symmetric_inverse_corr

# symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
# symdata = np.array([[L, val] for L, val in symdata])
# symdata = np.array(sorted(symdata, key = lambda x:x[0]))
# plt.sca(axs[1])
# fit_data = symdata[:, 1]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.expic
# fit_Ls = range(4, 9)
# fit_label = 'Fit'
# txt_coords = (3.1, 0.1)
# (fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)

# symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
# symdata = np.array([[L, val] for L, val in symdata])
# symdata = np.array(sorted(symdata, key = lambda x:x[0]))
# plt.sca(axs[1])
# fit_data = symdata[:, 1]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.expic
# fit_Ls = symdata[:, 0]
# fit_label = 'Fit'
# txt_coords = (3.1, 0.1)
# (fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
# symmetric_inverse_corr2.plot('L', 'Correlation Bound', ax = axs[1], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
# plt.plot(extrapolate_pnts, fit_data, label = fit_label)

# symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
# symdata = np.array([[L, val] for L, val in symdata])
# symdata = np.array(sorted(symdata, key = lambda x:x[0]))
# plt.sca(axs[1])
# fit_data = symdata[:, 1]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.power_law
# fit_Ls = symdata[:, 0]
# fit_label = 'Fit'
# txt_coords = (3.1, 0.1)
# (fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
# symmetric_inverse_corr2.plot('L', 'Correlation Bound', ax = axs[1], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
# plt.plot(extrapolate_pnts, fit_data, label = fit_label)
# plt.draw()
# plt.cla()
# plt.draw()
# fit_data
# symdata

# symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
# symdata = np.array([[L, val] for L, val in symdata])
# symdata = np.array(sorted(symdata, key = lambda x:x[0]))
# plt.sca(axs[1])
# fit_data = symdata[:, 1][-4:]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.power_law
# fit_Ls = symdata[:, 0][-4:]
# fit_label = 'Fit'
# txt_coords = (3.1, 0.1)
# (fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
# symmetric_inverse_corr2.plot('L', 'Correlation Bound', ax = axs[1], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
# plt.plot(extrapolate_pnts, fit_data, label = fit_label)
# params
# R
symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
symdata = np.array([[L, val] for L, val in symdata])
symdata = np.array(sorted(symdata, key = lambda x:x[0]))
plt.sca(axs[1])
fit_data = symdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = symdata[:, 0][-4:]
fit_label = 'Fit'
txt_coords = (3.1, 0.1)
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
# params
# R
# plt.cla()
symmetric_inverse_corr2.plot('L', 'Correlation Bound', ax = axs[1], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label)
txt_coords = (3.1, 0.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
plt.xlim(3, 10)
notsymdata = notsymmetric_inverse_corr.view_props(['L', 'Correlation Bound'])
notsymdata = np.array([[L, val] for L, val in notsymdata])
notsymdata = np.array(sorted(notsymdata, key = lambda x:x[0]))
plt.sca(axs[1])
fit_data = notsymdata[:, 1][-5:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-5:]
fit_label = 'Fit'
txt_coords = (3.1, 0.1)
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label)
params
notsymdata
fit_data = notsymdata[:, 1][-5:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-5:]
fit_data
fit_Ls
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
params
R
fit_data = notsymdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-4:]
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
params
R
plt.cla()
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label)
plt.cla()
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'k')
plt.xlim(3, 10)
txt_coords = (3.1, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
# (fit_data, params, R) = a.curve_fit_main(fit_data,a.power_law, fit_Ls, extrapolate_pnts)
# fit_data = notsymdata[:, 1][-4:]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.expic
# fit_Ls = notsymdata[:, 0][-4:]
# (fit_data, params, R) = a.curve_fit_main(fit_data,a.power_law, fit_Ls, extrapolate_pnts)
# params
# R
# (fit_data, params, R) = a.curve_fit_main(fit_data,a.expic, fit_Ls, extrapolate_pnts)
# fit_data = notsymdata[:, 1][-4:]
# extrapolate_pnts = np.arange(3, 10, 0.1)
# fit_func = a.expic
# fit_Ls = notsymdata[:, 0][-4:]
# (fit_data, params, R) = a.curve_fit_main(fit_data,a.expic, fit_Ls, extrapolate_pnts)
# R
from matplotlib import rcParams
plot_params = {'axes.labelsize': 12,   # 8
               'title.fontsize': 12,   # 8
               'legend.fontsize': 10,  # 8
               'font.size': 12,        # 8
               'xtick.labelsize': 12,  # 8
               'ytick.labelsize': 12,  # 8
               'lines.markersize': 12, # 8
               'savefig.bbox': 'tight',   # tight
               'savefig.pad_inches': 0.1, # 0.1
               # 'text.usetex': True,
               'font.family': 'serif',
               #'text.latex.preamble' : [r'\usepackage{amsmath}'],
               'figure.figsize': (10, 8)}
rcParams.update(plot_params)
plt.draw()
plt.tight_layout()
plt.legend(loc='best')
plt.sca(axs[1])
plt.legend(loc='best')
