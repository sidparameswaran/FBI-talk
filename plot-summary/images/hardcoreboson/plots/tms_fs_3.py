# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np
from __future__ import division
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
from DataAnalysis import PEPS_plots as pp
state = pp.IO.get_state()
Ls = range(2, 9)
spectrum = pp.load_transfer_matrix_spectrum(state, Ls)
pp.shift_and_scale(spectrum)
spectrum = spectrum.filter([lambda p:(0<=p['K']<4 and 0<=p['N']<4)], [])
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = pp.data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Transfer Matrix Spectrum',
                             xlabel='K', ylabel='Inverse Correlation Length -log(|T|) = 1/$\\xi$',
                             xlim=[-0.5, 2.1], ylim=[-0.1, 2.1], ls = ':',
                             markersize=8)

L_min = min(Ls)
L_max = max(Ls)
axs = []
grid_shape = (2,3)
main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
for indx in xrange(2):
    axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
                                                
                                                
spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
symmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 0, 1]))]
symmetric_inverse_corr2 = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [1, 0, 0]))]
notsymmetric_inverse_corr = spec_points[pp.data.odict(zip(['K', 'N', 'band'], [0, 1, 0]))]
symmetric_inverse_corr.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
symmetric_inverse_corr2.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
notsymmetric_inverse_corr.add_as_prop(lambda E:1/E, prop_name = 'Correlation Bound', E='E')
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'k')
plt.xlim(3, 10)
txt_coords = (3.1, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
fit_data = notsymdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-4:]
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
notsymdata = notsymmetric_inverse_corr.view_props(['L', 'Correlation Bound'])
notsymdata = np.array([[L, val] for L, val in notsymdata])
notsymdata = np.array(sorted(notsymdata, key = lambda x:x[0]))
fit_data = notsymdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-4:]
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
import PEPS.analysis as a
fit_data = notsymdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = notsymdata[:, 0][-4:]
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'k')
plt.xlim(3, 10)
txt_coords = (3.1, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
fit_label = 'Fit'
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'k')
plt.xlim(3, 10)
txt_coords = (3.1, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
plt.draw()
from matplotlib import rcParams
plot_params = {'axes.labelsize': 12,   # 8
               'title.fontsize': 12,   # 8
               'legend.fontsize': 12,  # 8
               'font.size': 14,        # 8
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
from matplotlib import rcParams
plot_params = {'axes.labelsize': 12,   # 8
               'title.fontsize': 12,   # 8
               'legend.fontsize': 12,  # 8
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
from matplotlib import rcParams
plot_params = {'axes.labelsize': 12,   # 8
               'title.fontsize': 12,   # 8
               'legend.fontsize': 12,  # 8
               'font.size': 14,        # 8
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
from matplotlib import rcParams
plot_params = {'axes.labelsize': 14,   # 8
               'title.fontsize': 14,   # 8
               'legend.fontsize': 14,  # 8
               'font.size': 14,        # 8
               'xtick.labelsize': 14,  # 8
               'ytick.labelsize': 14,  # 8
               'lines.markersize': 14, # 8
               'savefig.bbox': 'tight',   # tight
               'savefig.pad_inches': 0.1, # 0.1
               # 'text.usetex': True,
               'font.family': 'serif',
               #'text.latex.preamble' : [r'\usepackage{amsmath}'],
               'figure.figsize': (10, 8)}
rcParams.update(plot_params)
plt.draw()
plt.close()
axs = []
grid_shape = (2,3)
main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
for indx in xrange(2):
    axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
from matplotlib import rcParams
plot_params = {'axes.labelsize': 12,   # 8
               'title.fontsize': 12,   # 8
               'legend.fontsize': 12,  # 8
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
axs = []
grid_shape = (2,3)
main_ax = plt.subplot2grid(grid_shape, (0,0), rowspan = 2, colspan=2)
for indx in xrange(2):
    axs.append(plt.subplot2grid(grid_shape, (indx, 2)))
spec_points.plot_toolbox.spec_plot(spec_points.filter([],[lambda L:L>4], L='L'), 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min), ax = main_ax)
plt.draw()
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'k')
plt.xlim(3, 10)
txt_coords = (3.1, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
plt.cla()
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'b')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'b')
plt.xlim(3, 10)
txt_coords = (3.1, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
plt.cla()
notsymmetric_inverse_corr.plot('L', 'Correlation Bound', ax = axs[0], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Overall Correlation Length Bound', label = 'Data', marker = '*', color = 'b')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'b')
plt.xlim(3, 10)
txt_coords = (3.6, 2.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
plt.tight_layout()
symdata = symmetric_inverse_corr2.view_props(['L', 'Correlation Bound'])
symdata = np.array([[L, val] for L, val in symdata])
symdata = np.array(sorted(symdata, key = lambda x:x[0]))
plt.sca(axs[1])
fit_data = symdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.power_law
fit_Ls = symdata[:, 0][-4:]
fit_label = 'Fit'
txt_coords = (3.1, 0.1)
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
symmetric_inverse_corr2.plot('L', 'Correlation Bound', ax = axs[1], xlabel = 'L', ylabel='Correlation Length Bound', title = 'Symmetric Correlation Length Bound', label = 'Data', marker = '*', color = 'k')
plt.plot(extrapolate_pnts, fit_data, label = fit_label, color = 'k')
plt.xlim(3,10)
txt_coords = (3.5, 0.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
fit_data = symdata[:, 1][-4:]
extrapolate_pnts = np.arange(3, 10, 0.1)
fit_func = a.expic
fit_Ls = symdata[:, 0][-4:]
fit_label = 'Fit'
txt_coords = (3.1, 0.1)
(fit_data, params, R) = a.curve_fit_main(fit_data,fit_func, fit_Ls, extrapolate_pnts)
plt.xlim(3,10)
txt_coords = (3.5, 0.1)
txt = plt.text(txt_coords[0], txt_coords[1],  fit_label+': A-C*Exp(-L/B),\n A ={} \n B={} \nC={}'.format(*params))
plt.sca(main_ax)
grid = plt.grid()
grid = plt.grid()
plt.grid(b=0)
