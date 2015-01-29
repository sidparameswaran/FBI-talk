# coding: utf-8
from __future__ import division
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
IO.go_to_data_parent('softcoreboson')
state = IO.get_state()

Ls = range(5, 9)
L_min, L_max = min(Ls), max(Ls)

spectrum = pp.load_edge_spectrum(state, Ls)
pp.shift_and_scale(spectrum, scale=0) #just shift

all_points = list(spectrum.points())

scales = {}
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    K1N1 = filter(lambda d:d['K']==1 and d['N']==1 and d['L']==L, all_points)[0]
    scales[L] = K1N1['E'] - K0N1['E']

for point in all_points:
    point['scaled_E'] = point['E']/scales[point['L']]

chemical_potentials = {}    
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    chemical_potentials[L] = 1 - K0N1['scaled_E']

for point in all_points:
    point['chemscaled_E'] = point['scaled_E'] +                 chemical_potentials[point['L']]*point['N']

spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'chemscaled_E')

spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                  xlim=[-0.7, 4.1], ylim=[-0.1, 10])


fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'chemscaled_E',
                                                 shift_func=lambda L: 0.3*\
                                             (L-L_max-1)/(L_max - L_min))

plt.xlim(-0.5, 2.3)
plt.ylim(-0.1, 6.5)
plt.title('Entanglement Spectrum for Soft-core Boson', fontsize=14)
plt.ylabel('Energy after scaling and shifting', fontsize=14)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
        key= lambda x: x[1])
handles2, labels2 = zip(*hl)
ax.legend(handles2, labels2, loc = 'best', fontsize=14)

plt.hlines([0, 1, 2], -0.5, 2.3, colors='k', linestyles='dotted')
from __future__ import division
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
IO.go_to_data_parent('softcoreboson')
state = IO.get_state()

Ls = range(5, 9)
L_min, L_max = min(Ls), max(Ls)

spectrum = pp.load_edge_spectrum(state, Ls)
pp.shift_and_scale(spectrum, scale=0) #just shift

all_points = list(spectrum.points())

scales = {}
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    K1N1 = filter(lambda d:d['K']==1 and d['N']==1 and d['L']==L, all_points)[0]
    scales[L] = K1N1['E'] - K0N1['E']

for point in all_points:
    point['scaled_E'] = point['E']/scales[point['L']]

chemical_potentials = {}    
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    chemical_potentials[L] = 1 - K0N1['scaled_E']

for point in all_points:
    point['chemscaled_E'] = point['scaled_E'] +                 chemical_potentials[point['L']]*point['N']

spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'chemscaled_E')

spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                  xlim=[-0.7, 4.1], ylim=[-0.1, 10])


fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'chemscaled_E',
                                                 shift_func=lambda L: 0.3*\
                                             (L-L_max-1)/(L_max - L_min))

plt.xlim(-0.5, 2.3)
plt.ylim(-0.1, 6.5)
plt.title('Entanglement Spectrum for Soft-core Boson', fontsize=14)
plt.ylabel('Energy after scaling and shifting', fontsize=14)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
        key= lambda x: x[1])
handles2, labels2 = zip(*hl)
ax.legend(handles2[0:5], labels2[0:5], loc = 'best', fontsize=14)
plt.hlines([0, 1], -0.45, -0.05, colors='k', linestyles='dotted')
plt.hlines([2], 0.55, 0.95, colors='k', linestyles='dotted')
plt.tight_layout()
from __future__ import division
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
IO.go_to_data_parent('softcoreboson')
state = IO.get_state()

Ls = range(5, 9)
L_min, L_max = min(Ls), max(Ls)

spectrum = pp.load_edge_spectrum(state, Ls)
pp.shift_and_scale(spectrum, scale=0) #just shift

all_points = list(spectrum.points())

scales = {}
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    K1N1 = filter(lambda d:d['K']==1 and d['N']==1 and d['L']==L, all_points)[0]
    scales[L] = K1N1['E'] - K0N1['E']

for point in all_points:
    point['scaled_E'] = point['E']/scales[point['L']]

chemical_potentials = {}    
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    chemical_potentials[L] = 1 - K0N1['scaled_E']

for point in all_points:
    point['chemscaled_E'] = point['scaled_E'] +                 chemical_potentials[point['L']]*point['N']

spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'chemscaled_E')

spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                  xlim=[-0.7, 4.1], ylim=[-0.1, 10])


fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'chemscaled_E',
                                                 shift_func=lambda L: 0.3*\
                                             (L-L_max-1)/(L_max - L_min))

plt.xlim(-0.5, 2.3)
plt.ylim(-0.1, 6.5)
plt.title('Entanglement Spectrum for Soft-core Boson', fontsize=14)
plt.ylabel('Energy after scaling and shifting', fontsize=14)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
        key= lambda x: x[1])
handles2, labels2 = zip(*hl)
ax.legend(handles2[0:5], labels2[0:5], loc = 'best', fontsize=14)
plt.hlines([0, 1], -0.45, -0.05, colors='k', linestyles='dotted')
plt.hlines([2], 0.55, 0.95, colors='k', linestyles='dotted')
plt.tight_layout()
ax.legend(handles2[0:5], labels2[0:5], loc = 'best', fontsize=14)
ax.legend(handles2[0:5], labels2[0:5], loc = '4', fontsize=14)
ax.legend(handles2[0:5], labels2[0:5], loc = 4, fontsize=14)
plt.draw()
spec_points(data.odict(zip(['K', 'N', 'band'],[0,0,0])))
spec_points[data.odict(zip(['K', 'N', 'band'],[0,0,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[0,1,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[1,1,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[2,0,0]))].view('chemscaled_E')
spec_points[data.odict(zip(['K', 'N', 'band'],[1,0,0]))].view('chemscaled_E')
spec_points[data.odict(zip(['K', 'N', 'band'],[1,0,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[1,1,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[2,0,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))]
spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view('chemscaled_E')
spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view_props('L', 'chemscaled_E')
spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view_props(['L', 'chemscaled_E'])
spec_points[data.odict(zip(['K', 'N', 'band'],[0,3,0]))].view_props(['L', 'chemscaled_E'])
spec_points[data.odict(zip(['K', 'N', 'band'],[0,4,0]))].view_props(['L', 'chemscaled_E'])
6.411-6.3667
6.3667-6.3202
6.3202-6.26937
(2.436-2)/2
(4.25877-3)/6
(6.411-4)/12
spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view_props(['L', 'N','chemscaled_E'])
k2n0b0 = spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view_props(['L', 'chemscaled_E'])
k0n2b0 = spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view_props(['L', 'chemscaled_E'])
k0n3b0 = spec_points[data.odict(zip(['K', 'N', 'band'],[0,3,0]))].view_props(['L', 'chemscaled_E'])
k0n4b0 = spec_points[data.odict(zip(['K', 'N', 'band'],[0,4,0]))].view_props(['L', 'chemscaled_E'])
k0n2b0 = dict(spec_points[data.odict(zip(['K', 'N', 'band'],[0,2,0]))].view_props(['L', 'chemscaled_E']))
k0n3b0 = dict(spec_points[data.odict(zip(['K', 'N', 'band'],[0,3,0]))].view_props(['L', 'chemscaled_E']))
k0n4b0 = dict(spec_points[data.odict(zip(['K', 'N', 'band'],[0,4,0]))].view_props(['L', 'chemscaled_E']))
k0n2b0
for L in range(5, 9):
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, k0n2b0[L],k0n3b0[L],k0n4b0[L]]
    plt.plot(x_data, y_data)
for L in range(5, 9):
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, k0n2b0[L],k0n3b0[L],k0n4b0[L]]
    plt.plot(x_data, y_data, label = 'L = '+str(L), ls='.')
for L in range(5, 9):
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, k0n2b0[L],k0n3b0[L],k0n4b0[L]]
    plt.plot(x_data, y_data, label = 'L = '+str(L), ls='')
for L in range(5, 9):
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, k0n2b0[L],k0n3b0[L],k0n4b0[L]]
    plt.plot(x_data, y_data, label = 'L = '+str(L), marker = '*', ls='')
from PEPS import analysis as a
a.curve_fit_xy(x_data, y_data, lambda x, K:K(x^2-x)+x, np.arange(0, 4, 0.1))
a.curve_fit_xy(np.array(x_data), np.array(y_data), lambda x, K:K(x^2-x)+x, np.arange(0, 4, 0.1))
fit_points, params, R = a.curve_fit_xy(np.array(x_data), np.array(y_data), lambda x, K:K*(x^2-x)+x, np.arange(0, 4, 0.1))
fit_points, params, R = a.curve_fit_xy(np.array(x_data), np.array(y_data), lambda x, K:K*(x**2-x)+x, np.arange(0, 4, 0.1))
params
fit_points
zip(np.arange(0, 4, 0.1), fit_points)
y_data
fit_points, params, R = a.curve_fit_xy(np.array(x_data), np.array(y_data), lambda x, K:K*(x**2-x)+x, np.arange(0, 4.1, 0.1))
zip(np.arange(0, 4.1, 0.1), fit_points)
params
plt.plot(np.arange(0, 4.1, 0.1), fit_points, label = 'Fit', ls = ':')
plt.plot(np.arange(0, 4.1, 0.1), fit_points, label = 'Fit', ls = '-', color='k')
plt.xlim(-0.1,4.1)
plt.ylim(-0.1, 7)
plt.ylim(-0.2, 7)
plt.grid()
plt.xlabel('Boson Number e')
plt.xlabel('Boson Number e', fontsize=14)
plt.xlabel('Boson Number', fontsize=14)
plt.ylabel('Scaled Energy', fontsize = 14)
plt.title('Determining K using low-energy states', fontsize=14)
plt.text(0.5, 5, 'Fit $Ke^2 + (1-K)e: \n K = 0.20', fontsize=16)
_.remove()
plt.draw()
text = plt.text(0.5, 5, 'Fit $Ke^2 + (1-K)e$: \n $K = 0.20$', fontsize=16)
text.remove()
text = plt.text(0.5, 5, 'Energy fit: \n $H = Ke^2 + (1-K)e$: \n $K = 0.20$', fontsize=16)
plt.xlabel('Boson Number e', fontsize=14)
plt.ylabel('Scaled Energy H', fontsize = 14)
text.remove()
text = plt.text(0.49, 3, 'Energy fit: \n $H = Ke^2 + (1-K)e$: \n $K = 0.20$', fontsize=16)
leg = plt.legend()
leg = plt.legend(loc=4)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
labels[0]
labels[0, 1]
good = [0, 1, 2, 3, 5]
handles = [handles[indx] for indx in good]
labels = [labels[indx] for indx in good]
ax.legend(handles=handles, labels=labels)
ax.legend(handles, labels, loc=4, fontsize=14)
plt.draw()
text.remove()
plt.draw()
text = plt.text(0.4, 3, 'Energy fit using L=8: \n $H = Ke^2 + (1-K)e$: \n $K = 0.20$', fontsize=16)
plt.tight_layout()
plt.tight_layout()
IO.go_to_data_parent()
