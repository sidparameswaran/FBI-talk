# coding: utf-8
get_ipython().system(u'explorer .')
from __future__ import division
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
IO.go_to_data_parent('armchair_softcoreboson')
state = IO.get_state()
params = IO.Params.kw('LKN', L=[1,2, 3], K=[0], N=[0])
from PEPS import spectra_analyzer as spec
def armchair_state_params(Ls):
    params = data.pdict.kw('LKN', L = Ls, K = range(max(Ls)), N = range(-max(Ls),                                                                  2*max(Ls)+1))
    K_lt_L = lambda p: p['K']<p['L']
    N_le_2L = lambda p: -p['L'] <= p['N'] <= 2*p['L']
    params.add_constraints([K_lt_L, N_le_2L])
    return params
Ls = [1, 2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N, val=val, band=indx))
Ls = [1, 2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N, val=val, band=indx))

pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo', 'k', 'b', 'r']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [3, 0, 0])))
spec.edge_spectrum_plot(3, edge_spectrum_list, connect_bands=False, scale=2)
spectrum
get_ipython().magic(u'cls ')
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
next(spectrum.points())
next(iter(spectrum.points()))
len(spectrum)
len(list(spectrum.points())
)
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
len(list(spectrum.points()))
type(spectrum.points())
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N, val=val, band=indx))
len(list(spectrum.points()))
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K+L, N=N, val=val, band=indx))
def armchair_state_params(Ls):
    params = data.pdict.kw('LKN', L = Ls, K = range(max(Ls)), B = range(0,                                                                  3*max(Ls)+1))
    K_lt_L = lambda p: p['K']<p['L']
    N_le_2L = lambda p: 0 <= p['N'] <= 3*p['L']
    params.add_constraints([K_lt_L, N_le_2L])
    return params
def armchair_state_params(Ls):
    params = data.pdict.kw('LKB', L = Ls, K = range(max(Ls)), B = range(0,                                                                  3*max(Ls)+1))
    K_lt_L = lambda p: p['K']<p['L']
    N_le_2L = lambda p: 0 <= p['B'] <= 3*p['L']
    params.add_constraints([K_lt_L, N_le_2L])
    return params
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, B=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo', 'k', 'b', 'r']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
def armchair_state_params(Ls):
    params = data.pdict.kw('LKN', L = Ls, K = range(max(Ls)), N = range(0,                                                                  3*max(Ls)+1))
    K_lt_L = lambda p: p['K']<p['L']
    N_le_2L = lambda p: 0 <= p['N'] <= 3*p['L']
    params.add_constraints([K_lt_L, N_le_2L])
    return params
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo', 'k', 'b', 'r']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r''g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r''g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r''g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r''g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=1)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
plt.ylim(ymax=30)
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=0)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
pp.shift_and_scale(spectrum, scale=0)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=0)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 2.1], ylim=[-0.1, 10])
spec_points.plot_toolbox.colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'hotpink', 'orangered', 'violet', 'indigo']
L_min = min(Ls)
L_max = max(Ls)
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
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
plt.xlim(xmax=2)
plt.xlim(xmax=1.3)
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
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
                                  xlim=[-0.7, 2.1], ylim=[-0.1, 7])


fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'chemscaled_E',
                                                 shift_func=lambda L: 0.1*\
                                             (L-L_max-1)/(L_max - L_min))
plt.xlim(xmax=1.3)
plt.xlim(xmin=-0.4)
plt.ylim(-0.3, 10)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
        key= lambda x: x[1])
handles2, labels2 = zip(*hl)
ax.legend(handles2[0:5], labels2[0:5], loc = 4, fontsize=14)
plt.draw()
plt.rcParams
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
                                  xlim=[-0.7, 2.1], ylim=[-0.1, 7])


fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'chemscaled_E',
                                                 shift_func=lambda L: 0.1*\
                                             (L-L_max-1)/(L_max - L_min))
Ls = [2, 3]
params = armchair_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N+L, val=val, band=indx))
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
                                  xlim=[-0.7, 2.1], ylim=[-0.1, 7])


fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'chemscaled_E',
                                                 shift_func=lambda L: 0.1*\
                                             (L-L_max-1)/(L_max - L_min))
plt.ylim(-0.3, 10)
plt.xlim(-0.4, 1.5)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
        key= lambda x: x[1])
handles2, labels2 = zip(*hl)
ax.legend(handles2[0:5], labels2[0:5], loc = 4, fontsize=14)
plt.draw()
plt.title('Entanglement Spectrum for Soft-core Boson (Armchair)', fontsize=14)
plt.ylabel('Energy after scaling and shifting', fontsize=14)
plt.xlabel('K', fontsize=14)
plt.tight_layout()
plt.tight_layout()
plt.title('Armchair Entanglement Spectrum', fontsize=14)
plt.hlines([0, 1], -0.3, 0-.05, colors='k', linestyles='dotted')
plt.hlines([2], 1-0.3, 1-.05, colors='k', linestyles='dotted')
get_ipython().magic(u'save plots/scaled-entanglement-spectrum.py 1-65')
get_ipython().system(u'explorer .')
alllines = ax.get_lines()
len(alllines)
line = alllines[0]
line.get_xydata()
lowlines = alllines.copy()
for indx, line in enumerate(alllines):
    yd = line.get_ydata()
lowlines = list(alllines).copy()
for indx, line in enumerate(alllines):
    yd = line.get_ydata()
from copy import copy
lowlines = list(alllines).copy()
for indx, line in enumerate(alllines):
    yd = line.get_ydata()
lowlines = copy(list(alllines))
for indx, line in enumerate(alllines):
    yd = line.get_ydata()
lowlines = copy(list(alllines))
for indx, line in enumerate(alllines):
    yd = line.get_ydata()
    if min(yd) > 7:
        del lowlines[indx]
lowlines = [line for line in alllines if min(line.get_ydata())< 6]
for line in lowlines:
    print line.get_ydata()
