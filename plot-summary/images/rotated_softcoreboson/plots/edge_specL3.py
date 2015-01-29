# coding: utf-8
from __future__ import division
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
#IO.go_to_data_parent('rotated_softcoreboson')
state = IO.get_state()
params = IO.Params.kw('LKN', L=[1,2, 3], K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p)
reload(run)
params = IO.Params.kw('LKN', L=[1,2, 3], K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p)
def rotated_state_params(Ls):
    params = data.pdict.kw('LKN', L = Ls, K = range(max(Ls)), N = range(-max(Ls), max(Ls)+1))
    K_lt_L = lambda p: p['K']<p['L']
    N_le_L = lambda p: -p['L'] <= p['N'] <= p['L']
    params.add_constraints([K_lt_L, N_le_L])
    return params

Ls = [1, 2, 3]
L_min = min(Ls)
L_max = max(Ls)

params = rotated_state_params(Ls)
spectrum = data.ParametrizedDataSet(params)
for L in Ls:
    edge_spectrum_list = run.load_edge_spectrum(state, data.OrderedDict(zip(['L', 'K', 'N'], [L, 0, 0])))
    edge_spectrum_dict = spec.collect_spectrum_points(edge_spectrum_list)
    for K, N in edge_spectrum_dict:
        for indx, val in enumerate(edge_spectrum_dict[(K, N)]):
            spectrum.append(data.DataPoint(L=L, K=K, N=N, val=val, band=indx))
pp.shift_and_scale(spectrum, scale=0)
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', 'E')  # ParametrizedDataSet with K, N, band as parameters
spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                 xlim=[-0.7, 4.1], ylim=[-0.1, 10])
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, 'E',
                                                shift_func=lambda L: 0.3*(L-L_max-1)/(L_max - L_min))
axs = []
