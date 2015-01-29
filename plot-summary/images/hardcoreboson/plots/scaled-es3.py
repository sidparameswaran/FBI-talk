# coding: utf-8
IO.go_to_data_parent('hardcoreboson')
state = IO.get_state()

Ls = range(5, 9)
L_min, L_max = min(Ls), max(Ls)

spectrum = pp.load_edge_spectrum(state, Ls)
pp.shift_and_scale(spectrum, scale=0) #just shift

all_points = list(spectrum.points())
shifts = {}
for L in Ls:
    K0N0 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    shifts[L] = K0N0['E']
for point in all_points:
    point['shifted_E'] = point['E'] - shifts[point['L']]
K0N0
shifts = {}
for L in Ls:
    K0N0 = filter(lambda d:d['K']==0 and d['N']==0 and d['L']==L, all_points)[0]
    shifts[L] = K0N0['E']
for point in all_points:
    point['shifted_E'] = point['E'] - shifts[point['L']]
K0N0
scales = {}
for L in Ls:
    K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
    K1N1 = filter(lambda d:d['K']==1 and d['N']==1 and d['L']==L, all_points)[0]
    scales[L] = K1N1['shifted_E'] - K0N1['shifted_E']

for point in all_points:
    point['scaled_E'] = point['shifted_E']/scales[point['L']]

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
plt.title('Entanglement Spectrum for Hard-core Boson', fontsize=14)
plt.ylabel('Energy after scaling and shifting', fontsize=14)
plt.hlines([2], 1-0.5+0.05, 1-0.05, colors='k', linestyles='dotted')
plt.hlines([0, 1], -0.5+0.05, -0.05, colors='k', linestyles='dotted')
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels),
        key= lambda x: x[1])
handles2, labels2 = zip(*hl)
ax.legend(handles2[0:6], labels2[0:6], loc = 4, fontsize=14)
