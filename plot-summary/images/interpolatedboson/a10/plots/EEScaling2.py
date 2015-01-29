from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
i=10
stri = str(i)
IO.go_to_data_parent('interpolatedboson/a'+stri)
IO.add_to_path()
state = IO.get_state()
Ls = range(2, 11, 2)
L_min, L_max = min(Ls), max(Ls)
shift_func = lambda L: 0.3*(L-L_max-1)/(L_max - L_min)
#shift_func = lambda L: 0
params = IO.Params.kw('LKN', L=Ls, K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p, overwrite=True)

spectrum = pp.load_edge_spectrum(state, Ls)
spectrum = pp.shift_boson_number(spectrum)

spectrum.add_as_prop(pp.energy, val='val')
data_name = 'energy'

pp.shift_and_scale(spectrum, scale=0) #just shift
data_name = 'E'
all_points = list(spectrum.points())
params = IO.Params.kw('LKN', L=Ls, K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p, overwrite=False)

spectrum = pp.load_edge_spectrum(state, Ls)
spectrum = pp.shift_boson_number(spectrum)

spectrum.add_as_prop(pp.energy, val='val')
data_name = 'energy'

pp.shift_and_scale(spectrum, scale=0) #just shift
data_name = 'E'
all_points = list(spectrum.points())
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', data_name)
def get_spec_point(K, N, band):
    return spec_points[data.odict(zip(['K', 'N', 'band'],[K,N,band]))]
def try_fit(xdata, ydata, func, color, weights, max_factor, txt_coords, txt):
    extrap = np.arange(min(xdata), max(xdata)*max_factor, 0.1)
    interp, params, R = a.curve_fit_xy(xdata, ydata, func, extrap, weights=weights)
    #plt.plot(xdata, ydata, ls='', marker='.', markersize=4)
    plt.plot(extrap, interp, ls='-', color=color)
    text = plt.text(txt_coords[0], txt_coords[1], txt+str(params))
    plt.xlim(0, max(xdata)*max_factor)
    plt.ylim(0, 1.2*max(ydata))
    return params, R, text
from matplotlib import rcParams, ticker
plot_params = {'axes.labelsize': 8,   # 8
               'title.fontsize': 8,   # 8
               'legend.fontsize': 8,  # 8
               'font.size': 8,        # 8
               'xtick.labelsize': 8,  # 8
               'ytick.labelsize': 8,  # 8
               'lines.markersize': 8, # 8
               'savefig.bbox': 'tight',   # tight
               'savefig.pad_inches': 0.1, # 0.1
               'text.usetex': True,
               'font.family': 'serif',
               'text.latex.preamble' : [r'\usepackage{amsmath}'],
               'figure.figsize': (3.5, 2.5)}  # (3.5, 2.5) default
rcParams.update(plot_params)
def E(K, N, band):
    return get_spec_point(K, N, band).view('E')
def LL(K, N, band):
    return get_spec_point(K, N, band).view('L')
def try_fit(xdata, ydata, func, color, weights, max_x, txt_coords, txt):
    extrap = np.arange(min(xdata), max_x+0.01, 0.01)
    interp, params, R = a.curve_fit_xy(xdata, ydata, func, extrap, weights=weights)
    #plt.plot(xdata, ydata, ls='', marker='.', markersize=4)
    line = plt.plot(extrap, interp, ls='-', color=color)
    text = plt.text(txt_coords[0], txt_coords[1], txt+str(params))
    return params, R, line, text
def remove_most_recent_line(ax):
    lines = ax.get_lines()
    line = lines[-1]
    line.remove()
    plt.draw()
def try_fit(xdata, ydata, func, color, weights, max_x, txt_coords, txt_func):
    extrap = np.arange(min(xdata), max_x+0.01, 0.01)
    interp, params, R = a.curve_fit_xy(xdata, ydata, func, extrap, weights=weights)
    #plt.plot(xdata, ydata, ls='', marker='.', markersize=4)
    line = plt.plot(extrap, interp, ls='-', color=color)
    text = plt.text(txt_coords[0], txt_coords[1], txt_func(params))
    return params, R, line, text
pl_txt = lambda params: 'B='+str(IO.round_sig(params[0]))+'\nC='+str(IO.round_sig(params[1]))
#fig = plt.plot()
#ax = plt.gca()
#plt.xlim(1, 14)
#plt.xscale('log')
#ax.set_xticks(range(2, 14, 2))
#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(IO.round_sig(x, 2)))))
#plt.ylim(0.04, 4.1)
#plt.yscale('log')
#ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
#ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
#plt.draw()
#plt.xlabel('L', fontsize=14)
#plt.ylabel('Entanglement Energy $E=-Log(\\rho/\\rho_0)$', fontsize=14)
#plt.title('Entanglement energy versus system size', fontsize=14)
labels = ['$\\vert 1, 0 \\rangle$',
 u'$j_{-1} \\vert 0, 0 \\rangle$',
 u'$\\vert 2, 0 \\rangle$',
 u'$\\vert 3, 0 \\rangle$',
 u'$j_{-1}\\vert 2, 0 \\rangle$']
#plt.xlim(4, 12)
#plt.ylim(0.09, 3.6)
#ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
#ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
#plt.draw()
from PEPS import analysis as a
pl = lambda x, a, b: b*x**(-a)
pl3_txt = lambda params: 'B='+str(IO.round_sig(params[0],3))+'\nC='+str(IO.round_sig(params[1],3))
#params, R, line, text = try_fit(LL(0, 2, 0), E(0, 2, 0), pl, 'b', LL(0, 2, 0), 12, [6, 0.19], pl_txt)
#remove_most_recent_line()
#text.remove()
#plt.draw()
#remove_most_recent_line(ax)
#text.remove()
#plt.draw()
fig = plt.plot()
ax = plt.gca()
plt.xlim(1, 14)
plt.xscale('log')
ax.set_xticks(range(2, 14, 2))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(IO.round_sig(x, 2)))))
plt.ylim(0.04, 4.1)
plt.yscale('log')
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
plt.draw()
plt.xlim(4, 12)
plt.ylim(0.09, 3.6)
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
plt.draw()
plt.plot(LL(0, 2, 0), E(0, 2, 0), label=labels[0], marker='.', color='b', ls='')
plt.tight_layout()
plt.xlim(xmin=3.9)
#params, R, line, text = try_fit(LL(0, 2, 0)[1:], E(0, 2, 0)[1:], pl, 'b', [1/x for x in LL(0, 2, 0)[1:]], 10.9, [11, .15], pl_txt)
#remove_most_recent_line(ax)
#text.remove()
#plt.draw()
params, R, line, text = try_fit(LL(0, 2, 0)[1:], E(0, 2, 0)[1:], pl, 'b', [1/x for x in LL(0, 2, 0)[1:]], 10.9, [11, .15], pl3_txt)
plt.plot(LL(0, 4, 0), E(0, 4, 0), label=labels[1], marker='.',markersize=10, color='g', ls='')
#params, R, line, text = try_fit(LL(0, 4, 0), E(0, 4, 0), pl, 'g', [1/x for x in LL(0, 4, 0)], 10.9, [11, .5], pl3_txt)
#7.17/1.63
#remove_most_recent_line(ax)
#text.remove()
#plt.draw()
#params, R, line, text = try_fit(LL(0, 4, 0), E(0, 4, 0), pl, 'g', [1/x**2 for x in LL(0, 4, 0)], 10.9, [11, .5], pl3_txt)
#7.14/1.63
#remove_most_recent_line(ax)
#text.remove()
#plt.draw()
#remove_most_recent_line(ax)
#plt.draw()
#remove_most_recent_line(ax)
#plt.draw()
fig = plt.plot()
ax = plt.gca()
plt.xlim(1, 14)
plt.xscale('log')
ax.set_xticks(range(2, 14, 2))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(IO.round_sig(x, 2)))))
plt.ylim(0.04, 4.1)
plt.yscale('log')
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
plt.draw()
plt.xlim(3.9, 12)
plt.ylim(0.09, 3.6)
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
plt.draw()
plt.plot(LL(0, 2, 0), E(0, 2, 0), label=labels[0], marker='.',markersize=10, color='b', ls='')
params, R, line, text = try_fit(LL(0, 2, 0)[1:], E(0, 2, 0)[1:], pl, 'b', [1/(2**x) for x in LL(0, 2, 0)[1:]], 10.9, [11, .15], pl3_txt)
plt.xlim(3.9, 14)
plt.ylim(0.09, 3.6)
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))
plt.draw()
params, R, line, text = try_fit(LL(0, 4, 0), E(0, 4, 0), pl, 'g', [1/2**x for x in LL(0, 4, 0)], 10.9, [11, .5], pl3_txt)
7.06/1.67
plt.plot(LL(0, 4, 0), E(0, 4, 0), label=labels[1], marker='.',markersize=10, color='g', ls='')
E(0, 4, 0)[-1]/E(0, 2, 0)[-1]
plt.plot(LL(1, 0, 0), E(1, 0, 0), label=labels[2], marker='.',markersize=10, color='k', ls='')
params, R, line, text = try_fit(LL(1, 0, 0)[1:], E(1, 0, 0)[1:], pl, 'k', [1/2**x for x in LL(1, 0, 0)[1:]], 10.9, [11, 0.8], pl_txt)
plt.plot(LL(1, 2, 0), E(1, 2, 0), label='$j_{-1}\\vert 1, 0 \\rangle$', marker='.',markersize=10, color='c', ls='')
plt.plot(LL(0, 6, 0), E(0, 6, 0), label=labels[3], marker='.',markersize=10, color='r', ls='')
params, R, line, text = try_fit(LL(0, 6, 0), E(0, 6, 0), pl, 'r', [1/2**x for x in LL(0, 6, 0)], 10.9, [11, 1.4], pl3_txt)
17/1.67
text = plt.text(11, 1.9, 'Fit:\n $C/L^B$\n', fontsize=14)
text.remove()
plt.draw()
text = plt.text(11, 1.9, 'Fit:\n $C/L^B$', fontsize=14)
text.set_y(1.95)
plt.draw()
text.set_text('Fit: $C/L^B$')
plt.draw()
text.set_x(9.9)
text.set_y(2.0)
plt.draw()
text.set_x(9.8)
text.set_y(2.1)
plt.draw()
text.set_x(9.7)
text.set_y(2.2)
plt.draw()
plt.legend(loc=3, fontsize=14, numpoints=1)
handles, labels = ax.get_legend_handles_labels()
labels
plt.xlabel('L', fontsize=14)
plt.ylabel('Entanglement Energy $E=-Log(\\rho/\\rho_0)$', fontsize=14)
plt.title('Entanglement energy versus system size', fontsize=14)
plt.tight_layout()
plt.xlim(xmax=13.1)
text.set_x(9.9)
text.set_y(2.2)
plt.draw()
text.set_x(9.9)
text.set_y(2.1)
plt.draw()
plt.show()
plt.tight_layout()
