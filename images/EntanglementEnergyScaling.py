# coding: utf-8
from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run

############ Go to state data
i=10
stri = str(i)
IO.go_to_data_parent('interpolatedboson/a'+stri)
IO.add_to_path()
state = IO.get_state()
Ls = range(2, 11, 2)
L_min, L_max = min(Ls), max(Ls)
shift_func = lambda L: 0.3*(L-L_max-1)/(L_max - L_min)

# if needed create edge spectrum
params = IO.Params.kw('LKN', L=Ls, K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p, overwrite=True)

# load and prepare spectrum    
spectrum = pp.load_edge_spectrum(state, Ls)
spectrum = pp.shift_boson_number(spectrum)
spectrum.add_as_prop(pp.energy, val='val')
data_name = 'energy'
pp.shift_and_scale(spectrum, scale=0) #just shift
data_name = 'E'
all_points = list(spectrum.points())
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', data_name)


# set plot details
from sfig import ticker

# from matplotlib import rcParams, ticker
# plot_params = {'axes.labelsize': 8,   # 8
               # 'title.fontsize': 8,   # 8
               # 'legend.fontsize': 8,  # 8
               # 'font.size': 8,        # 8
               # 'xtick.labelsize': 8,  # 8
               # 'ytick.labelsize': 8,  # 8
               # 'lines.markersize': 8, # 8
               # 'savefig.bbox': 'tight',   # tight
               # 'savefig.pad_inches': 0.1, # 0.1
               # 'text.usetex': True,
               # 'font.family': 'serif',
               # 'text.latex.preamble' : [r'\usepackage{amsmath}'],
               # 'figure.figsize': (3.5, 2.5)}  # (3.5, 2.5) default
# rcParams.update(plot_params)

#some helper functions
from PEPS import analysis as a

def get_spec_point(K, N, band):
    return spec_points[data.odict(zip(['K', 'N', 'band'],[K,N,band]))]
def E(K, N, band):
    return get_spec_point(K, N, band).view('E')
def LL(K, N, band):
    return get_spec_point(K, N, band).view('L')
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
    
#set up the plot

fig = plt.plot()
ax = plt.gca()
plt.xscale('log')
ax.set_xticks(range(2, 12, 2))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(IO.round_sig(x, 2)))))
plt.yscale('log')
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(IO.round_sig(x, 2))))

labels = ['$\\vert 1, 0 \\rangle$', u'$\\vert 2, 0 \\rangle$', u'$\\vert 3, 0 \\rangle$',
  u'$j_{-1} \\vert 0, 0 \\rangle$', '$j_{-1}\\vert 1, 0 \\rangle$']
 
pl = lambda x, a, b: b*x**(-a)
pl3_txt = lambda params: 'B='+str(IO.round_sig(params[0],3))+'\nC='+str(IO.round_sig(params[1],3))
pl_txt = lambda params: 'B='+str(IO.round_sig(params[0]))+'\nC='+str(IO.round_sig(params[1]))

text_x = 11.5
max_x_in_fit = 11.1

#line 2
plt.plot(LL(0, 6, 0), E(0, 6, 0), label=labels[2], marker='.', color='r', ls='')
params, R, line, text = try_fit(LL(0, 6, 0), E(0, 6, 0), pl, 'r', [1/2**x for x in LL(0, 6, 0)], max_x_in_fit, [text_x, 1.4], pl3_txt)

#line 4
plt.plot(LL(1, 2, 0), E(1, 2, 0), label=labels[4], marker='.',markersize=8, color='c', ls='')
#params, R, line, text = try_fit(LL(1, 2, 0)[1:], E(1, 2, 0)[1:], pl, 'k', [1/2**x for x in LL(1, 0, 0)[1:]], max_x_in_fit, [11, 0.8], pl3_txt)

#line 3 (j-1 |00>)
plt.plot(LL(1, 0, 0), E(1, 0, 0), label=labels[3], marker='.', color='k', ls='')
#params, R, line, text = try_fit(LL(1, 0, 0)[1:], E(1, 0, 0)[1:], pl, 'k', [1/2**x for x in LL(1, 0, 0)[1:]], max_x_in_fit, [text_x, 1.6], pl3_txt)

#line 1
plt.plot(LL(0, 4, 0), E(0, 4, 0), label=labels[1], marker='.', color='g', ls='')
params, R, line, text = try_fit(LL(0, 4, 0), E(0, 4, 0), pl, 'g', [1/2**x for x in LL(0, 4, 0)], max_x_in_fit, [text_x, .5], pl3_txt)

# line 0
plt.plot(LL(0, 2, 0), E(0, 2, 0), label=labels[0], marker='.', color='b', ls='')
params, R, line, text = try_fit(LL(0, 2, 0)[1:], E(0, 2, 0)[1:], pl, 'b', [1/2**x for x in LL(0, 2, 0)[1:]], max_x_in_fit, [text_x, .13], pl3_txt)

# final text
text = plt.text(text_x-0.6, 2.8, 'Fit: $C/L^B$')
#text.set_x(text_x-0.3)
#text.set_y(2.6)

# legend
plt.legend(loc=6, numpoints=1, frameon=False)
#handles, labels = ax.get_legend_handles_labels()
plt.xlabel('L')
plt.ylabel('Entanglement Energy $E=-Log(\\rho/\\rho_0)$')
plt.title('Entanglement energy versus system size')
plt.xlim(2.1, 16.5)
plt.ylim(0.1, 4.1)
plt.tight_layout()
plt.show()
