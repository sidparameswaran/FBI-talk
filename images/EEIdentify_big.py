# coding: utf-8
from __future__ import division
#import matplotlib
#matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages
from DataAnalysis import data
from DataAnalysis import PEPS_plots as pp
import numpy as np
from matplotlib import pyplot as plt
from PEPS import IO_helper as IO
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker

from matplotlib import rcParams, ticker
plot_params = {'axes.labelsize': 14,   # 8
               'title.fontsize': 14,   # 8
               'legend.fontsize': 14,  # 8
               'font.size': 14,        # 8
               'xtick.labelsize': 14,  # 8
               'ytick.labelsize': 14,  # 8
               'lines.markersize': 16, # 8
               'savefig.bbox': 'tight',   # tight
               'savefig.pad_inches': 0.1, # 0.1
               'text.usetex': True,
               'font.family': 'serif',
               'text.latex.preamble' : [r'\usepackage{amsmath}'],
               'figure.figsize': (10, 7)}  # (3.5, 2.5) default
rcParams.update(plot_params)
i=10
stri = str(i)
IO.go_to_data_parent('interpolatedboson/a'+stri)
IO.add_to_path()
state = IO.get_state()
Ls = range(6, 11, 1)
L_min, L_max = min(Ls), max(Ls)
shift_func = lambda L: 0.4*(L-L_max-1)/(L_max - L_min)
#shift_func = lambda L: 0

params = IO.Params.kw('LKN', L=Ls, K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p, overwrite=False)

spectrum = pp.load_edge_spectrum(state, Ls)
spectrum = pp.shift_boson_number(spectrum)

spectrum.add_as_prop(pp.energy, val='val')
data_name = 'energy'

pp.shift_and_scale(spectrum, scale=1) #just shift
data_name = 'E'
all_points = list(spectrum.points())

for point in all_points:
    if not point['L']%2:
        point['shifted_E'] = point['E']
    else:
        point['shifted_E'] = 2*point['E']+1/4
data_name = 'shifted_E'
spec_points = spectrum.collect(['K', 'N', 'band'], 'L', data_name)

spec_points.plot_toolbox = data.SpectraPlotTools()
spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
#spec_points.plot_toolbox.markers
spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                  xlim=[-0.6, 2.4], ylim=[-1, 20])
#even_colors = ['k', 'b', 'g', 'r', 'gold', 'orange', 'orangered', 'y','m', 'lime', 'c']    
color_pairs = [('k', 'k'), ('b', 'c'),('g', 'lime'),('r', 'm'),('orange', 'orangered'),('gold', 'y')] 
colors = [c for cp in color_pairs for c in cp]
def color_func(N):
    x=0
    if N<0:
        x=1
    if N%2==0: 
        return abs(N)+x
    else: #
        return abs(N)+1+4+x
        # return 11-abs(N)+x
spec_points.plot_toolbox.colors = colors
spec_points.plot_toolbox.color_func = color_func
fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, data_name,
                                                 shift_func=shift_func)
if i!=10 and i!=0:
    plt.title('Entanglement Spectrum for b=0.'+stri, fontsize=14)
elif i==10:
    plt.title('Entanglement Spectrum for soft-core FBI', fontsize=14)
elif i==0:
    plt.title('Entanglement Spectrum for hard-core FBI', fontsize=14)
plt.ylabel('Energy', fontsize=14)
ax = plt.gca()
Ks = range(0, 3)
ax.set_xticks(Ks)
ax.set_xticks([K+shift_func(L) for K in Ks for L in Ls], minor=True)
ax.set_xticklabels([L for K in Ks for L in Ls], minor=True, fontsize=11)
ax.xaxis.set_tick_params(which='minor', direction='in', pad=-14)
ax.set_yticks([0, 1, 9/4, 4, 25/4, 9, 49/4, 16])
ax.set_yticklabels(['0', '1', '9/4', '4', '25/4', '9', '49/4', '16'])
ax.legend().set_visible(False)
print 'hi'
plt.draw()
hlin1=plt.hlines([0, 1/4, 1, 9/4], shift_func(L_min), shift_func(L_max), colors='k', linestyles='-')
hlin2=plt.hlines([4, 25/4, 9, 49/4, 16], shift_func(L_min), shift_func(L_max), colors='k', linestyles='dotted')
color_pairs = [('k', 'k'), ('b', 'c'),('g', 'lime'),('r', 'm'),('orange', 'orangered'),('gold', 'y')]
texts = []
text = plt.text(0,0,'$|0, 0\\rangle$', fontsize=10, color='k') 
text.remove()
plt.draw()
def pop_text():
    texts[-1].remove()
    plt.draw()
text = plt.text(0,0-0.25,'$|0, 0\\rangle$', fontsize=10, color='k')
texts.append(text)
text = plt.text(0.3,0.25-0.25,'$|1/2, 0\\rangle$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(0.2,0.25-0.25,'$|1/2, 0\\rangle$', fontsize=10, color='m')
texts.append(text)
text = plt.text(0,1-0.25,'$|1/2, 0\\rangle$', fontsize=10, color='b')
texts.append(text)
pop_text()
text = plt.text(0,1-0.25,'$|1, 0\\rangle$', fontsize=10, color='b')
texts.append(text)
text = plt.text(0.2, 9/4-0.25,'$|3/2, 0\\rangle$', fontsize=10, color='orangered')
texts.append(text)
text = plt.text(0,4-0.25,'$|2, 0\\rangle$', fontsize=10, color='g')
texts.append(text)
text = plt.text(0.2, 25/4-0.25,'$|5/2, 0\\rangle$', fontsize=10, color='yellow')
texts.append(text)
pop_text()
text = plt.text(0.2, 25/4-0.25,'$|5/2, 0\\rangle$', fontsize=10, color='gold')
texts.append(text)
pop_text()
text = plt.text(0.2, 25/4-0.25,'$|5/2, 0\\rangle$', fontsize=10, color='#FBB117')
texts.append(text)
pop_text()
text = plt.text(0.2, 25/4-0.25,'$|5/2, 0\\rangle$', fontsize=10, color='#CD7F32')
texts.append(text)
text = plt.text(0,9-0.25,'$|3, 0\\rangle$', fontsize=10, color='r')
texts.append(text)
text = plt.text(0.2, 49/4-0.25,'$|7/2, 0\\rangle$', fontsize=10, color='#493D26')
texts.append(text)
text = plt.text(0, 50/4-0.25,'$|0, 0\\rangle_{1, 1}$', fontsize=10, color='k')
texts.append(text)
pop_text()
text = plt.text(0, 51/4-0.25,'$|0, 0\\rangle_{1, 1}$', fontsize=10, color='k')
texts.append(text)
text = plt.text(0, 53/4-0.25,'$|1/2, 0\\rangle_{1, 1}$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(0.2, 53/4-0.25,'$|1/2, 0\\rangle_{1, 1}$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(0.2, 52/4-0.25,'$|1/2, 0\\rangle_{1, 1}$', fontsize=10, color='m')
texts.append(text)
text = plt.text(0, 57/4-0.25,'$|1, 0\\rangle_{1, 1}$', fontsize=10, color='b')
texts.append(text)
pop_text()
text = plt.text(0, 56/4-0.25,'$|1, 0\\rangle_{1, 1}$', fontsize=10, color='b')
texts.append(text)
text = plt.text(0.2, 60/4-0.25,'$|3/2, 0\\rangle_{1, 1}$', fontsize=10, color='orangered')
texts.append(text)
text = plt.text(0, 65/4-0.25,'$|4, 0\\rangle$', fontsize=10, color='orange')
texts.append(text)
text = plt.text(0, 70/4-0.25,'$|2, 0\\rangle$_{1, 1}', fontsize=10, color='green')
texts.append(text)
pop_text()
text = plt.text(0, 69/4-0.25,'$|2, 0\\rangle_{1, 1}$', fontsize=10, color='green')
texts.append(text)
text = plt.text(1,26/4-0.25,'$|0, 0\\rangle_{1, 0}$', fontsize=10, color='k')
texts.append(text)
text = plt.text(1.2 ,27/4-0.25,'$|1/2, 0\\rangle_{1, 0}$', fontsize=10, color='m')
texts.append(text)
text = plt.text(1,1+26/4-0.25,'$|1, 0\\rangle_{1, 0}$', fontsize=10, color='b')
texts.append(text)
text = plt.text(1.19,9/4+26/4-0.25,'$|3/2, 0\\rangle_{1, 0}$', fontsize=10, color='orangered')
texts.append(text)
text = plt.text(1,4+26/4-0.25,'$|2, 0\\rangle_{1, 0}$', fontsize=10, color='green')
texts.append(text)
text = plt.text(1.19,25/4+26/4-0.25,'$|5/2, 0\\rangle_{1, 0}$', fontsize=10, color='#CD7F32')
texts.append(text)
text = plt.text(1,9+26/4-0.25,'$|3, 0\\rangle_{1, 0}$', fontsize=10, color='r')
texts.append(text)
text = plt.text(1,11.5+26/4-0.25,'$|0, 0\\rangle_{2, 1}$', fontsize=10, color='k')
texts.append(text)
text = plt.text(1.19,11.5+1/4+26/4-0.25,'$|1/2, 0\\rangle_{2, 1}$', fontsize=10, color='m')
texts.append(text)
text = plt.text(1,11.5+1+26/4-0.25,'$|1, 0\\rangle_{2, 1}$', fontsize=10, color='b')
texts.append(text)
text = plt.text(1,11-0.25,'$|0, 0\\rangle_{2, 0}$', fontsize=10, color='k')
texts.append(text)
pop_text()
text = plt.text(2,11.25-0.25,'$|0, 0\\rangle_{2, 0}$', fontsize=10, color='k')
texts.append(text)
text = plt.text(2.2,1/4+11.25-0.25,'$|1/2, 0\\rangle_{2, 0}$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(2.18,2/4+11.25-0.25,'$|1/2, 0\\rangle_{2, 0}$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(2.16,2/4+11.25-0.25,'$|1/2, 0\\rangle_{2, 0}$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(2.15,2/4+11.25-0.25,'$|1/2, 0\\rangle_{2, 0}$', fontsize=10, color='m')
texts.append(text)
pop_text()
text = plt.text(2.14,2/4+11.25-0.25,'$|1/2, 0\\rangle_{2, 0}$', fontsize=10, color='m')
texts.append(text)
text = plt.text(2,1+2/4+11.25-0.25,'$|1, 0\\rangle_{2, 0}$', fontsize=10, color='g')
texts.append(text)
pop_text()
text = plt.text(2,1+2/4+11.25-0.25,'$|1, 0\\rangle_{2, 0}$', fontsize=10, color='b')
texts.append(text)
text = plt.text(2.14,1+3/4+2/4+11.25-0.25,'$|1, 0\\rangle_{3/2, 0}$', fontsize=10, color='orangered')
texts.append(text)
pop_text()
text = plt.text(2.14,1+3/4+2/4+11.25-0.25,'$|3/2, 0\\rangle_{2, 0}$', fontsize=10, color='orangered')
texts.append(text)
text = plt.text(2,4+11.25-0.25,'$|2, 0\\rangle_{2, 0}$', fontsize=10, color='g')
texts.append(text)
plt.title('Entanglement spectrum for soft-core state', fontsize=16)
plt.xlabel('Scaled Entanglement Energy', fontsize=16)
plt.ylabel('Scaled Entanglement Energy', fontsize=16)
plt.xlabel('Momentum K parallel to cut', fontsize=16)
plt.xlabel('Momentum K (parallel to cut)', fontsize=16)
#plt.text(1.75, 1, 'Circumference L', fontsize=16)
#_.remove()
plt.draw()
text = plt.text(1.7, 1, 'L', fontsize=16)
text.set_y(0.25)
plt.draw()
plt.show()
plt.tight_layout()
