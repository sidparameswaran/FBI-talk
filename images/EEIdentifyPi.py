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

from sfig import ticker
# from matplotlib import rcParams, ticker
# plot_params = {'axes.labelsize': 14,   # 8
               # 'title.fontsize': 14,   # 8
               # 'legend.fontsize': 14,  # 8
               # 'font.size': 14,        # 8
               # 'xtick.labelsize': 14,  # 8
               # 'ytick.labelsize': 14,  # 8
               # 'lines.markersize': 16, # 8
               # 'savefig.bbox': 'tight',   # tight
               # 'savefig.pad_inches': 0.1, # 0.1
               # 'text.usetex': True,
               # 'font.family': 'serif',
               # 'text.latex.preamble' : [r'\usepackage{amsmath}'],
               # 'figure.figsize': (10, 7)}  # (3.5, 2.5) default
# rcParams.update(plot_params)

i=10
stri = str(i)
IO.go_to_data_parent('interpolatedboson/a'+stri)
IO.add_to_path()
state = IO.get_state()
Ls = range(10, 11, 1)
L_min, L_max = min(Ls), max(Ls)
shift_func = lambda L: -L/2
#shift_func = lambda L: 0.2*(L-L_max)/(L_max - L_min)
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
                                  xlim=[-2.1, 2.1], ylim=[-1, 20])
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

# if i!=10 and i!=0:
    # plt.title('Entanglement Spectrum for b=0.'+stri)
# elif i==10:
    # plt.title('Entanglement Spectrum for soft-core FBI')
# elif i==0:
    # plt.title('Entanglement Spectrum for hard-core FBI')
ax = plt.gca()
Ks = range(0, 2)

plt.title('')
plt.ylabel('Entanglement Energy $EE-4\kappa^2$')
plt.xlabel('Momentum K')

plt.xlim([-1.45, 1.45])
plt.ylim([10, 16])

# #x major ticks
ax.set_xticks([-1, -1/2, 0, 1/2, 1])
# ax.xaxis.set_tick_params(which='major', direction='out', pad=12)
ax.set_xticklabels([r'$\pi-\frac{2\pi}{L}$', '', r'$\pi$', '', r'$\pi+\frac{2\pi}{L}$'])
#ax.set_xticklabels([r'K-$\pi$($\frac{2\pi}{L}$):$\qquad$'+str(K) if K==0 else str(K) for K in Ks], ha='right')
#
# #x minor ticks
# ax.xaxis.set_tick_params(which='minor', direction='out')#, pad=-14)
# ax.set_xticks([K+shift_func(L) for K in Ks for L in Ls], minor=True)
# ax.set_xticklabels(['L:$\quad$'+str(L) if (K==0 and L==10) else str(L) for K in Ks for L in Ls], minor=True, ha='right')
#
# y ticks
kappa = 6.53604/4

ax.set_yticks([4*kappa**2+y for y in [0, 1, 4*kappa]])
ax.set_yticklabels(['0', '1','$4\kappa$'])
ax.legend().set_visible(False)

#
eps = 0.1
hlin1=plt.hlines([4*kappa**2+0], -eps, +eps, colors='k', linestyles='dotted')
hlin1=plt.hlines([4*kappa**2+1], 1-eps, 1+eps, colors='k', linestyles='dotted')
hlin1=plt.hlines([4*kappa**2+1], -1-eps, -1+eps, colors='k', linestyles='dotted')
#hlin1=plt.hlines([1/4, 9/4], shift_func(L_min)-eps, shift_func(L_min)+eps, colors='k', linestyles='-')
#hlin2=plt.hlines([4, 9, 16], shift_func(L_max)-eps, shift_func(L_max)+eps, colors='k', linestyles='dotted')
#hlin2=plt.hlines([25/4, 49/4], shift_func(L_min)-eps, shift_func(L_min)+eps, colors='k', linestyles='dotted')

texts = []
def pop_text():
    texts[-1].remove()
    plt.draw()

text = plt.text(-0.3, 4*kappa**2+0.4,'$e=0$, $m=\pm1$', color='k')
texts.append(text)
text = plt.text(1-0.4, 4*kappa**2+1+0.4,'$e=\pm1$, $m=\pm1$', color='b')
texts.append(text)
text = plt.text(-1-0.4, 4*kappa**2+1+0.4,'$e=\pm1$, $m=\mp1$', color='b')
texts.append(text)

plt.annotate('', (0.95, 4*kappa**2+4), xycoords='data',
                xytext=(0.3, 4*kappa**2+1), textcoords='data',
                size=10, arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))
plt.text(0.3, 4*kappa**2+2, r'$j_{-1}$')
plt.text(-0.5, 4*kappa**2+2, r'$\bar{j}_{-1}$')
plt.annotate('', (-0.95, 4*kappa**2+4), xycoords='data',
                xytext=(-0.3, 4*kappa**2+1), textcoords='data',
                size=10, arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

plt.annotate('', (0.1, 4*kappa**2+5), xycoords='data',
                xytext=(1, 4*kappa**2+2), textcoords='data',
                size=10, arrowprops=dict(arrowstyle="simple",
                                fc=(0.4, 0.6, 0.8), ec="none"))
plt.annotate('', (-0.1, 4*kappa**2+5), xycoords='data',
                xytext=(-1, 4*kappa**2+2), textcoords='data',
                size=10, arrowprops=dict(arrowstyle="simple",
                                fc=(0.4, 0.6, 0.8), ec="none"))

plt.text(0.3, 4*kappa**2+3.2, r'$j_{-1}$', color='b')
plt.text(-0.5, 4*kappa**2+3.2, r'$\bar{j}_{-1}$', color='b')

#plt.annotate(', (-1, 4*kappa**2+3))
#plt.arrow(0, 4*kappa**2+0.8, -0.8, 3, fc="k", ec="k", head_width=0.2, head_length=0.3)


# text = plt.text(0.2,1/4-0.5,'$1/2$', color='m')
# texts.append(text)
# text = plt.text(-0.35,1-0.5,'$1$', color='b')
# texts.append(text)
# text = plt.text(0.2, 9/4-0.5,'$3/2$', color='orangered')
# texts.append(text)
# text = plt.text(-0.35,4-0.5,'$2$', color='g')
# texts.append(text)
# text = plt.text(0.2, 25/4-0.5,'$5/2$', color='#CD7F32')
# texts.append(text)
# text = plt.text(-0.35,9-0.5,'$3$', color='r')
# texts.append(text)
# text = plt.text(0.2, 49/4-0.5-0.4,'$7/2$', color='#493D26')
# texts.append(text)
# text = plt.text(-0.45, 51/4-0.5,'$0_{1, 1}$', color='k')
# texts.append(text)
# text = plt.text(0.15, 52/4-0.5,'$1/2_{1, 1}$', color='m')
# texts.append(text)
# text = plt.text(-0.45, 56/4-0.5,'$1_{1, 1}$', color='b')
# texts.append(text)
# text = plt.text(0.15, 60/4-0.5,'$3/2_{1, 1}$', color='orangered')
# texts.append(text)
# text = plt.text(-0.35, 65/4-0.5-0.05,'$4$', color='orange')
# texts.append(text)
# text = plt.text(-0.45, 69/4-0.5,'$2_{1, 1}$', color='green')
# texts.append(text)
# text = plt.text(1-0.45,26/4-0.5,'$0_{1, 0}$', color='k')
# texts.append(text)
# text = plt.text(1.15,27/4-0.5-0.1,'$1/2_{1, 0}$', color='m')
# texts.append(text)
# text = plt.text(1-0.45,1+26/4-0.5,'$1_{1, 0}$', color='b')
# texts.append(text)
# text = plt.text(1.15,9/4+26/4-0.5-0.1,'$3/2_{1, 0}$', color='orangered')
# texts.append(text)
# text = plt.text(1-0.45,4+26/4-0.5,'$2_{1, 0}$', color='green')
# texts.append(text)
# text = plt.text(1.15,25/4+26/4-0.5-0.1,'$5/2_{1, 0}$', color='#CD7F32')
# texts.append(text)
# text = plt.text(1-0.45,9+26/4-0.5,'$3_{1, 0}$', color='r')
# texts.append(text)
# text = plt.text(1-0.45,11.5+26/4-0.5,'$0_{2, 1}$', color='k')
# texts.append(text)
# text = plt.text(1.15,11.5+1/4+25/4-0.5-1,'$7/2_{1, 0}$', color='k')
# texts.append(text)
# text = plt.text(1.15,11.5+1/4+26/4-0.5,'$1/2_{2, 1}$', color='m')
# texts.append(text)
# text = plt.text(1-0.45,11.5+1+26/4-0.5,'$1_{2, 1}$', color='b')
# texts.append(text)
# text = plt.text(2-0.45,11.25-0.5,'$0_{2, 0}$', color='k')
# texts.append(text)
# text = plt.text(2.15,2/4+11.25-0.5-0.1,'$1/2_{2, 0}$', color='m')
# texts.append(text)
# text = plt.text(2-0.45,1+2/4+11.25-0.5,'$1_{2, 0}$', color='b')
# texts.append(text)
# text = plt.text(2.15,1+3/4+2/4+11.25-0.5-0.1,'$3/2_{2, 0}$', color='orangered')
# texts.append(text)
# text = plt.text(2.15,4+3/4+2/4+11.25-0.5-0.1,'$5/2_{2, 0}$', color='#CD7F32')
# texts.append(text)
# text = plt.text(2-0.45,4+11.25-0.5,'$2_{2, 0}$', color='g')
# texts.append(text)


#text = plt.text(1.7, 1, 'L')
#text.set_y(0.25)
plt.tight_layout()
plt.show()