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
from sfig import ticker, rcParams

makelegend = False
if makelegend:
    rcParams.update({'figure.figsize':(6.25, 2.5)})
else:
    rcParams.update({'figure.figsize':(3.75, 2.5)})

def go(L):
    #with PdfPages('L_10_all_mom_10.pdf') as pdf:
    ###########################################################
    #get state
    i=10
    stri = str(i)
    IO.go_to_data_parent('interpolatedboson/a'+stri)
    IO.add_to_path()
    state = IO.get_state()
    Ls = [L]
    L_min, L_max = min(Ls), max(Ls)
    shift_func = lambda L: 0
    ###########################################################
    #make edge spectrum if necessary
    params = IO.Params.kw('LKN', L=Ls, K=[0], N=[0])
    for p in params:
        run.create_edge_spectrum(state, p, overwrite=False)
    ###########################################################
    #load and prepeare edge spectrum
    spectrum = pp.load_edge_spectrum(state, Ls)
    spectrum = pp.shift_boson_number(spectrum)
    spectrum.add_as_prop(pp.energy, val='val')
    data_name = 'energy'
    pp.shift_and_scale(spectrum, scale=0) #just shift
    data_name = 'E'
    all_points = list(spectrum.points())

    spec_points = spectrum.collect(['K', 'N', 'band'], 'L', data_name)
    spec_points = spec_points.filter([lambda p: p['K']<=L/2])
    spec_points.plot_toolbox = data.SpectraPlotTools()
    spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
    spec_points.update_plot_args(title='', xlabel='K', ylabel='Energy',
                                      xlim=[-0.5, 12], ylim=[-0.4, 6])
    #even_colors = ['k', 'b', 'g', 'r', 'gold', 'orange', 'orangered', 'y','m', 'lime', 'c']    
    color_pairs = [('k', 'k'), ('b', 'c'),('g', 'lime'),('r', 'm'),('orange', 'orangered'),('gold', 'y')] 
    colors = [c for cp in color_pairs for c in cp]
    def color_func(N):
        x=0
        if N<0:
            x=1
        if N%2==0: 
            return abs(N)+x
        else:
            return abs(N)+1+4+x

    spec_points.plot_toolbox.colors = colors
    spec_points.plot_toolbox.color_func = color_func
    ###########################################################
    #make plot
    if makelegend:
        xlim=[-0.5, np.floor(L/2)+1.1]
    else:
        xlim=[-0.5, np.floor(L/2)+.5]

    fig, spec_ax = spec_points.plot_toolbox.spec_plot(spec_points, data_name,
                                                     shift_func=shift_func, copy_points=True,xlim=xlim, legend=None)

    # if i!=10 and i!=0:
        # plt.title('Entanglement Spectrum for b=0.'+stri)
    # elif i==10:
        # plt.title('Entanglement Spectrum for soft-core state')
    # elif i==0:
        # plt.title('Entanglement Spectrum for hard-core FBI')
    if L==10:
        plt.ylabel('Entanglement Energy')
    if L==9:
        plt.ylabel('')
    ax = plt.gca()
    Ks = np.arange(0, np.floor(L/2+1))
    ax.set_xticks(Ks)
    if L==10:
        ax.set_xticklabels(['0', '$\pi/5$', '$2\pi/5$', '$3\pi/5$', '$4\pi/5$', '$\pi$'])
    if L==9:
        ax.set_xticklabels(['0', '$2\pi/9$', '$4\pi/9$', '$6\pi/9$', '$8\pi/9$'])  
    #ax.set_xticklabels(['$2\pi'+str(K)+'/'+str(L)+'$' if K in [1for K in Ks])
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    if makelegend:
        handles, labels = ax.get_legend_handles_labels()
        hl = sorted(zip(handles, labels),
                key= lambda x: abs(int(x[1])))
        handles2, labels2 = zip(*hl)
        def label_helper(string_representing_integer):
            i = int(string_representing_integer)
            if not i%2:
                if i<=0:
                    return str(i//2)
                else:
                    return '+'+str(i//2)
            else:
                if i<=0:
                    return str(i)+'/2'
                else:
                    return '+'+str(i)+'/2'
                
        labels3 = [label_helper(l) for l in labels2]
        if L==9:
            legendloc = (0.8, 0.15)
        if L==10:
            legendloc = (0.825, 0.15)
        leg=ax.legend(handles2, labels3, loc = legendloc, numpoints=1, frameon=False, labelspacing=0.25)
    ##leg.get_frame().set_linewidth(0.0)
    
    #plt.tight_layout()
    #plt.show()

    #pdf.savefig()
    #IO.mkdirs('plots')
    #plt.savefig(IO.os.getcwd()+'\\plots\\iso_es2')
    #plt.close()
    
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.sca(ax1)
go(10)
plt.sca(ax2)
go(9)
plt.tight_layout()
plt.show()

