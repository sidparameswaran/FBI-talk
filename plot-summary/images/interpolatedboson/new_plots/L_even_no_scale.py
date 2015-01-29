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
               # 'text.usetex': True,
               'font.family': 'serif',
               #'text.latex.preamble' : [r'\usepackage{amsmath}'],
               'figure.figsize': (7, 5.2)}  # (3.5, 2.5) default
rcParams.update(plot_params)

with PdfPages('L_even_no_scale.pdf') as pdf:
    for i in range(0, 11):
        stri = str(i)
        IO.go_to_data_parent('interpolatedboson/a'+stri)
        IO.add_to_path()
        state = IO.get_state()
        Ls = range(4, 11, 2)
        L_min, L_max = min(Ls), max(Ls)
        shift_func = lambda L: 0.5*(L-L_max-1)/(L_max - L_min)
        #shift_func = lambda L: 0

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

        # shifts = {}
        # for L in Ls:
            # K0N0 = filter(lambda d:d['K']==0 and d['N']==0 and d['L']==L, all_points)[0]
            # shifts[L] = K0N0[data_name]
        # for point in all_points:
            # point['shifted_E'] = point[data_name] - shifts[point['L']]
        # data_name = 'shifted_E'

        # scales = {}
        # for L in Ls:
            # if L%2:  # L is odd
                # K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
                # K1N1 = filter(lambda d:d['K']==1 and d['N']==1 and d['L']==L, all_points)[0]
                # scales[L] = K1N1[data_name] - K0N1[data_name]
            # else:  # L is even
                # K0N0 = filter(lambda d:d['K']==0 and d['N']==0 and d['L']==L, all_points)[0]
                # K1N0 = filter(lambda d:d['K']==1 and d['N']==0 and d['L']==L, all_points)[0]
                # scales[L] = K1N0[data_name] - K0N0[data_name]

        # for point in all_points:
            # point['scaled_E'] = point[data_name]/scales[point['L']]
        # data_name = 'scaled_E'

        # chemical_potentials = {}
        # for L in Ls:
            # K0N1 = filter(lambda d:d['K']==0 and d['N']==1 and d['L']==L, all_points)[0]
            # chemical_potentials[L] = 1 - K0N1[data_name]

        # for point in all_points:
            # point['chemscaled_E'] = point[data_name] + chemical_potentials[point['L']]*point['N']
        # data_name = 'chemscaled_E'

        spec_points = spectrum.collect(['K', 'N', 'band'], 'L', data_name)

        spec_points.plot_toolbox = data.SpectraPlotTools()
        spec_points.plot_toolbox.markers = ['1', '2', '3', '4']
        #spec_points.plot_toolbox.markers
        spec_points.update_plot_args(title='Edge Spectrum', xlabel='K', ylabel='Energy',
                                          xlim=[-0.7, 2.5], ylim=[-0.4, 3.7])
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
                #return abs(N)+1+x
                return 11-abs(N)+x
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
        #ax.xaxis.set_ticklabel_direction(which='minor', dir
        #ax.xaxis.minor_ticklabels.set_axis_direction("top")
        
        # box1 = TextArea("L", textprops=dict(color="k"))
        # anchored_box = AnchoredOffsetbox(loc=3,
                                 # child=box1, pad=0.,
                                 # frameon=False,
                                 # bbox_to_anchor=(0.3, -0.1),
                                 # bbox_transform=ax.transAxes,
                                 # borderpad=0.,
                                 # )
        handles, labels = ax.get_legend_handles_labels()
        hl = sorted(zip(handles, labels),
                key= lambda x: abs(int(x[1])))
        handles2, labels2 = zip(*hl)
        def label_helper(string_representing_integer):
            i = int(string_representing_integer)
            if not i%2:
                return str(i//2)
            else:
                return str(i)+'/2'
        labels3 = [label_helper(l) for l in labels2]
        ax.legend(handles2[0:9], labels3[0:9], loc = 4, fontsize=14, numpoints=1)
        plt.hlines([0], -0.65, -0.05, colors='k', linestyles='dotted')
        # plt.hlines([2], 1-0.45, 1-0.05, colors='k', linestyles='dotted')
        plt.tight_layout()
        pdf.savefig()
        #IO.mkdirs('plots')
        #plt.savefig(IO.os.getcwd()+'\\plots\\iso_es2')
        plt.close()

