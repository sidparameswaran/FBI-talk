from matplotlib import rc, rcParams, ticker, matplotlib_fname

params = {'font.size':8,
          'axes.titlesize': 8,
          'axes.labelsize': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'savefig.bbox': 'tight',
          'savefig.pad_inches': 0.1,
          'font.family' : 'serif',
          'figure.figsize': (3.5, 2.5),
          'text.latex.preamble' : [r'\usepackage{amsmath}']
          }

rcParams.update(params)
