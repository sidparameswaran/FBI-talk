figsize = (3.5,2.5)

from matplotlib import rc, rcParams
params = {'axes.labelsize': 8,
          'text.fontsize': 8,
          'title.fontsize': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'font.family' : 'serif',
          'figure.figsize': (3.5, 2.5),
          'text.latex.preamble' : [r'\usepackage{amsmath}']}
rcParams.update(params)
