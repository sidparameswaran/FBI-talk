from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from SimplePEPS.tools import hexplot
from SimplePEPS.tools.plot import shiftedColorMap


from PEPS import IO_helper as IO
from PEPS.CylinderPEPS import run
from PEPS.CylinderPEPS import plot

import sfig
fig, (ax1, ax2) = plt.subplots(1, 2)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0)

L = 5
#radius1 = 3
#radius2 = 3
#hexradius1 = 4
#hexradius2 = 4
#L=8
radius1 = 5.0
radius2 = 3.61
hexradius1 = 4.2
hexradius2 = 3.5


IO.go_to_data_parent('hardcoreboson', parent='Data//FBI-TM')
params = IO.Params.kw('LKN', L=[L], K=[0], N=[0])
param = next(iter(params))
corrmap, corrmap_string = plot.load_combined_corrmap(param, ('bdag', 'b'))

hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, L, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=radius1) if p.sub!='C'])
hexcorrmap = {k:np.real(v).item() for k,v in hexcorrmap.items()}
p = hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=hexradius1, cmap=hexplot.cmap_seq, ax=ax1, scale=1, colorbar=False)
cbar1 = plt.colorbar(p, cax=cax1, ticks=MultipleLocator(0.05), format="%.2f")

corrmap, corrmap_string = plot.load_combined_corrmap(param, ('n', 'n'))


hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, L, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=radius2) if p.sub!='C'])
hexcorrmap = {k:np.real(v-0.25).item() for k,v in hexcorrmap.items() if v is not None}
vmin, vmax = min(hexcorrmap.values()), max(hexcorrmap.values())
cmap = shiftedColorMap(hexplot.cmap_div,  midpoint=1 - vmax/(vmax + abs(vmin)))
p2 = hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=hexradius2, cmap=cmap, scale=1, ax=ax2, colorbar=False)
cbar2 = plt.colorbar(p2, cax=cax2, ticks=MultipleLocator(0.05), format="%.2f")

#plt.sca(ax1)
#plt.xlim(xmax=6)
#plt.sca(ax2)
#plt.xlim(xmax=4)
plt.tight_layout()
plt.subplots_adjust(#left=-0.03, bottom=0, right=0.9, top=1.2,
                    wspace=0.15, hspace=0)
plt.show()