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
# fig, (ax1, ax2) = plt.subplots(1, 2)
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes("right", size="5%", pad=0)
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes("right", size="5%", pad=0)

fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)
divider00 = make_axes_locatable(ax00)
cax00 = divider00.append_axes("right", size="5%", pad=0)
divider01 = make_axes_locatable(ax01)
cax01 = divider01.append_axes("right", size="5%", pad=0)
divider10 = make_axes_locatable(ax10)
cax10 = divider10.append_axes("right", size="5%", pad=0)
divider11 = make_axes_locatable(ax11)
cax11 = divider11.append_axes("right", size="5%", pad=0)


L = 8
radius1 = radius2 = 5.0
hexradius1 = hexradius2 = 4.2
#adius1 = radius2 = 14.0
#hexradius1 = hexradius2 = 13.5


## Left two plots for softcoreboson

IO.go_to_data_parent('softcoreboson', parent='Data//FBI-TM')
params = IO.Params.kw('LKN', L=[L], K=[0], N=[0])
param = next(iter(params))

corrmap, corrmap_string = plot.load_combined_corrmap(param, ('bdag', 'b'))
hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, L, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=radius1) if p.sub!='C'])
hexcorrmap = {k:np.real(v).item() for k,v in hexcorrmap.items()}
p = hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=hexradius1, cmap=hexplot.cmap_seq, ax=ax00, circle_size='radius', scale=1, colorbar=False)
cbar1 = plt.colorbar(p, cax=cax00, ticks=MultipleLocator(0.05), format="%.2f")

corrmap, corrmap_string = plot.load_combined_corrmap(param, ('n', 'n'))
hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, L, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=radius2) if p.sub!='C'])
hexcorrmap = {k:np.real(v-0.25).item() for k,v in hexcorrmap.items() if v is not None}
vmin, vmax = min(hexcorrmap.values()), max(hexcorrmap.values())
cmap = shiftedColorMap(hexplot.cmap_div,  midpoint=1 - vmax/(vmax + abs(vmin)))
p2 = hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=hexradius2, cmap=cmap, scale=1, ax=ax10, colorbar=False)
cbar2 = plt.colorbar(p2, cax=cax10, ticks=MultipleLocator(0.05), format="%.2f")

## Right two plots for hardcoreboson

IO.go_to_data_parent('hardcoreboson', parent='Data//FBI-TM')
params = IO.Params.kw('LKN', L=[L], K=[0], N=[0])
param = next(iter(params))

corrmap, corrmap_string = plot.load_combined_corrmap(param, ('bdag', 'b'))
hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, L, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=radius1) if p.sub!='C'])
hexcorrmap = {k:np.real(v).item() for k,v in hexcorrmap.items()}
p = hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=hexradius1, cmap=hexplot.cmap_seq, ax=ax01, circle_size='radius', scale=1, colorbar=False)
cbar1 = plt.colorbar(p, cax=cax01, ticks=MultipleLocator(0.05), format="%.2f")

corrmap, corrmap_string = plot.load_combined_corrmap(param, ('n', 'n'))
hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, L, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=radius2) if p.sub!='C'])
hexcorrmap = {k:np.real(v-0.25).item() for k,v in hexcorrmap.items() if v is not None}
vmin, vmax = min(hexcorrmap.values()), max(hexcorrmap.values())
cmap = shiftedColorMap(hexplot.cmap_div,  midpoint=1 - vmax/(vmax + abs(vmin)))
p2 = hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=hexradius2, cmap=cmap, scale=1, ax=ax11, colorbar=False)
cbar2 = plt.colorbar(p2, cax=cax11, ticks=MultipleLocator(0.05), format="%.2f")

#plt.sca(ax1)
#plt.xlim(xmax=6)
#plt.sca(ax2)
#plt.xlim(xmax=4)
plt.tight_layout()
plt.subplots_adjust(#left=-0.03, bottom=0, right=0.9, top=1.2,
                    wspace=0.25, hspace=0.1)
plt.show()
