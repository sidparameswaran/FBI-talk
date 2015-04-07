from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from SimplePEPS.tools import hexplot
import sfig


fig, (ax1, ax2) = plt.subplots(1, 2)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0)
                
data = hexplot.test_corr_data(xi=0.3, radius=3.4)
p = hexplot.hex_circle_plot(data, hexlattice_radius=3.4, cmap=hexplot.cmap_seq, ax=ax1, colorbar=False)
cbar1 = plt.colorbar(p, cax=cax1, ticks=MultipleLocator(0.05), format="%.2f")

data2 = hexplot.test_corr_data(xi=0.5+1.2j, c0=0.25, radius=3.4)
p2 = hexplot.hex_circle_plot(data2, hexlattice_radius=3.4, cmap=hexplot.cmap_div, ax=ax2, scale=2, colorbar=False)
cbar2 = plt.colorbar(p2, cax=cax2, ticks=MultipleLocator(0.05), format="%.2f")

plt.tight_layout()
plt.subplots_adjust(left=-0.03, bottom=0, right=0.9, top=1.4,
                wspace=0.15, hspace=0)
plt.show()
