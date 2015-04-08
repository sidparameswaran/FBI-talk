from PEPS import IO_helper as IO
from PEPS.CylinderPEPS import run
from PEPS.CylinderPEPS import plot
from matplotlib import pyplot as plt
import numpy as np

IO.go_to_data_parent('softcoreboson')
state = IO.grab_state('softcoreboson')

params = IO.Params.kw('LKN', L=[8], K=[0], N=[0])
p = next(iter(params))
corrmap, corrmap_string = plot.load_corrmap(p, ('bdag', 'b'))

from SimplePEPS.tools import hexplot
hexcorrmap = hexplot.convert_zigzag_dict_to_hex(corrmap, 8, [p for p, dist in hexplot.Hex(-1, 2).disc(radius=5.0) if p.sub!='C'])
hexcorrmap = {k:np.real(v).item() for k,v in hexcorrmap.items()}
hexplot.hex_circle_plot(hexcorrmap, hexlattice_radius=4.2, cmap=hexplot.cmap_seq, scale=1)
plt.show()