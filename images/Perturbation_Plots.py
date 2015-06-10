from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import sfig

import matplotlib.ticker as mtick
def sci_notation(val):
    """The two args are the value and tick position"""
    if val==0:
        return '$0$'
    ancilla, exponent = '{:1.0e}'.format(val).split('e')
    if exponent[0]=='+':
        exponent = exponent.lstrip('+0')
    if exponent[0]=='-':
        exponent = '-'+exponent.lstrip('-0')
    if exponent=='':
        exponent='0'
    return '${} \cdot 10^{{ {} }}$'.format(ancilla, exponent)


mpl.rcParams['axes.color_cycle'] = ['k', 'k', 'b', 'b', 'r', 'r', 'g', 'g']


TEBD_data_location = "I:/Data/FBI/L3/itebd/"

from SimplePEPS.tools import fileio
from SimplePEPS.tools.containers import OrderedDict

b_es = OrderedDict()
for x in [0, 1, 2, 3, 5]:
    results = fileio.undump('{}b24/results{}'.format(TEBD_data_location, x))
    b_es[results['c']] = results['es']

bb_es = OrderedDict()
for x in [0, 1, 2, 3, 4, 5]:
    results = fileio.undump('{}bb24/results{}'.format(TEBD_data_location, x))
    bb_es[results['c']] = results['es']

b_es_diff = OrderedDict()
for c in b_es:
    #print c
    b_es_diff[c] = np.array(b_es[c][:8])-np.array(b_es[0][:8])
bb_es_diff = OrderedDict()
for c in bb_es:
    bb_es_diff[c] = np.array(bb_es[c][:8])-np.array(bb_es[0][:8])

f = lambda x: np.exp(-x)
b_points = [[b_es_diff[c][j] for c in b_es_diff] if x else b_es_diff.keys() for j in range(8) for x in range(2)]
bb_points = [[bb_es_diff[c][j] for c in bb_es_diff] if x else bb_es_diff.keys() for j in range(8) for x in range(2)]

b_ee = OrderedDict()
for c in b_es:
    b_ee[c] = -np.log(b_es[c])
b_ee_diff = OrderedDict()
for c in b_ee:
    b_ee_diff[c] = np.array(b_ee[c][:8])-np.array(b_ee[0][:8])
b_ee_points = [[b_ee_diff[c][j] for c in b_ee_diff] if x else b_ee_diff.keys() for j in range(8) for x in range(2)]
bb_ee = OrderedDict()
for c in bb_es:
    bb_ee[c] = -np.log(bb_es[c])
bb_ee_diff = OrderedDict()
for c in bb_es:
    bb_ee_diff[c] = np.array(bb_ee[c][:8])-np.array(bb_ee[0][:8])
bb_ee_points = [[bb_ee_diff[c][j] for c in bb_ee_diff] if x else bb_ee_diff.keys() for j in range(8) for x in range(2)]

#print b_ee_points
from SimplePEPS.ed import perturb as edp
c_func = lambda x, y, j, num, t: ['k', 'k', 'b', 'b', 'r', 'r', 'g', 'g'][t] if t<8 else 'y'
fig, ax = edp._spec_plot(b_es, yfunc = lambda e:-np.log(e), cfunc=c_func,
              xlabel='Perturbation', ylabel='Entanglement energy',# title='Quasi-1D HFBI Perturbed Entanglement Spectrum',
              ylim=[-1, 30])
formatter = mtick.FuncFormatter(lambda x, y: sci_notation(b_es.keys()[y]))
ax.xaxis.set_major_formatter(formatter)
ax_inset=fig.add_axes([0.25,0.45,0.3,0.3])
ax_inset.loglog(*b_ee_points, marker='.', ls='-')
plt.yscale('symlog', linthreshy=1e-6)
plt.xscale('symlog', linthreshx=1e-5)
plt.xlim(0, 2e-3)
plt.yticks([1, -1])
plt.xticks([])
#plt.xlabel('Symmetry Breaking Perturbation')
#plt.ylabel('Change in entanglement energy')
#plt.title('Splitting of entanglement spectrum in perturbed HFBI')
plt.tight_layout()
plt.show()
plt.close()


from SimplePEPS.ed import perturb as edp
c_func = lambda x, y, j, num, t: ['k', 'k', 'b', 'b', 'r', 'r', 'r', 'r'][t] if t<8 else 'y'
fig, ax = edp._spec_plot(bb_es, yfunc = lambda e:-np.log(e), cfunc=c_func,
              xlabel='Perturbation', ylabel='Entanglement energy',# title='Quasi-1D HFBI Perturbed Entanglement Spectrum',
              ylim=[-1, 30])

formatter = mtick.FuncFormatter(lambda x, y: sci_notation(bb_es.keys()[y]))
ax.xaxis.set_major_formatter(formatter)
ax_inset=fig.add_axes([0.25,0.45,0.3,0.3])
ax_inset.loglog(*bb_ee_points, marker='.', ls='-')
plt.yscale('symlog', linthreshy=1e-5)
plt.xscale('symlog', linthreshx=1e-5)
plt.xlim(0, 2e-3)
plt.yticks([1e-4, -1e-4])
plt.xticks([])
#plt.xlabel('Symmetry Breaking Perturbation')
#plt.ylabel('Change in entanglement energy')
#plt.title('Splitting of entanglement spectrum in perturbed HFBI')
plt.tight_layout()
plt.show()
plt.close()


#
# edp.spec_plot(bb_es, yfunc = lambda e:-np.log2(e), cfunc=c_func,
#               xlabel='Symmetry preserving perturbation', ylabel='Entanglement energy',# title='Quasi-1D HFBI Perturbed Entanglement Spectrum',
#               ylim=[-1, 40])
#
#
#
# plt.loglog(*bb_ee_points, marker='.', ls='-')
# plt.yscale('symlog', linthreshy=1e-4)
# plt.xscale('symlog', linthreshx=1e-5)
# plt.xlim(0, 2e-3)
# plt.xlabel('Symmetry Preserving Perturbation')
# plt.ylabel('Change in entanglement energy')
# plt.title('Splitting of entanglement spectrum in perturbed HFBI')
# plt.ylim(-1e-1, 1e-1)
# plt.tight_layout()
# plt.show()
# plt.close()
