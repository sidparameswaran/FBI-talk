from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from DataAnalysis import PEPS_plots as pp
from PEPS import IO_helper as IO
from PEPS.CylinderPEPS import run
import sfig

ev_to_corr = lambda e: -1/np.log(np.abs(e))
phase = lambda e: 'e^(i pi {})'.format(IO.round_sig(np.angle(e)/np.pi, 2))

def corrbound_data(state, Ls):
    ans = {}
    for L in Ls:
        tmspec = run.load_transfer_matrix_spectrum(state, L)
        if tmspec:
            norm = tmspec[(0, 0)][0]
            ev1 = tmspec[(0, 1)][0]/norm
            if L>1:
                symev00 = tmspec[(0, 0)][1]/norm
                symev10 = tmspec[(1, 0)][0]/norm #K=+-1, N=0 ev is bigger than K=0, N=0 sometimes
                print abs(symev00), abs(symev10)
                symev = symev00 if abs(symev00)>abs(symev10) else symev10
                #symev = symev00
            else:
                symev00 = tmspec[(0, 0)][1]/norm
                symev = symev00
            ans[L] = ev_to_corr(ev1), phase(ev1), ev_to_corr(symev), phase(symev)
    return ans



IO.go_to_data_parent('softcoreboson', parent='Data//FBI-TM')
IO.add_to_path()
scb = IO.get_state()
maxL = 10
scb_ans = corrbound_data(scb, range(1,maxL+1))


IO.go_to_data_parent('hardcoreboson', parent='Data//FBI-TM')
IO.add_to_path()
hcb = IO.get_state()
maxL = 10
hcb_ans = corrbound_data(hcb, range(1, maxL+1))


data = [(L,)+scb_ans.get(L, (None, None, None, None))+hcb_ans.get(L,(None, None, None, None)) for L in scb_ans]
from tabulate import tabulate
#print tabulate(data, ['L', 'SCB O', '', 'SCB S', '', 'HCB O', '', 'HCB S', ''])

from SimplePEPS.tools.plot import subplots
(fig, axs), bigax = subplots(2, 2, sharex='all', sharey='row')

plt.sca(axs[0][0])
plt.plot([d[0] for d in data], [d[1] for d in data], ls='', marker='.')
plt.sca(axs[1][0])
plt.plot([d[0] for d in data], [d[3] for d in data], ls='', marker='.')
plt.sca(axs[0][1])
plt.plot([d[0] for d in data], [d[5] for d in data], ls='', marker='.')
plt.sca(axs[1][1])
plt.plot([d[0] for d in data], [d[7] for d in data], ls='', marker='.')

# import PEPS.analysis as a
# fit_func = a.expic
# min_indx = 4
# fit_Ls = ([d[0] for d in data])[min_indx:]
# extrapolate_pnts = np.arange(1, 11, 0.05)
#
# fit_data = ([d[1] for d in data])[min_indx:]
# (curve_data, params, R) = a.curve_fit_main(fit_data, fit_func, fit_Ls, extrapolate_pnts)
# print params
#
# fit_data = ([d[3] for d in data])[min_indx:]
# (curve_data, params, R) = a.curve_fit_main(fit_data, fit_func, fit_Ls, extrapolate_pnts)
# print params
#
# fit_data = ([d[5] for d in data])[min_indx:]
# (curve_data, params, R) = a.curve_fit_main(fit_data, fit_func, fit_Ls, extrapolate_pnts)
# print params
#
# fit_data = ([d[7] for d in data])[min_indx:]
# (curve_data, params, R) = a.curve_fit_main(fit_data, fit_func, fit_Ls, extrapolate_pnts)
# print params

bigax.set_ylabel('Correlation Length', labelpad=12)
bigax.set_xlabel('W')
axs[0][0].set_ylabel('Overall')
axs[0][0].set_title('Soft-Core')
axs[0][1].set_title('Hard-Core')
axs[1][0].set_ylabel('Symmetric')
plt.tight_layout()
plt.show()