from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from DataAnalysis import PEPS_plots as pp
from PEPS import IO_helper as IO
from PEPS.CylinderPEPS import run
import sfig

Ls = range(1,7+1)

IO.go_to_data_parent('softcoreboson', parent='Data//FBI-TM')
IO.add_to_path()
state = IO.get_state()

ev_to_corr = lambda e: -1/np.log(np.abs(e))
phase = lambda e: 'e^(i pi {})'.format(IO.round_sig(np.angle(e)/np.pi, 2))

scb_ans = {}
for L in Ls:
    tmspec = run.load_transfer_matrix_spectrum(state, L)
    norm = tmspec[(0, 0)][0]
    symev0 = tmspec[(0, 0)][1]/norm
    ev1 = tmspec[(0, 1)][0]/norm

    scb_ans[L] = ev_to_corr(ev1), phase(ev1), ev_to_corr(symev0), phase(symev0)

IO.go_to_data_parent('hardcoreboson', parent='Data//FBI-TM')
IO.add_to_path()
state = IO.get_state()

hcb_ans = {}
for L in Ls:
    tmspec = run.load_transfer_matrix_spectrum(state, L)
    norm = tmspec[(0, 0)][0]
    symev0 = tmspec[(0, 0)][1]/norm
    ev1 = tmspec[(0, 1)][0]/norm
    hcb_ans[L] = ev_to_corr(ev1), phase(ev1), ev_to_corr(symev0), phase(symev0)

from tabulate import tabulate
data = [(L,)+scb_ans[L]+hcb_ans[L] for L in Ls]
print tabulate(data, ['L', 'SCB O', '', 'SCB S', '', 'HCB O', '', 'HCB S', ''])