import numpy as np
from matplotlib import pyplot as plt
from DataAnalysis import PEPS_plots as pp
from PEPS import spectra_analyzer as spec
from PEPS.CylinderPEPS import run
from PEPS import analysis
from PEPS import IO_helper as IO
import sfig

i=10
stri = str(i)
IO.go_to_data_parent('interpolatedboson/a'+stri)
IO.add_to_path()
state = IO.get_state()

Ls = range(2, 11)

params = IO.Params.kw('LKN', L=Ls, K=[0], N=[0])
for p in params:
    run.create_edge_spectrum(state, p, overwrite=False)
    
def entanglement_entropy(es):
    return sum(-v * np.log2(v) for v in es if not np.isclose(v, 0))

entropy = []
for p in params:
    spec = run.load_edge_spectrum(state, p, evecs=False)
    es = [spec_pnt[1] for spec_pnt in spec]
    entropy.append(entanglement_entropy(es))
    
entropy = np.real(entropy)
Ls = np.array(Ls)
extrap_Ls = np.array([0, 1]+list(Ls))

fit_pts, params, R = analysis.curve_fit_main(entropy, analysis.linear, Ls, extrap_Ls)
# flip sign of gamma
params = [params[0], -1*params[1]]

print params
print fit_pts[0], fit_pts[1]

plt.plot(Ls, entropy, 'k.', label='Data')
plt.plot(extrap_Ls, fit_pts, 'b-', label='Fit')

text = text = plt.text(1, 5, 'S = \\alpha L-\gamma')
pl2_txt = lambda params: '\\alpha='+str(IO.round_sig(params[0],2))+'\n\gamma='+str(IO.round_sig(params[1], 2))
text2 = plt.text(1, 3.5, pl2_txt(params))

plt.xlim(xmin=-1, xmax=11)
plt.ylim(ymin=-1)
plt.axhline(0, color='k', ls=':')
plt.axvline(0, color='k', ls=':')

#plt.title('Computing Topological Entanglement Entropy')
plt.xlabel('L')
plt.ylabel('Entanglement Entropy')
plt.legend(loc='best', numpoints=1, frameon=False)
plt.tight_layout()
plt.show()
    


#spectrum = pp.load_edge_spectrum(state, Ls)