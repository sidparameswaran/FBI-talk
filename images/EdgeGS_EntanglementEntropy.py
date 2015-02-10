from matplotlib import pyplot as plt
from PEPS import cft_entanglement_entropy as cft
from PEPS import IO_helper as IO
import sfig

i=10
stri = str(i)
IO.go_to_data_parent('interpolatedboson/a'+stri)
IO.add_to_path()
state = IO.get_state()

params = IO.Params.kw('LKN', L=[10], K=[0], N=[0])

for p in params:
    cft.compare_edge_gs_ee_to_cft(state, p)
    plt.xlabel('Number of sites in cut')
    plt.ylabel('Entanglement entropy of cut')
    plt.tight_layout()
    plt.show()