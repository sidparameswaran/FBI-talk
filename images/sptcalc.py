# coding: utf-8
from PEPS import mps_tools
from __future__ import division
import math
import numpy as np
from PEPS import IO_helper as IO
IO.go_to_data_parent('softcoreboson')
IO.add_to_path()
state = IO.get_state()
mps = mps_tools
mps.example_SPT_calc()
L=3
p=2
edge_d=2

#lcst = get_left_canonical_site_tensor(statename='hardcoreboson', L=L)
gcst, Lambda, V_basis_change = mps.get_canonical_site_tensor(statename='hardcoreboson', L=L)
p_dim, bond_dim, _ = gcst.shape
reload(mps)
L=3
p=2
edge_d=2

#lcst = get_left_canonical_site_tensor(statename='hardcoreboson', L=L)
gcst, Lambda, V_basis_change = mps.get_canonical_site_tensor(statename='hardcoreboson', L=L)
p_dim, bond_dim, _ = gcst.shape
#print rhoL
XHC = np.array([[0, 1], [1, 0]])
ZHC = np.array([[1, 0],[0,-1]])
ZSC = np.array([[1, 0, 0, 0],[0,-1, 0, 0],[0,0, 1, 0],[0,0, 0, -1]])
Id = np.identity(p_dim)
UX = reduce(np.kron, [XHC]*(2*L))
UZ = reduce(np.kron, [ZHC]*(2*L))
from scipy.linalg import expm
def U(theta):
    return expm(1j*theta*UZ)
U(0)
np.shape(UZ)
U(np.pi)
np.diag(U(np.pi))
UZ
np.diag(UZ)
UID = reduce(np.kron, [Id]*(2*L))
np.shape(Id)
Q = (UZ+Id)/2
np.diag(Q)
Q = (Id-UZ)/2
np.diag(Q)
from SimplePEPS.sym.boson_charge import charge
Q = np.diag([charge(x, 6) for x in xrange(64)])
np.diag(Q)
from scipy.linalg import expm
def U(theta):
    return expm(1j*theta*Q)
np.diag(U(np.pi))
V = mps.determine_V(gcst, Lambda, U(np.pi))
lcst = dot(Lambda, gcst)
lcst = np.dot(Lambda, gcst)
test = np.tensordot(lcst.conj(), lcst, axes=[[0, 1], [0, 1]])
test
np.diag(test)
np.trace(Lambda**2)
gcst = gcst/np.sqrt(np.test[0, 0])
gcst = gcst/np.sqrt(test[0, 0])
lcst = np.dot(Lambda, gcst)
test = np.tensordot(lcst.conj(), lcst, axes=[[0, 1], [0, 1]])
np.diag(test)
rcst = np.dot(gcst, Lambda)
test = np.tensordot(rcst.conj(), rcst, axes=[[0, 2], [0, 2]])
np.diag(test)
V = mps.determine_V(gcst, Lambda, U(np.pi))
V = mps.determine_V(gcst, Lambda, U(np.pi/10))
reload(mps)
edge_d
L
V_posbasis = dot(V_basis_change, V, V_basis_change.conj().transpose())
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, U(0))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.transpose())
mps.print_mat(V_posbasis, edge_d, L)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, U(math.pi))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
VI = mps.determine_VI(gcst, Lambda, U(math.pi))
permutation = list(reversed(range(2*L)))
#print permutation
UIY = mps.ppermute(permutation, p=p)
np.shape(UIY)
VI = mps.determine_VI(gcst, Lambda, np.dot(UIY, U(math.pi)))
VI = mps.determine_VI(gcst, Lambda, np.dot(UIY, Id))
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.conj().transpose())
mps.print_mat(VI_posbasis, edge_d, L)
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.transpose())
mps.print_mat(VI_posbasis, edge_d, L)
VI = mps.determine_VI(gcst, Lambda, np.dot(UIY, U(np.pi)))
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.transpose())
mps.print_mat(VI_posbasis, edge_d, L)
L=1
gcst, Lambda, V_basis_change = mps.get_canonical_site_tensor(statename='hardcoreboson', L=L)
Lambda
V_basis_change
gcst
XHC = np.array([[0, 1], [1, 0]])
ZHC = np.array([[1, 0],[0,-1]])
ZSC = np.array([[1, 0, 0, 0],[0,-1, 0, 0],[0,0, 1, 0],[0,0, 0, -1]])
Id = np.identity(p_dim)
UX = reduce(np.kron, [XHC]*(2*L))
UZ = reduce(np.kron, [ZHC]*(2*L))
Q = np.diag([charge(x, 2*L) for x in xrange(2**(2*L))])
from scipy.linalg import expm
def U(theta):
    return expm(1j*theta*Q)
V = mps.determine_V(gcst, Lambda, U(math.pi))
permutation = list(reversed(range(2*L)))
#print permutation
UIY = mps.ppermute(permutation, p=p)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, U(math.pi)))
VI = mps.determine_VI(gcst, Lambda, np.dot(UIY, U(math.pi)))
VI = mps.determine_VI(gcst, Lambda, UIY)
VI = mps.determine_VI(gcst, Lambda, U(math.pi))
VI = mps.determine_VI(gcst, Lambda, U(math.pi/10))
VI = mps.determine_VI(gcst, Lambda, U(math.pi*2/10))
VI = mps.determine_VI(gcst, Lambda, U(math.pi))
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.transpose())
mps.print_mat(VI_posbasis, edge_d, L)
gcst = gcst/np.sqrt(10)
VI = mps.determine_VI(gcst, Lambda, U(0))
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.transpose())
mps.print_mat(VI_posbasis, edge_d, L)
VI = mps.determine_VI(gcst, Lambda, UIY)
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.transpose())
mps.print_mat(VI_posbasis, edge_d, L)
VI = mps.determine_VI(gcst, Lambda, UIY)
VI_posbasis = mps.dotall(V_basis_change, VI, V_basis_change.conj().transpose())
mps.print_mat(VI_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, UIY)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, U(0))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, U(np.pi/10))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, U(np.pi/10))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, U(np.pi/10))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, UX)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, UZ)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
UX
UZ
UZ
V = mps.determine_V(gcst, Lambda, UX)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, UZ)
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(UX, UZ))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
VZ = mps.determine_V(gcst, Lambda, UZ)
VZ_posbasis = mps.dotall(V_basis_change, VZ, V_basis_change.conj().transpose())
mps.print_mat(VZ_posbasis, edge_d, L)
VX = mps.determine_V(gcst, Lambda, UX)
VX_posbasis = mps.dotall(V_basis_change, VX, V_basis_change.conj().transpose())
mps.print_mat(VX_posbasis, edge_d, L)
VXZ = mps.determine_V(gcst, Lambda, np.dot(UX, UZ))
VXZ_posbasis = mps.dotall(V_basis_change, VXZ, V_basis_change.conj().transpose())
mps.print_mat(VXZ_posbasis, edge_d, L)
np.dot(VX, VZ)
VXZ
VZX = mps.determine_V(gcst, Lambda, np.dot(UZ, UX))
VZX_posbasis = mps.dotall(V_basis_change, VZX, V_basis_change.conj().transpose())
mps.print_mat(VZX_posbasis, edge_d, L)
VZX
np.dot(UX, UZ) - np.dot(UZ, UX)
UZ
UX
np.dot(VZ, VX) - np.dot(VX, VZ)
L=3
XHC = np.array([[0, 1], [1, 0]])
ZHC = np.array([[1, 0],[0,-1]])
ZSC = np.array([[1, 0, 0, 0],[0,-1, 0, 0],[0,0, 1, 0],[0,0, 0, -1]])
Id = np.identity(p_dim)
UX = reduce(np.kron, [XHC]*(2*L))
UZ = reduce(np.kron, [ZHC]*(2*L))
gcst, Lambda, V_basis_change = mps.get_canonical_site_tensor(statename='hardcoreboson', L=L)
Q = np.diag([charge(x, 2*L) for x in xrange(2**(2*L))])
from scipy.linalg import expm
def U(theta):
    return expm(1j*theta*Q)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
UIY.shape
permutation = list(reversed(range(2*L)))
#print permutation
UIY = mps.ppermute(permutation, p=p)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(U(0), U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
gcst/np.sqrt(306.41224316)
gcst = gcst/np.sqrt(306.41224316)
V = mps.determine_V(gcst, Lambda, np.dot(U(0), U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
permutation = range(1, 2*L)+[0]
#print permutation
UIY = mps.ppermute(permutation, p=p)
permutation = list(reversed(range(2*L)))
#print permutation
UIY = mps.ppermute(permutation, p=p)
permutation = range(1, 2*L)+[0]
#print permutation
UT1 = mps.ppermute(permutation, p=p)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, UT1)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, UT1))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, U(0))))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(UIY, U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
permutation = range(2, 2*L)+[0, 1]
#print permutation
UT2 = mps.ppermute(permutation, p=p)
V = mps.determine_V(gcst, Lambda, np.dot(UT2, U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.conj().transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(UT2, U(0)))
V_posbasis = mps.dotall(V_basis_change, V, V_basis_change.transpose())
mps.print_mat(V_posbasis, edge_d, L)
V = mps.determine_V(gcst, Lambda, np.dot(UT2, U(0)))
V_posbasis = mps.dotall(V_basis_change.conj().transpose(), V, V_basis_change)
mps.print_mat(V_posbasis, edge_d, L)
permutation = [4, 5, 2, 3, 0, 1]
#print permutation
UIY1 = mps.ppermute(permutation, p=p)
V = mps.determine_V(gcst, Lambda, np.dot(UIY1, U(0)))
V_posbasis = mps.dotall(V_basis_change.conj().transpose(), V, V_basis_change)
mps.print_mat(V_posbasis, edge_d, L)
permutation = [4, 3, 2, 1, 0, 5]
#print permutation
UIY1 = mps.ppermute(permutation, p=p)
permutation = [4, 3, 2, 1, 0, 5]
#print permutation
UIY2 = mps.ppermute(permutation, p=p)
V = mps.determine_V(gcst, Lambda, np.dot(UIY2, U(0)))
V_posbasis = mps.dotall(V_basis_change.conj().transpose(), V, V_basis_change)
mps.print_mat(V_posbasis, edge_d, L)
mps.print_mat(UIY2, p, L)
mps.print_mat(UIY2, p, 2*L)
