import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter as pc


try:
	import data
	import display
except:
	import pkg.data as data
	import pkg.display as display



def load_basis(name):
	file = data.get_basis_file(name)
	with open(file, 'r') as f:
		j = json.load(f)

	return j


def double_factorial(n):
	res = 1
	while n > 1: res *= n; n -= 2
	return res


def evaluate(b, z, p, n, l, ml, c=(0,0,0)):
	'''
	p - nx3 matrix of points to evaluate
	'''
	def orb(prefactor):
		res = np.zeros(r.size)
		for c, e in zip(cs, es):
			res += prefactor * c * math.sqrt(math.pi/e) * np.exp(-e*r**2)
		return res

	X,Y,Z = p[:,0], p[:,1], p[:,2]
	try:
		r = np.sqrt((X-c[0])**2 + (Y-c[1])**2 + (Z-c[2])**2)
	except:
		r = np.linalg.norm(p-c)
	phi = np.arctan2(Y,X)
	theta = np.arctan2(Z,r)

	ce = b['elements'][str(z)]['electron_shells'][n-1]

	cs = ce['coefficients'][l]
	es = ce['exponents']

	cs = [float(c) for c in cs]
	es = [float(e) for e in es]

	if l == 0:
		res = orb(1)
	elif l == 1:
		if ml == -1:
			res = orb(X)
		if ml == 0:
			res = orb(Y)
		if ml == 1:
			res = orb(Z)

	return res


def get_primitive(l, ml, alpha):
	#s-orbital
	if l == 0:
		i = 0; j = 0; k = 0

	#p-orbital
	elif l == 1:
		#px
		if ml == -1:
			i = 1; j = 0; k = 0
		#py
		if ml ==  0:
			i = 0; j = 1; k = 0
		#pz
		if ml ==  1:
			i = 0; j = 0; k = 1

	#d-orbital
	elif l == 2:
		#dyz
		if ml == -2:
			i = 0; j = 1; k = 1
		#dxz
		if ml == -1:
			i = 1; j = 0; k = 1
		#dz2
		if ml ==  0:
			i = 0; j = 1; k = 2
		#dxy
		if ml ==  1:
			i = 1; j = 1; k = 0
		#d(x2-y2)
		if ml ==  2:
			i = 2; j = 0; k = -2

	N = math.sqrt( (4*alpha)**(2*(i+j+k)) / (double_factorial(2*i-1) * double_factorial(2*j-1) * double_factorial(2*k-1)) * math.sqrt(2*alpha/math.pi) )
	return lambda X, Y, Z: N * X**i * Y**j * Z**k * np.exp(-alpha*(X**2 + Y**2 + Z**2))


def contracted_basis(basis, atom_number, n, l, ml):
	params_n = b['elements'][str(atom_number)]['electron_shells'][n-1]

	coefficients = params_n['coefficients'][l]
	coefficients = [float(c) for c in coefficients]

	exponents = params_n['exponents']
	exponents = [float(e) for e in exponents]

	#gather all primitives for atom
	primitives = []
	for alpha in exponents:
		primitives.append(get_primitive(l, ml, alpha))
	
	#return sum over primitives multiplied with coefficients
	return lambda X, Y, Z: sum( c*p(X,Y,Z) for c, p in zip(coefficients, primitives))


b = load_basis('STO-4G')

C_2s  = contracted_basis(b, 8, 2, 0,  0)
C_2px = contracted_basis(b, 8, 2, 1, -1)
C_2py = contracted_basis(b, 8, 2, 1,  0)
C_2pz = contracted_basis(b, 8, 2, 1,  1)



dims = (600,600,1)
dims_prod = dims[0]*dims[1]*dims[2]
ranges = ((-5,5),(-5,5),(0,0))
X,Y = np.meshgrid(np.linspace(*ranges[0],dims[0]), np.linspace(*ranges[1],dims[1]))
Z = np.ones(dims_prod) * 5


X = X.reshape(dims_prod,1)
Y = Y.reshape(dims_prod,1)
Z = Z.reshape(dims_prod,1)



psi_C_2s  = C_2s(X,Y,Z)
psi_C_2px = C_2px(X,Y,Z)
psi_C_2py = C_2py(X,Y,Z)
psi_C_2pz = C_2pz(X,Y,Z)

# psi = psi.reshape((dims[0], dims[1]))



plt.subplot(2,2,1)
psi = psi_C_2s
psi = psi.reshape((dims[0], dims[1]))
plt.imshow(psi)

plt.subplot(2,2,2)
psi = psi_C_2px
psi = psi.reshape((dims[0], dims[1]))
plt.imshow(psi)

plt.subplot(2,2,3)
psi = psi_C_2py
psi = psi.reshape((dims[0], dims[1]))
plt.imshow(psi)

plt.subplot(2,2,4)
psi = psi_C_2pz
psi = psi.reshape((dims[0], dims[1]))
plt.imshow(psi)

plt.show()


