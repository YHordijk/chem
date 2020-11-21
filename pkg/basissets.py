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



class ContractedGaussian:
	def __init__(self, z, n, l, ml, center, params):
		self.z = z
		self.n = n
		self.l = l
		self.ml = ml
		self.center = center
		self.params = params
		self.powers = self.get_powers(l, ml)


	def get_powers(self, l, ml):
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

		return i, j, k


	def primitive(self):
		N = math.sqrt( (4*alpha)**(2*(i+j+k)) / (double_factorial(2*i-1) * double_factorial(2*j-1) * double_factorial(2*k-1)) * math.sqrt(2*alpha/math.pi) )
		return lambda X, Y, Z: N * np.exp(-alpha*((X-c[0])**2 + (Y-c[1])**2 + (Z-c[2])**2))

	
	#return sum over primitives multiplied with coefficients
	

	def __call__(self, X, Y, Z):
		i, j, k = self.powers
		X**i * Y**j * Z**k
		return sum(c*p(X,Y,Z) for c, p in zip(self.params['coefficients'], self.params['primitives']))



class Basis:
	def __init__(self, name='STO-4G'):
		self.name = name
		self.basis_parameters = self.load_basis_parameters(name)
		self.basis_functions = {}


	def load_molecule(self, mol):
		#loop over atoms and add their primitives to the list
		b = {}
		for a in mol.atoms:
			b[a] = self.load_atom_basis(a)

		self.basis_functions = b


	def load_atom_basis(self, a):
		z = a.atom_number
		c = np.asarray(a.position)

		params = self.basis_parameters['elements'][str(z)]['electron_shells']
		max_n = len(params)

		atomb = {}
		for n in range(1, max_n+1):
			atomb[n] = {}
			for l in range(0, n):
				atomb[n][l] = {}
				for ml in range(-l, l+1):
					atomb[n][l][ml] = self.get_contracted_basis(z, c, n, l, ml)

		return atomb


	def load_basis_parameters(self, name):
		file = data.get_basis_file(name)
		with open(file, 'r') as f:
			j = json.load(f)

		return j


	def get_primitive(self, c, powers, alpha):
		i, j, k = powers
		N = math.sqrt( (4*alpha)**(2*(i+j+k)) / (double_factorial(2*i-1) * double_factorial(2*j-1) * double_factorial(2*k-1)) * math.sqrt(2*alpha/math.pi) )
		return lambda X, Y, Z: N * X**i * Y**j * Z**k * np.exp(-alpha*((X-c[0])**2 + (Y-c[1])**2 + (Z-c[2])**2))


	def get_contracted_basis(self, z, c, n, l, ml):
		exponents = self.basis_parameters['elements'][str(z)]['electron_shells'][n-1]['exponents']
		exponents = [float(e) for e in exponents]
		coefficients = self.basis_parameters['elements'][str(z)]['electron_shells'][n-1]['coefficients'][l]
		coefficients = [float(c) for c in coefficients]

		params = {}
		params['exponents'] = exponents
		params['coefficients'] = coefficients

		return ContractedGaussian(z, n, l, ml, c, params)

		#gather all primitives for atom
		




def double_factorial(n):
	res = 1
	while n > 1: res *= n; n -= 2
	return res