import pkg.data as data
import pkg.integral as integral
import pkg.molecule as molecule
import periodictable as pt
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from scipy.special import factorial, factorial2, comb



def set_basis_set(name='STO-4G', extension='json'):
	file = data.get_basis_file(name, extension)
	with open(file, 'r') as f:
		params = json.load(f)

	global BASIS_PARAMS
	BASIS_PARAMS = params


def load_molecule(mol):
	aos = {}
	for a in mol.atoms:
		aos[a] = (AtomicOrbitals(a.atom_number, a.position, label=a.label))

	return aos


def get_ao_name(ao):
	st = str(pt.elements[ao.atom_number]) + '[' + str(ao.label) + ']('
	st += str(ao.n)
	#s-orbital
	if ao.l == 0: return st + 's)'
	#p-orbital
	elif ao.l == 1:
		if ao.ml == -1: return st + 'px)' #px
		if ao.ml ==  0: return st + 'py)' #py
		if ao.ml ==  1: return st + 'pz)' #pz
	#d-orbital
	elif ao.l == 2:
		if ao.ml == -2: return st + 'dyz)' #dyz
		if ao.ml == -1: return st + 'dxz)' #dxz
		if ao.ml ==  0: return st + 'dz2)' #dz2
		if ao.ml ==  1: return st + 'dxy)' #dxy
		if ao.ml ==  2: return st + 'd(x2-y2))' #d(x2-y2)


class Primitive:
	def __init__(self, exponent, center, angular):
		self.exponent = exponent
		self.center = np.asarray(center)
		self.angular = angular


	def __call__(self, x, y, z):
		mx, my, mz = self.center
		r2 = (x-mx)**2 + (y-my)**2 + (z-mz)**2
		return np.exp(-self.exponent * r2)



class ContractedBasis:
	def __init__(self, pre_exponents, exponents, center, n, l, ml, pre_factor=1, atom_number=0, label=''):
		self.pre_exponents = pre_exponents
		self.exponents = exponents
		self.center = center
		self.n = n
		self.l = l 
		self.ml = ml 
		self.angular = self.get_angular_parts(l, ml)
		self.pre_factor = pre_factor

		self.primitives = []
		for e in exponents:
			self.primitives.append(Primitive(e, center, self.angular))	

		self.atom_number = atom_number
		self.label = label
		self.name = get_ao_name(self)
		

	def __repr__(self):
		return self.name


	def __call__(self, x, y, z):
		try:
			s = np.zeros(len(x),1)
		except:
			s = 0

		i, j, k = self.angular
		for w, p in zip(self.pre_exponents, self.primitives):
			s = s + w * p(x, y, z)

		return self.pre_factor * x**i * y**j * z**k * s


	def get_angular_parts(self, l, ml):
		#s-orbital
		if l == 0: return (0,0,0)
		#p-orbital
		elif l == 1:
			if ml == -1: return (1,0,0) #px
			if ml ==  0: return (0,1,0) #py
			if ml ==  1: return (0,0,1) #pz
		#d-orbital
		elif l == 2:
			if ml == -2: return (0,1,1) #dyz
			if ml == -1: return (1,0,1) #dxz
			if ml ==  0: return (0,0,2) #dz2
			if ml ==  1: return (1,1,0) #dxy
			if ml ==  2: return (2,0,-2) #d(x2-y2)



class AtomicOrbitals:
	def __init__(self, atom_number, center, label=''):
		self.atom_number = atom_number
		self.center = center
		self.label = label
		self.orbital_dict = {}
		self.orbital_list = []

		self.load_params()

	def load_params(self):
		self.basis_parameters = BASIS_PARAMS['elements'][str(self.atom_number)]['electron_shells']
		self.max_n = len(self.basis_parameters)


		for n in range(1, self.max_n+1):
			self.orbital_dict[n] = {}
			angular_momenta = self.basis_parameters[n-1]['angular_momentum']
			for l in angular_momenta:
				self.orbital_dict[n][l] = {}
				for ml in range(-l, l+1):
					exponents = self.basis_parameters[n-1]['exponents']
					exponents = [float(e) for e in exponents]

					# print(atom_number, self.basis_parameters[n-1]['coefficients'], len(self.basis_parameters[n-1]['coefficients']), l)
					coefficients = self.basis_parameters[n-1]['coefficients'][l]
					coefficients = [float(c) for c in coefficients]

					# norm_consts = [math.sqrt(e/math.pi)**3 for e in exponents]
					b = ContractedBasis(coefficients, exponents, self.center, n, l, ml, atom_number=self.atom_number, label=self.label)
					b.pre_factor = math.sqrt(1/integral.overlap(b, b))
					self.orbital_dict[n][l][ml] = b
					self.orbital_list.append(b)



	def __getitem__(self, val):
		if type(val) is tuple:
			return self.orbital_dict[val[0]][val[1]][val[2]]
		else:
			return self.orbital_list[val]



set_basis_set()
