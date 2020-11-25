import numpy as np 
import pubchempy as pcp
import os
import periodictable as pt
from math import sin, cos, log, sqrt, exp
import math
import networkx as nx
import itertools
import scipy.optimize as sciopt

try:
	import data
except:
	import pkg.data as data




def minimize_molecule(mol, force_field='UFF', use_scipy=False):
	mollist = []
	def callback(xk):
		# mol.center()
		mollist.append(mol.copy())
		mollist[-1].set_coordinates(xk.reshape((xk.size//3,3)))

		return False

	if force_field == 'UFF':
		ff = UFF(unit='hartree')

	x0 = mol.get_coordinates()
	x0 = x0.reshape((x0.size))
	params = ff.prepare_parameters(mol)

	if use_scipy:
		sciopt.minimize(ff._get_energy, x0, jac=ff._get_energy_gradient, 
			args=(mol, params, False), options={'maxiter':20000, 'disp':True}, 
			callback=callback, method='Newton-CG')

	else:
		maxiter = 2000
		maxcartstep = 0.1
		update_strength = .2
		u = update_strength
		energies = [float('inf')]
		thresh = 1e-8
		x = x0
		for i in range(maxiter):
			grad = ff._get_energy_gradient(x, mol, params)
			grad = np.maximum(grad, -maxcartstep/update_strength)
			grad = np.minimum(grad,  maxcartstep/update_strength)

			x = x - grad * u

			callback(x)
			energies.append(ff.get_energy(mollist[-1], morse=False))

			energy_diff = abs(energies[-1] - energies[-2])


			if energy_diff < thresh:
				print(f'Converged after {i} steps with energy {energies[-1]}.')
				break


	return mollist





class IMUFF:
	def __init__(self, *args, **kwargs):
		self.parameters = data.FF_UFF_PARAMETERS
		self.optimal_bond_lengths = {}




class UFF:
	'''
	RESOURCES:
	http://towhee.sourceforge.net/forcefields/uff.html
	A. K. Rappe; C. J. Casewit; K. S. Colwell; W. A. Goddard III; W. M. Skiff; "UFF, a
	Full Periodic Table Force Field for Molecular Mechanics and Molecular Dynamics 
	Simulations", J. Am. Chem. Soc. 114 10024-10035 (1992). 
	'''

	def __init__(self, verbose=False, unit='kcal/mol', *args, **kwargs):
		self.verbose = verbose
		self.unit = unit


	def convert_energy(self, E):
		if self.unit == 'kcal/mol':
			return E
		elif self.unit == 'kJ/mol':
			return E * 4.182
		elif self.unit == 'eV':
			return E * 0.04336
		elif self.unit == 'hartree':
			return E * 0.04336 / 27.211386245988

	def get_atom_types(self, mol):
		types = []
		for atom in mol.atoms:
			el = atom.element 
			el += '_'*(len(el)==1)
			if atom.ring == 'AR':
				el += 'R'
			elif atom.hybridisation != 0:
				el += str(atom.hybridisation)

			types.append(el)



	def set_atom_types(self, mol):
		for atom in mol.atoms:
			el = atom.element 
			el += '_'*(len(el)==1)
			if atom.ring == 'AR':
				el += 'R'
			elif atom.hybridisation != 0:
				el += str(atom.hybridisation)

			atom.uff_atom_type = el


	def prepare_parameters(self, mol):
		#set the atom types for mol
		self.set_atom_types(mol)

		#load UFF parameters
		UFF_PARAM = data.FF_UFF_PARAMETERS
		#initialize empty parameters
		PARAM = {'stretch':{}, 'bend':{}, 'torsion':{}, 'vdw':{}, 'el':{}}

		### BOND STRETCH ENERGY
		for a1, a2, order in mol.unique_bonds:
			#get the atom types and sort them
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))

			if a1t == a2t == 'C_R':
				order = 1.5

			#check if already calculated and skip if we did
			if sorted_at in PARAM['stretch'].keys():
				continue
			PARAM['stretch'][sorted_at] = {}


			#calculate params
			rI, rJ = UFF_PARAM['valence_bond'][a1t], UFF_PARAM['valence_bond'][a2t]
			xI, xJ = UFF_PARAM['electro_negativity'][a1t], UFF_PARAM['electro_negativity'][a2t]

			#rIJ = rI + rJ + rBO + rEN
			rBO = -0.1332 * (rI + rJ) * log(order)
			rEN = rI * rJ * (sqrt(xI) - sqrt(xJ))**2 / (xI*rI + xJ*rJ)
			rIJ = rI + rJ + rBO - rEN

			ZI, ZJ = UFF_PARAM['effective_charge'][a1t], UFF_PARAM['effective_charge'][a2t]
			kIJ = 664.12 * ZI * ZJ / rIJ**3

			PARAM['stretch'][sorted_at]['rIJ'] = rIJ
			PARAM['stretch'][sorted_at]['kIJ'] = kIJ
			PARAM['stretch'][sorted_at]['DIJ'] = order * 70

		### BOND ANGLE ENERGY
		for a1, a2, a3 in mol.unique_bond_angles:
			a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t)))

			if sorted_at in PARAM['bend'].keys():
				continue
			PARAM['bend'][sorted_at] = {}

			theta0 = UFF_PARAM['valence_angle'][a2t] * math.pi/180
			costheta0 = cos(theta0)
			sintheta0 = sin(theta0)

			r12 = PARAM['stretch'][tuple(sorted((a1t, a2t)))]['rIJ']
			r12_2 = r12 * r12 #squared
			r23 = PARAM['stretch'][tuple(sorted((a2t, a3t)))]['rIJ']
			r23_2 = r23 * r23
			r1223 = r12 * r23
			r13_2 = r12_2 + r23_2 - 2 * r1223 * cos(theta0)

			Z1, Z3 = UFF_PARAM['effective_charge'][a1t], UFF_PARAM['effective_charge'][a3t]

			beta = 664.12/r1223

			KIJK = beta * (Z1*Z3/r13_2**2.5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13_2**2 * costheta0)

			C2 = 1/(4*sintheta0**2)
			C1 = -4 * C2 * costheta0
			C0 = C2 * (2 * costheta0**2 + 1)

			PARAM['bend'][sorted_at]['theta0'] = theta0
			PARAM['bend'][sorted_at]['Z13'] = Z1 * Z3
			PARAM['bend'][sorted_at]['r1223'] = r1223
			PARAM['bend'][sorted_at]['C0'] = C0
			PARAM['bend'][sorted_at]['C1'] = C1
			PARAM['bend'][sorted_at]['C2'] = C2

		### BOND TORSION ENERGY
		for a1, a2, a3, a4 in mol.unique_torsion_angles:
			a1t, a2t, a3t, a4t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type, a4.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t, a4t)))

			if sorted_at in PARAM['torsion'].keys():
				continue
			PARAM['torsion'][sorted_at] = {}

			Vbarr = 1
			if a2.hybridisation == a3.hybridisation == 3:
				V2, V3 = UFF_PARAM['sp3_torsional_barrier_params'][a2t], UFF_PARAM['sp3_torsional_barrier_params'][a3t]
				Vbarr = sqrt(V2*V3)
				n = 3
				phi0 = math.pi, math.pi/3 #two different phi0 possible

			if (a2.hybridisation == 3 and a3.hybridisation == 2) or (a2.hybridisation == 2 and a3.hybridisation == 3):
				Vbarr = 1
				n = 6
				phi0 = 0, 0

			if a2.hybridisation == a3.hybridisation == 2:
				U2 = UFF_PARAM['sp2_torsional_barrier_params'][a2t] # period starts at 1
				U3 = UFF_PARAM['sp2_torsional_barrier_params'][a3t]
				Vbarr = 5*sqrt(U2*U3)*(1+4.18*log(mol.get_bond_order(a2, a3)))
				n = 2
				phi0 = math.pi, 1.047198

			PARAM['torsion'][sorted_at]['phi0'] = phi0
			PARAM['torsion'][sorted_at]['n'] = n
			PARAM['torsion'][sorted_at]['Vbarr'] = Vbarr

		### VDW ENERGY
		for a1, a2 in mol.unique_pairs_3:
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))

			if sorted_at in PARAM['vdw'].keys():
				continue
			PARAM['vdw'][sorted_at] = {}

			x1, x2 = UFF_PARAM['nonbond_distance'][a1t], UFF_PARAM['nonbond_distance'][a2t]
			x12 = (x1+x2)/2

			D1, D2 = UFF_PARAM['nonbond_energy'][a1t], UFF_PARAM['nonbond_energy'][a2t]
			D12 = sqrt(D1*D2)

			PARAM['vdw'][sorted_at]['D12'] = D12
			PARAM['vdw'][sorted_at]['x12'] = x12

		return PARAM


	@staticmethod
	def _get_energy_gradient(x, mol, params, morse=False):
		x = x.reshape((x.size//3,3))
		#### utility functions:
		def distance(a1, a2):
			return np.linalg.norm(a1.position - a2.position)

		def bond_angle(a1, a2, a3):
			u = a1.position - a2.position
			v = a3.position - a2.position
			return np.arccos((u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v)))

		def torsion_angle(a1, a2, a3, a4):
			b1 = a2.position - a1.position
			b2 = a3.position - a2.position
			b3 = a4.position - a3.position

			return math.atan2(np.dot(np.cross(np.cross(b1, b2), np.cross(b2, b3)), b2/np.linalg.norm(b2)), np.dot(np.cross(b1, b2), np.cross(b2, b3)))
			
		mol.set_coordinates(x)

		grad = np.zeros_like(x)

		### BOND STRETCH ENERGY
		#Er = kIJ/2(r-rIJ)**2
		param = params['stretch']
		for a1, a2, order in mol.unique_bonds:
			#get atome types
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))

			r = distance(a1,a2) #get distance
			r1 = x[a1.index]
			r2 = x[a2.index]

			#get params
			DIJ = param[sorted_at]['DIJ']
			rIJ = param[sorted_at]['rIJ']
			kIJ = param[sorted_at]['kIJ']

			if morse:
				E = DIJ * (exp(-sqrt(kIJ/(2*DIJ)) * (r-rIJ)) - 1)**2
			else:
				g = kIJ * (r-rIJ) * (r1-r2) / rIJ
				grad[a1.index] += g
				grad[a2.index] -= g

		### BOND ANGLE ENERGY
		#Etheta = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
		param = params['bend']
		for a1, a2, a3 in mol.unique_bond_angles:
			#get atom types
			a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t)))

			theta = bond_angle(a1,a2,a3) #get bond angle

			#get params
			theta0 = param[sorted_at]['theta0']
			Z13 = param[sorted_at]['Z13']
			r1223 = param[sorted_at]['r1223']
			r12 = params['stretch'][tuple(sorted((a1t, a2t)))]['rIJ']
			r12_2 = r12 * r12 #squared
			r23 = params['stretch'][tuple(sorted((a2t, a3t)))]['rIJ']
			r23_2 = r23 * r23
			C0 = param[sorted_at]['C0']
			C1 = param[sorted_at]['C1']
			C2 = param[sorted_at]['C2']

			beta = 664.12/r1223
			costheta0 = cos(theta0)
			r13_2 = r12_2 + r23_2 - 2 * r1223 * cos(theta0)
			KIJK = beta * (Z13/r13_2**2.5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13_2**2 * costheta0)

			r1 = x[a1.index]
			r2 = x[a2.index]
			r3 = x[a3.index]

			aa = r1@r1
			bb = r2@r2
			cc = r3@r3 
			ba = r2@r1
			bc = r2@r3
			ac = r1@r3

			u = r1 - r2
			v = r3 - r2

			g = u@v / np.sqrt(u@u * v@v)
			dtdg = -1/np.sqrt(1-g**2)

			dtdr1 = dtdg * (r2 * (-2*bb + 4*bc - 2*cc) + 2*r1 * (bb - 2*bc + cc))
			dtdr2 = dtdg * (r1 * (-2*bb + 4*bc - 2*cc) + r3 * (-2*bb + 4*ba - 2*aa) + r2 * (4*bb - 2*bc + 2*cc - 4*ba + 2*aa))
			dtdr3 = dtdg * (r2 * (-2*bb + 4*ba - 2*aa) + 2*r3 * (bb - 2*ba + aa))

			dEdt = -KIJK * (C1 * sin(theta) + C2 * sin(2*theta))

			grad[a1.index] += dtdr1 * dEdt
			grad[a2.index] += dtdr2 * dEdt
			grad[a3.index] += dtdr3 * dEdt

		grad = grad * 0.04336 / 27.211386245988
		grad = grad.reshape((x.size))

		return grad


	@staticmethod
	def _get_energy(x, mol, params, morse=True):
		x = x.reshape((x.size//3,3))
		#### utility functions:
		def distance(a1, a2):
			return np.linalg.norm(a1.position - a2.position)

		def bond_angle(a1, a2, a3):
			u = a1.position - a2.position
			v = a3.position - a2.position
			return np.arccos((u @ v) / (np.sqrt(u @ u * v @ v)))

		def torsion_angle(a1, a2, a3, a4):
			b1 = a2.position - a1.position
			b2 = a3.position - a2.position
			b3 = a4.position - a3.position

			return math.atan2(np.dot(np.cross(np.cross(b1, b2), np.cross(b2, b3)), b2/np.linalg.norm(b2)), np.dot(np.cross(b1, b2), np.cross(b2, b3)))
			
		mol.set_coordinates(x)
		#### ENERGY CALCULATION
		#E = Er + Etheta + Ephi + Eohm + Evdw + Eel
		Er = 0
		Etheta = 0
		Ephi = 0
		Eohm = 0
		Evdw = 0
		Eel = 0

		### BOND STRETCH ENERGY
		#Er = kIJ/2(r-rIJ)**2
		param = params['stretch']
		for a1, a2, order in mol.unique_bonds:
			#get atome types
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))

			r = distance(a1,a2) #get distance

			#get params
			DIJ = param[sorted_at]['DIJ']
			rIJ = param[sorted_at]['rIJ']
			kIJ = param[sorted_at]['kIJ']

			#calculate energy
			if morse:
				E = DIJ * (exp(-sqrt(kIJ/(2*DIJ)) * (r-rIJ)) - 1)**2
			else:
				E = kIJ/2 * (r - rIJ)**2
			Er += E

		### BOND ANGLE ENERGY
		#Etheta = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
		param = params['bend']
		for a1, a2, a3 in mol.unique_bond_angles:
			#get atom types
			a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t)))

			theta = bond_angle(a1,a2,a3) #get bond angle

			#get params
			theta0 = param[sorted_at]['theta0']
			Z13 = param[sorted_at]['Z13']
			r1223 = param[sorted_at]['r1223']
			r12 = params['stretch'][tuple(sorted((a1t, a2t)))]['rIJ']
			r12_2 = r12 * r12 #squared
			r23 = params['stretch'][tuple(sorted((a2t, a3t)))]['rIJ']
			r23_2 = r23 * r23
			C0 = param[sorted_at]['C0']
			C1 = param[sorted_at]['C1']
			C2 = param[sorted_at]['C2']

			beta = 664.12/r1223
			costheta0 = cos(theta0)
			r13_2 = r12_2 + r23_2 - 2 * r1223 * cos(theta0)
			KIJK = beta * (Z13/r13_2**2.5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13_2**2 * costheta0)

			E = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
			Etheta += E

		### BOND TORSION ENERGY
		#Ephi = Vphi/2 * (1 - cos(n*phi0)*cos(n*phi))
		param = params['torsion']
		for a1, a2, a3, a4 in mol.unique_torsion_angles:
			#get atom types
			a1t, a2t, a3t, a4t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type, a4.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t, a4t)))

			phi = torsion_angle(a1, a2, a3, a4) #get torsion angle

			Vbarr = param[sorted_at]['Vbarr']
			phi0 = param[sorted_at]['phi0']
			n = param[sorted_at]['n']

			E_phi1 = 0.5*Vbarr * (1-cos(n*phi0[0])*cos(n*phi))
			E_phi2 = 0.5*Vbarr * (1-cos(n*phi0[1])*cos(n*phi))
			E = min(E_phi1, E_phi2)

			if E == E_phi1:
				phi0 = phi0[0]
			else:
				phi0 = phi0[1]
			Ephi += E

		### VDW ENERGY
		#Evdw = DIJ * (-2 * (xIJ/x)**6 + (xIJ/x)**12)
		param = params['vdw']
		for a1, a2 in mol.unique_pairs_3:
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))

			x = a1.distance_to(a2)

			x12 = param[sorted_at]['x12']
			D12 = param[sorted_at]['D12']

			x_6 = (x12/x)**6

			E = D12 * (x_6**2 - 2*x_6)
			Evdw += E

		return (Er + Etheta + Ephi + Eohm + Evdw + Eel) * 0.04336 / 27.211386245988

	
	def get_energy(self, mol, morse=True):
		params = self.prepare_parameters(mol)
		#### utility functions:
		def distance(a1, a2):
			return np.linalg.norm(a1.position - a2.position)

		def bond_angle(a1, a2, a3):
			u = a1.position - a2.position
			v = a3.position - a2.position
			return np.arccos((u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v)))

		def torsion_angle(a1, a2, a3, a4):
			b1 = a2.position - a1.position
			b2 = a3.position - a2.position
			b3 = a4.position - a3.position

			return math.atan2(np.dot(np.cross(np.cross(b1, b2), np.cross(b2, b3)), b2/np.linalg.norm(b2)), np.dot(np.cross(b1, b2), np.cross(b2, b3)))


		#### ENERGY CALCULATION
		#E = Er + Etheta + Ephi + Eohm + Evdw + Eel

		if self.verbose:
			print('Molecule:')
			for a in mol.atoms:
				print(f'\t{a.element: <2} {a.position[0]: >7.3f} {a.position[1]: >7.3f} {a.position[2]: >7.3f} {a.uff_atom_type}')

		Er = 0
		Etheta = 0
		Ephi = 0
		Eohm = 0
		Evdw = 0
		Eel = 0

		### BOND STRETCH ENERGY
		#Er = kIJ/2(r-rIJ)**2
		param = params['stretch']

		if self.verbose:
			print('BOND STRETCH ENERGY')
			print('\tATOM 1 | ATOM 2 | BO  | BOND LEN | IDEAL LEN | FORCE CONSTANT | DELTA  | ENERGY')
		
		for a1, a2, order in mol.unique_bonds:
			#get atome types
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))
			#if a1 and a2 are conjugated, set order to 1.5
			if a1t == a2t == 'C_R':
				order = 1.5

			r = distance(a1,a2) #get distance

			#get params
			DIJ = param[sorted_at]['DIJ']
			rIJ = param[sorted_at]['rIJ']
			kIJ = param[sorted_at]['kIJ']

			#calculate energy
			if morse:
				E = DIJ * (exp(-sqrt(kIJ/(2*DIJ)) * (r-rIJ)) - 1)**2
			else:
				E = kIJ/2 * (r - rIJ)**2
			Er += E

			if self.verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {order: <3.1f} | {r: <8.3f} | {rIJ: <9.3f} | {kIJ: <14.3f} | {r-rIJ: <6.3f} | {E: <.3f}')
		
		if self.verbose:
			print(f'\tTOTAL BONDING ENERGY = {self.convert_energy(Er):.3f} {self.unit}\n')


		### BOND ANGLE ENERGY
		#Etheta = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
		param = params['bend']

		if self.verbose:
			print('BOND ANGLE ENERGY')
			print('\tATOM 1 | ATOM 2 | ATOM 3 | theta0  | theta   | FORCE CONSTANT | DELTA  | ENERGY')

		for a1, a2, a3 in mol.unique_bond_angles:
			#get atom types
			a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t)))

			theta = bond_angle(a1,a2,a3) #get bond angle

			#get params
			theta0 = param[sorted_at]['theta0']
			Z13 = param[sorted_at]['Z13']
			r1223 = param[sorted_at]['r1223']
			r12 = params['stretch'][tuple(sorted((a1t, a2t)))]['rIJ']
			r12_2 = r12 * r12 #squared
			r23 = params['stretch'][tuple(sorted((a2t, a3t)))]['rIJ']
			r23_2 = r23 * r23
			C0 = param[sorted_at]['C0']
			C1 = param[sorted_at]['C1']
			C2 = param[sorted_at]['C2']

			beta = 664.12/r1223
			costheta0 = cos(theta0)
			r13_2 = r12_2 + r23_2 - 2 * r1223 * cos(theta0)
			KIJK = beta * (Z13/r13_2**2.5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13_2**2 * costheta0)

			E = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
			Etheta += E

			if self.verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {a3t: <6} | {theta0: <7.3f} | {theta: <7.3f} | {KIJK: <14.3f} | {theta-theta0: >6.3f} | {E: <.3f}')

		if self.verbose:
			print(f'\tTOTAL ANGLE ENERGY = {self.convert_energy(Etheta):.3f} {self.unit}\n')


		### BOND TORSION ENERGY
		#Ephi = Vphi/2 * (1 - cos(n*phi0)*cos(n*phi))
		param = params['torsion']

		if self.verbose:
			print('BOND TORSION ENERGY')
			print('\tATOM 1 | ATOM 2 | ATOM 3 | ATOM 4 | phi0    | phi     | FORCE CONSTANT | DELTA  | ENERGY')

		for a1, a2, a3, a4 in mol.unique_torsion_angles:
			#get atom types
			a1t, a2t, a3t, a4t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type, a4.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t, a3t, a4t)))

			phi = torsion_angle(a1, a2, a3, a4) #get torsion angle

			Vbarr = param[sorted_at]['Vbarr']
			phi0 = param[sorted_at]['phi0']
			n = param[sorted_at]['n']

			E_phi1 = 0.5*Vbarr * (1-cos(n*phi0[0])*cos(n*phi))
			E_phi2 = 0.5*Vbarr * (1-cos(n*phi0[1])*cos(n*phi))
			E = min(E_phi1, E_phi2)

			if E == E_phi1:
				phi0 = phi0[0]
			else:
				phi0 = phi0[1]
			Ephi += E

			if self.verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {a3t: <6} | {a4t: <6} | {phi0%(math.pi): <7.3f} | {phi%(math.pi): <7.3f} | {Vbarr: <14.3f} | {(phi-phi0)%(math.pi): >6.3f} | {E: <.3f}')

		if self.verbose:
			print(f'\tTOTAL TORSION ENERGY = {self.convert_energy(Ephi):.3f} {self.unit}\n')


		### VDW ENERGY
		#Evdw = DIJ * (-2 * (xIJ/x)**6 + (xIJ/x)**12)
		param = params['vdw']

		if self.verbose:
			print('VDW ENERGY')
			print('\tATOM 1 | ATOM 2 | x      | x12    | D12    | ENERGY')

		for a1, a2 in mol.unique_pairs_3:
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))

			x = a1.distance_to(a2)

			x12 = param[sorted_at]['x12']
			D12 = param[sorted_at]['D12']

			x_6 = (x12/x)**6

			E = D12 * (x_6**2 - 2*x_6)
			Evdw += E

			if self.verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {x: <6.3f} | {x12: <6.3f} | {D12: <6.3f} | {E: >7.3f}')

		if self.verbose:
			print(f'\tTOTAL VDW ENERGY = {self.convert_energy(Evdw):.3f} {self.unit}\n')


		Etot = Er + Etheta + Ephi + Eohm + Evdw + Eel
		if self.verbose:
			print(f'TOTAL ENERGY = {self.convert_energy(Etot):.3f} {self.unit}\n')

		return self.convert_energy(Etot)



	# def get_energy(self, mol, morse=True, ):
	# 	#### utility functions:
	# 	def distance(a1, a2):
	# 		return np.linalg.norm(a1.position - a2.position)

	# 	def bond_angle(a1, a2, a3):
	# 		u = a1.position - a2.position
	# 		v = a3.position - a2.position
	# 		return np.arccos((u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v)))

	# 	def torsion_angle(a1, a2, a3, a4):
	# 		b1 = a2.position - a1.position
	# 		b2 = a3.position - a2.position
	# 		b3 = a4.position - a3.position

	# 		return math.atan2(np.dot(np.cross(np.cross(b1, b2), np.cross(b2, b3)), b2/np.linalg.norm(b2)), np.dot(np.cross(b1, b2), np.cross(b2, b3)))

	# 	#set UFF atom types for all atoms
	# 	self.set_atom_types(mol)
	# 	params = self.parameters
		
	# 	#store optimal bond lengths to ease calculation
		

	# 	atoms = mol.atoms


	# 	#### ENERGY CALCULATION
	# 	#E = Er + Etheta + Ephi + Eohm + Evdw + Eel

	# 	### BOND STRETCH ENERGY
	# 	#Er = kIJ/2(r-rIJ)**2
	# 	Er = 0
	# 	Etheta = 0
	# 	Ephi = 0
	# 	Eohm = 0
	# 	Evdw = 0
	# 	Eel = 0

	# 	if verbose:
	# 		print('BOND STRETCH ENERGY')
	# 		print('\tATOM 1 | ATOM 2 | BO  | BOND LEN | IDEAL LEN | FORCE CONSTANT | DELTA  | ENERGY')
		
	# 	for a1, a2, order in mol.unique_bonds:
	# 		#get atome types
	# 		a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
	# 		sorted_at = tuple(sorted((a1t, a2t)))
	# 		#if a1 and a2 are conjugated, set order to 1.5
	# 		if a1t == a2t == 'C_R':
	# 			order = 1.5

	# 		r = distance(a1,a2) #get distance

	# 		## CALCULATE rIJ IF NOT YET STORED:
	# 		if not sorted_at in self.optimal_bond_lengths.keys():
	# 			rI, rJ = params['valence_bond'][a1t], params['valence_bond'][a2t]
	# 			xI, xJ = params['electro_negativity'][a1t], params['electro_negativity'][a2t]

	# 			#rIJ = rI + rJ + rBO + rEN
	# 			rBO = -0.1332 * (rI + rJ) * log(order)
	# 			rEN = rI * rJ * (sqrt(xI) - sqrt(xJ))**2 / (xI*rI + xJ*rJ)

	# 			rIJ = rI + rJ + rBO - rEN

	# 			self.optimal_bond_lengths[sorted_at] = rIJ
	# 		else:
	# 			rIJ = self.optimal_bond_lengths[sorted_at]

			
	# 		#kIJ = 664.12*ZI*ZJ/rIJ**3
	# 		ZI, ZJ = params['effective_charge'][a1t], params['effective_charge'][a2t]
	# 		kIJ = 664.12 * ZI * ZJ / rIJ**3


	# 		if morse:
	# 			DIJ = order * 70
	# 			E = DIJ * (exp(-sqrt(kIJ/(2*DIJ)) * (r-rIJ)) - 1)**2
	# 		else:
	# 			E = kIJ/2 * (r - rIJ)**2
	# 		Er += E

	# 		if verbose:
	# 			print(f'\t{a1t: <6} | {a2t: <6} | {order: <3.1f} | {r: <8.3f} | {rIJ: <9.3f} | {kIJ: <14.3f} | {r-rIJ: <6.3f} | {E: <.3f}')
		
	# 	if verbose:
	# 		print(f'\tTOTAL BONDING ENERGY = {Er:.3f} kcal/mol; {Er*4.182:.3f} kJ/mol\n')


	# 	### BOND ANGLE ENERGY
	# 	#Etheta = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
	# 	if verbose:
	# 		print('BOND ANGLE ENERGY')
	# 		print('\tATOM 1 | ATOM 2 | ATOM 3 | theta0  | theta   | FORCE CONSTANT | DELTA  | ENERGY')

	# 	for a1, a2, a3 in mol.unique_bond_angles:
	# 		#get atom types
	# 		a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type

	# 		theta = bond_angle(a1,a2,a3) #get bond angle
	# 		theta0 = params['valence_angle'][a2t] * math.pi/180
	# 		costheta0 = cos(theta0)
	# 		sintheta0 = sin(theta0)

	# 		r12 = self.optimal_bond_lengths[tuple(sorted((a1t, a2t)))]
	# 		r12_2 = r12 * r12 #squared
	# 		r23 = self.optimal_bond_lengths[tuple(sorted((a2t, a3t)))]
	# 		r23_2 = r23 * r23
	# 		r1223 = r12 * r23
	# 		r13 = sqrt(r12_2 + r23_2 - 2 * r1223 * cos(theta))

	# 		Z1, Z3 = params['effective_charge'][a1t], params['effective_charge'][a3t]

	# 		beta = 664.12/r1223

	# 		KIJK = beta * (Z1*Z3/r13**5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13**2 * costheta0)

	# 		C2 = 1/(4*sintheta0**2)
	# 		C1 = -4 * C2 * costheta0
	# 		C0 = C2 * (2 * costheta0**2 + 1)

	# 		E = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))

	# 		Etheta += E
	# 		if verbose:
	# 			print(f'\t{a1t: <6} | {a2t: <6} | {a3t: <6} | {theta0: <7.3f} | {theta: <7.3f} | {KIJK: <14.3f} | {theta-theta0: >6.3f} | {E: <.3f}')

	# 	if verbose:
	# 		print(f'\tTOTAL ANGLE ENERGY = {Etheta:.3f} kcal/mol; {Etheta*4.182:.3f} kJ/mol\n')


	# 	### BOND TORSION ENERGY
	# 	#Ephi = Vphi/2 * (1 - cos(n*phi0)*cos(n*phi))
	# 	if verbose:
	# 		print('BOND TORSION ENERGY')
	# 		print('\tATOM 1 | ATOM 2 | ATOM 3 | ATOM 4 | phi0    | phi     | FORCE CONSTANT | DELTA  | ENERGY')

	# 	for a1, a2, a3, a4 in mol.unique_torsion_angles:
	# 		a1t, a2t, a3t, a4t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type, a4.uff_atom_type
			
	# 		phi = torsion_angle(a1, a2, a3, a4)

	# 		Vbarr = 1
	# 		if a2.hybridisation == a3.hybridisation == 3:
	# 			V2, V3 = params['sp3_torsional_barrier_params'][a2t], params['sp3_torsional_barrier_params'][a3t]
	# 			Vbarr = sqrt(V2*V3)
	# 			n = 3
	# 			phi0 = math.pi, math.pi/3 #two different phi0 possible

	# 		if (a2.hybridisation == 3 and a3.hybridisation == 2) or (a2.hybridisation == 2 and a3.hybridisation == 3):
	# 			Vbarr = 1
	# 			n = 6
	# 			phi0 = 0, 0

	# 		if a2.hybridisation == a3.hybridisation == 2:
	# 			U2 = params['sp2_torsional_barrier_params'][a2t] # period starts at 1
	# 			U3 = params['sp2_torsional_barrier_params'][a3t]
	# 			Vbarr = 5*sqrt(U2*U3)*(1+4.18*log(mol.get_bond_order(a2, a3)))
	# 			n = 2
	# 			phi0 = math.pi, 1.047198

	# 		E_phi1 = 0.5*Vbarr * (1-cos(n*phi0[0])*cos(n*phi))
	# 		E_phi2 = 0.5*Vbarr * (1-cos(n*phi0[1])*cos(n*phi))
	# 		E = min(E_phi1, E_phi2)
	# 		if E == E_phi1:
	# 			phi0 = phi0[0]
	# 		else:
	# 			phi0 = phi0[1]
	# 		Ephi += E

	# 		if verbose:
	# 			print(f'\t{a1t: <6} | {a2t: <6} | {a3t: <6} | {a4t: <6} | {phi0%(math.pi): <7.3f} | {phi%(math.pi): <7.3f} | {Vbarr: <14.3f} | {(phi-phi0)%(math.pi): >6.3f} | {E: <.3f}')
	# 	if verbose:
	# 		print(f'\tTOTAL TORSION ENERGY = {Ephi:.3f} kcal/mol; {Ephi*4.182:.3f} kJ/mol\n')


	# 	### VDW ENERGY
	# 	#Evdw = DIJ * (-2 * (xIJ/x)**6 + (xIJ/x)**12)
	# 	if verbose:
	# 		print('VDW ENERGY')
	# 		print('\tATOM 1 | ATOM 2 | x      | x12    | D12    | ENERGY')

	# 	for a1, a2 in mol.unique_pairs_3:
	# 		a1t, a2t = a1.uff_atom_type, a2.uff_atom_type

	# 		x1, x2 = params['nonbond_distance'][a1t], params['nonbond_distance'][a2t]
	# 		x12 = (x1+x2)/2

	# 		D1, D2 = params['nonbond_energy'][a1t], params['nonbond_energy'][a2t]
	# 		D12 = sqrt(D1*D2)

	# 		x = a1.distance_to(a2)

	# 		x_6 = (x12/x)**6

	# 		E = D12 * (x_6**2 - 2*x_6)

	# 		Evdw += E

	# 		if verbose:
	# 			print(f'\t{a1t: <6} | {a2t: <6} | {x: <6.3f} | {x12: <6.3f} | {D12: <6.3f} | {E: >7.3f}')
	# 	if verbose:
	# 		print(f'\tTOTAL VDW ENERGY = {Evdw:.3f} kcal/mol; {Evdw*4.182:.3f} kJ/mol\n')




	# 	Etot = Er + Etheta + Ephi + Eohm + Evdw + Eel
	# 	if verbose:
	# 		print(f'TOTAL ENERGY = {Etot:.3f} kcal/mol; {Etot*4.182:.3f} kJ/mol\n')

	# 	return Etot


	def get_gradient(self, mol, morse=True, verbose=False):
		def distance(a1, a2):
			return np.linalg.norm(a1.position - a2.position)

		Etot = self.get_energy(mol, morse=morse, verbose=False)
		params = self.parameters
		atoms = mol.atoms


		grad = {a:np.zeros(3) for a in atoms}
		for a1, a2, order in mol.unique_bonds:
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type
			sorted_at = tuple(sorted((a1t, a2t)))
			rIJ = self.optimal_bond_lengths[sorted_at]
			ZI, ZJ = params['effective_charge'][a1t], params['effective_charge'][a2t]
			kIJ = 664.12 * ZI * ZJ / rIJ**3

			r = distance(a1, a2)

			grad[a1] = grad[a1] + kIJ * (r - rIJ) * a1.position / r
			grad[a2] = grad[a2] + kIJ * (r - rIJ) * a2.position / r

		# for a1, a2, a3 in mol.unique_bond_angles:
		# 	try:
		# 		a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
		# 		u = a1.position - a2.position
		# 		v = a3.position - a2.position

		# 		g = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
		# 		if g**2 > 1:
		# 			print(g**2)
		# 		theta = np.arccos(g)

		# 		mu, mv = np.sqrt(u @ u), np.sqrt(v @ v)

		# 		dmda1 = -1/(u * mu * mv)
		# 		dmda3 = -1/(v * mu * mv)
		# 		dmda2 = -dmda1 - dmda3

		# 		dgda1 = v/dmda1
		# 		dgda2 = (a2.position @ a2.position - a1.position - a3.position)*dmda2
		# 		dgda3 = u/dmda3
		# 		dtdg = -1/sqrt(1-g**2)
		# 		dtda1 = dtdg * dgda1
		# 		dtda2 = dtdg * dgda2
		# 		dtda3 = dtdg * dgda3

		# 		theta0 = params['valence_angle'][a2t] * math.pi/180
		# 		costheta0 = cos(theta0)
		# 		sintheta0 = sin(theta0)

		# 		r12 = self.optimal_bond_lengths[tuple(sorted((a1t, a2t)))]
		# 		r12_2 = r12 * r12 #squared
		# 		r23 = self.optimal_bond_lengths[tuple(sorted((a2t, a3t)))]
		# 		r23_2 = r23 * r23
		# 		r1223 = r12 * r23
		# 		r13 = sqrt(r12_2 + r23_2 - 2 * r1223 * cos(theta))

		# 		Z1, Z3 = params['effective_charge'][a1t], params['effective_charge'][a3t]

		# 		beta = 664.12/r1223

		# 		KIJK = beta * (Z1*Z3/r13**5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13**2 * costheta0)

		# 		C2 = 1/(4*sintheta0**2)
		# 		C1 = -4 * C2 * costheta0

		# 		f = (KIJK*C1*sin(theta) + 2*KIJK*C2*sin(2*theta))
		# 		grad[a1] = grad[a1] - f *dtda1
		# 		grad[a2] = grad[a2] - f *dtda2
		# 		grad[a3] = grad[a3] - f *dtda3
		# 	except:
		# 		print(u, v)
		# 		raise

		# for a1, a2, a3 in mol.unique_bond_angles:
		# 	a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
		# 	u = a1.position - a2.position
		# 	v = a3.position - a2.position

		# 	mu, mv = np.sqrt(u @ u), np.sqrt(v @ v)

		# 	g = (u @ v) / (mu * mv)
		# 	theta = np.arccos(g)


		# 	#GET C2 and C1
		# 	theta0 = params['valence_angle'][a2t] * math.pi/180
		# 	costheta0 = cos(theta0)
		# 	sintheta0 = sin(theta0)

		# 	r12 = self.optimal_bond_lengths[tuple(sorted((a1t, a2t)))]
		# 	r12_2 = r12 * r12 #squared
		# 	r23 = self.optimal_bond_lengths[tuple(sorted((a2t, a3t)))]
		# 	r23_2 = r23 * r23
		# 	r1223 = r12 * r23
		# 	r13 = sqrt(r12_2 + r23_2 - 2 * r1223 * cos(theta))

		# 	Z1, Z3 = params['effective_charge'][a1t], params['effective_charge'][a3t]

		# 	beta = 664.12/r1223

		# 	KIJK = beta * (Z1*Z3/r13**5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13**2 * costheta0)

		# 	C2 = 1/(4*sintheta0**2)
		# 	C1 = -4 * C2 * costheta0

		# 	f = (KIJK*C1*sin(theta) + 2*KIJK*C2*sin(2*theta))


		# 	#GET DERIVATIVE
		# 	muv = mu*mv
		# 	grad[a2] = grad[a2] + f * ( 1/math.sqrt(1-g**2) * ((v/2 + u/2)/(muv) - (v*mu**2/2 + u*mv**2/2)/(muv**3)) )

		return grad



	




