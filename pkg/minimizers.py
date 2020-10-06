import numpy as np 
import pubchempy as pcp
import os
import periodictable as pt
from math import sin, cos, log, sqrt, exp
import math
import networkx as nx
import itertools

try:
	import data
except:
	import pkg.data as data




def minimize_molecule(self, mol, force_field='UFF'):
	if force_field == 'UFF':
		ff = UFF()





class IMUFF:
	def __init__(self, *args, **kwargs):
		self.parameters = data.FF_UFF_PARAMETERS
		self.optimal_bond_lengths = {}




class UFF():
	'''
	RESOURCES:
	http://towhee.sourceforge.net/forcefields/uff.html
	A. K. Rappe; C. J. Casewit; K. S. Colwell; W. A. Goddard III; W. M. Skiff; "UFF, a Full Periodic Table Force Field for Molecular Mechanics and Molecular Dynamics Simulations", J. Am. Chem. Soc. 114 10024-10035 (1992). 
	'''

	def __init__(self, *args, **kwargs):
		self.parameters = data.FF_UFF_PARAMETERS
		self.optimal_bond_lengths = {}


	def set_atom_types(self, mol):
		for atom in mol.atoms:
			el = atom.element 
			el += '_'*(len(el)==1)
			if atom.ring == 'AR':
				el += 'R'
			elif atom.hybridisation != 0:
				el += str(atom.hybridisation)

			atom.uff_atom_type = el


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

		for a1, a2, a3 in mol.unique_bond_angles:
			a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type
			u = a1.position - a2.position
			v = a3.position - a2.position

			mu, mv = np.sqrt(u @ u), np.sqrt(v @ v)

			g = (u @ v) / (mu * mv)
			theta = np.arccos(g)


			#GET C2 and C1
			theta0 = params['valence_angle'][a2t] * math.pi/180
			costheta0 = cos(theta0)
			sintheta0 = sin(theta0)

			r12 = self.optimal_bond_lengths[tuple(sorted((a1t, a2t)))]
			r12_2 = r12 * r12 #squared
			r23 = self.optimal_bond_lengths[tuple(sorted((a2t, a3t)))]
			r23_2 = r23 * r23
			r1223 = r12 * r23
			r13 = sqrt(r12_2 + r23_2 - 2 * r1223 * cos(theta))

			Z1, Z3 = params['effective_charge'][a1t], params['effective_charge'][a3t]

			beta = 664.12/r1223

			KIJK = beta * (Z1*Z3/r13**5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13**2 * costheta0)

			C2 = 1/(4*sintheta0**2)
			C1 = -4 * C2 * costheta0

			f = (KIJK*C1*sin(theta) + 2*KIJK*C2*sin(2*theta))


			#GET DERIVATIVE
			muv = mu*mv
			grad[a2] = grad[a2] + f * ( 1/math.sqrt(1-g**2) * ((v/2 + u/2)/(muv) - (v*mu**2/2 + u*mv**2/2)/(muv**3)) )

		return grad



	def get_energy(self, mol, morse=True, verbose=False):
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

		#set UFF atom types for all atoms
		self.set_atom_types(mol)
		params = self.parameters
		
		#store optimal bond lengths to ease calculation
		

		atoms = mol.atoms


		#### ENERGY CALCULATION
		#E = Er + Etheta + Ephi + Eohm + Evdw + Eel

		### BOND STRETCH ENERGY
		#Er = kIJ/2(r-rIJ)**2
		Er = 0
		Etheta = 0
		Ephi = 0
		Eohm = 0
		Evdw = 0
		Eel = 0

		if verbose:
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

			## CALCULATE rIJ IF NOT YET STORED:
			if not sorted_at in self.optimal_bond_lengths.keys():
				rI, rJ = params['valence_bond'][a1t], params['valence_bond'][a2t]
				xI, xJ = params['electro_negativity'][a1t], params['electro_negativity'][a2t]

				#rIJ = rI + rJ + rBO + rEN
				rBO = -0.1332 * (rI + rJ) * log(order)
				rEN = rI * rJ * (sqrt(xI) - sqrt(xJ))**2 / (xI*rI + xJ*rJ)

				rIJ = rI + rJ + rBO - rEN

				self.optimal_bond_lengths[sorted_at] = rIJ
			else:
				rIJ = self.optimal_bond_lengths[sorted_at]

			
			#kIJ = 664.12*ZI*ZJ/rIJ**3
			ZI, ZJ = params['effective_charge'][a1t], params['effective_charge'][a2t]
			kIJ = 664.12 * ZI * ZJ / rIJ**3


			if morse:
				DIJ = order * 70
				E = DIJ * (exp(-sqrt(kIJ/(2*DIJ)) * (r-rIJ)) - 1)**2
			else:
				E = kIJ/2 * (r - rIJ)**2
			Er += E

			if verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {order: <3.1f} | {r: <8.3f} | {rIJ: <9.3f} | {kIJ: <14.3f} | {r-rIJ: <6.3f} | {E: <.3f}')
		
		if verbose:
			print(f'\tTOTAL BONDING ENERGY = {Er:.3f} kcal/mol; {Er*4.182:.3f} kJ/mol\n')


		### BOND ANGLE ENERGY
		#Etheta = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))
		if verbose:
			print('BOND ANGLE ENERGY')
			print('\tATOM 1 | ATOM 2 | ATOM 3 | theta0  | theta   | FORCE CONSTANT | DELTA  | ENERGY')

		for a1, a2, a3 in mol.unique_bond_angles:
			#get atom types
			a1t, a2t, a3t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type

			theta = bond_angle(a1,a2,a3) #get bond angle
			theta0 = params['valence_angle'][a2t] * math.pi/180
			costheta0 = cos(theta0)
			sintheta0 = sin(theta0)

			r12 = self.optimal_bond_lengths[tuple(sorted((a1t, a2t)))]
			r12_2 = r12 * r12 #squared
			r23 = self.optimal_bond_lengths[tuple(sorted((a2t, a3t)))]
			r23_2 = r23 * r23
			r1223 = r12 * r23
			r13 = sqrt(r12_2 + r23_2 - 2 * r1223 * cos(theta))

			Z1, Z3 = params['effective_charge'][a1t], params['effective_charge'][a3t]

			beta = 664.12/r1223

			KIJK = beta * (Z1*Z3/r13**5) * r1223 * (3*r1223 * (1 - costheta0**2) - r13**2 * costheta0)

			C2 = 1/(4*sintheta0**2)
			C1 = -4 * C2 * costheta0
			C0 = C2 * (2 * costheta0**2 + 1)

			E = KIJK * (C0 + C1 * cos(theta) + C2 * cos(2*theta))

			Etheta += E
			if verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {a3t: <6} | {theta0: <7.3f} | {theta: <7.3f} | {KIJK: <14.3f} | {theta-theta0: >6.3f} | {E: <.3f}')

		if verbose:
			print(f'\tTOTAL ANGLE ENERGY = {Etheta:.3f} kcal/mol; {Etheta*4.182:.3f} kJ/mol\n')


		### BOND TORSION ENERGY
		#Ephi = Vphi/2 * (1 - cos(n*phi0)*cos(n*phi))
		if verbose:
			print('BOND TORSION ENERGY')
			print('\tATOM 1 | ATOM 2 | ATOM 3 | ATOM 4 | phi0    | phi     | FORCE CONSTANT | DELTA  | ENERGY')

		for a1, a2, a3, a4 in mol.unique_torsion_angles:
			a1t, a2t, a3t, a4t = a1.uff_atom_type, a2.uff_atom_type, a3.uff_atom_type, a4.uff_atom_type
			
			phi = torsion_angle(a1, a2, a3, a4)

			Vbarr = 1
			if a2.hybridisation == a3.hybridisation == 3:
				V2, V3 = params['sp3_torsional_barrier_params'][a2t], params['sp3_torsional_barrier_params'][a3t]
				Vbarr = sqrt(V2*V3)
				n = 3
				phi0 = math.pi, math.pi/3 #two different phi0 possible

			if (a2.hybridisation == 3 and a3.hybridisation == 2) or (a2.hybridisation == 2 and a3.hybridisation == 3):
				Vbarr = 1
				n = 6
				phi0 = 0, 0

			if a2.hybridisation == a3.hybridisation == 2:
				U2 = params['sp2_torsional_barrier_params'][a2t] # period starts at 1
				U3 = params['sp2_torsional_barrier_params'][a3t]
				Vbarr = 5*sqrt(U2*U3)*(1+4.18*log(mol.get_bond_order(a2, a3)))
				n = 2
				phi0 = math.pi, 1.047198

			E_phi1 = 0.5*Vbarr * (1-cos(n*phi0[0])*cos(n*phi))
			E_phi2 = 0.5*Vbarr * (1-cos(n*phi0[1])*cos(n*phi))
			E = min(E_phi1, E_phi2)
			if E == E_phi1:
				phi0 = phi0[0]
			else:
				phi0 = phi0[1]
			Ephi += E

			if verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {a3t: <6} | {a4t: <6} | {phi0%(math.pi): <7.3f} | {phi%(math.pi): <7.3f} | {Vbarr: <14.3f} | {(phi-phi0)%(math.pi): >6.3f} | {E: <.3f}')
		if verbose:
			print(f'\tTOTAL TORSION ENERGY = {Ephi:.3f} kcal/mol; {Ephi*4.182:.3f} kJ/mol\n')


		### VDW ENERGY
		#Evdw = DIJ * (-2 * (xIJ/x)**6 + (xIJ/x)**12)
		if verbose:
			print('VDW ENERGY')
			print('\tATOM 1 | ATOM 2 | x      | x12    | D12    | ENERGY')

		for a1, a2 in mol.unique_pairs_3:
			a1t, a2t = a1.uff_atom_type, a2.uff_atom_type

			x1, x2 = params['nonbond_distance'][a1t], params['nonbond_distance'][a2t]
			x12 = (x1+x2)/2

			D1, D2 = params['nonbond_energy'][a1t], params['nonbond_energy'][a2t]
			D12 = sqrt(D1*D2)

			x = a1.distance_to(a2)

			x_6 = (x12/x)**6

			E = D12 * (x_6**2 - 2*x_6)

			Evdw += E

			if verbose:
				print(f'\t{a1t: <6} | {a2t: <6} | {x: <6.3f} | {x12: <6.3f} | {D12: <6.3f} | {E: >7.3f}')
		if verbose:
			print(f'\tTOTAL VDW ENERGY = {Evdw:.3f} kcal/mol; {Evdw*4.182:.3f} kJ/mol\n')




		Etot = Er + Etheta + Ephi + Eohm + Evdw + Eel
		if verbose:
			print(f'TOTAL ENERGY = {Etot:.3f} kcal/mol; {Etot*4.182:.3f} kJ/mol\n')

		return Etot




