import numpy as np 
import pubchempy as pcp
import Bio.PDB as pdb
import os
import periodictable as pt
from math import sin, cos
import math
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length

try:
	import data
	import display
	# import basissets
except:
	import pkg.data as data
	import pkg.display as display
	import pkg.basissets as basissets



##### ========================================== FUNCTIONS FOR LOADING AND SAVING MOLECULES ========================================== ####

def load_from_file(file):
	name = file.split('/')[-1].strip('.xyz').capitalize()
	name = name.split('\\')[-1].strip('.xyz').capitalize()

	elements = list(np.loadtxt(file, skiprows=2, usecols=0, dtype=str))
	positions = list(np.loadtxt(file, skiprows=2, usecols=(1,2,3), dtype=float))
		

	return Molecule(name, elements, positions)


def get_from_pubchem(name, record_type='3d'):
	try:
		name = int(name)
	except:
		pass

	mol = pcp.get_compounds(name, ('name', 'cid')[type(name) is int], record_type=record_type)
	if len(mol) == 0:
		print(f'No compound or 3d structure with name {name} found on Pubchem. Please try again with CID.')
		return 

	print(f'{len(mol)} compounds found on Pubchem with name {name}.')
	mol = mol[0]
	positions = np.asarray([[a.x, a.y, a.z] for a in mol.atoms])
	positions = np.where(positions == None, 0, positions).astype(float)
	elements = np.asarray([a.element for a in mol.atoms])

	mol = Molecule(name.capitalize(), elements, positions)
	path = structures_folder + name + '.xyz'
	save_to_xyz(mol, path)

	return mol


def get_from_pdb(name):
	pdbl = pdb.PDBList()
	pdbl.retrieve_pdb_file(name, pdir=structures_folder, file_format='pdb')
	parser = pdb.PDBParser(PERMISSIVE=True, QUIET=True)
	data = parser.get_structure(name, structures_folder + 'pdb' + name + '.ent')
	model = list(data.get_models())[0]
	chains = list(model.get_chains())
	residues = [list(chain.get_residues()) for chain in chains]
	residues = [residue for reslist in residues for residue in reslist]
	atoms = [atom for residue in residues for atom in residue.get_atoms()]
	print(dir(atoms[0]))

	return mol


def find_mol(name, root=None, exact=True, record_type='3d'):
	'''
	Function that returns the path to a file in a folder in root or on pubchem
	If found on pubchem, save to root

	name - name of molecule (str)
	root - path to root folder (str)
	exact - specify if file should exactly match name
	'''

	if root is None: root = structures_folder

	#search root
	paths = []
	for d, _, files in os.walk(root):
		for file in files:
			if file.endswith('.xyz'):
				if exact:
					if name.lower() == file.strip('.xyz').lower():
						paths.append(d + '\\' + file)
				else:
					if name.lower() in file.strip('.xyz').lower():
						paths.append(d + '\\' + file)

	if len(paths) > 0:
		return paths[0]

	#if not found, search pubchem
	if get_from_pubchem(name, record_type=record_type) is None:
		return

	return find_mol(name, root=root)


def save_to_xyz(mol, path, comment=''):
	'''
	Function that writes a molecule to xyz format

	mol - molecule object
	path - path to write xyz file to. If not a valid path is given it will write to a default directory
	'''

	# if not os.path.exists(path): path = structures_folder + f'{path}.xyz'

	elements = mol.elements
	positions = mol.positions

	with open(path, 'w+') as f:
		f.write(f'{len(elements)}\n')
		f.write(comment + '\n')
		for i, e in enumerate(elements):
			f.write(f'{e: <2} \t {positions[i][0]: >8.5f} \t {positions[i][1]: >8.5f} \t {positions[i][2]: >8.5f}\n')

	print(f'Saved {mol.name} to {path}.')


def load_mol(name, download_from_pubchem=False, download_from_pdb=False, record_type='3d'):
	if download_from_pubchem:
		mol = get_from_pubchem(name, record_type=record_type)
	elif download_from_pdb:
		mol = get_from_pdb(name)
	else:
		path = find_mol(name.capitalize())
		mol = load_from_file(path)
	return mol




##### ========================================== ATOM AND MOLECULE CLASS ========================================== ####

class Atom:
	def __init__(self, element=None, position=None, charge=None, label='', index=0):
		self.element = element
		self.atom_number = pt.elements.symbol(element).number
		self.position = position
		self.charge = charge
		self.label = label
		self.index = index

		self.set_max_valence()
		self.set_mass()
		self.set_covalent_radius()
		self.set_colour()


	def __repr__(self):
		return f'{self.element}({self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f})'


	def copy(self):
		a = Atom(self.element, self.position, self.charge)
		a.hybridisation = self.hybridisation
		return a


	def distance_to(self, p):
		if type(p) is Atom:
			return np.linalg.norm(self.position - p.position)
		return np.linalg.norm(self.position - p)


	def set_max_valence(self):
		#retrieve from data files
		try:
			self.max_valence = int(data.MAX_VALENCE[self.atom_number][0])
		#default to 1
		except:
			self.max_valence = 1


	def set_mass(self):
		self.mass = pt.elements[self.atom_number].mass


	def set_covalent_radius(self):
		self.covalent_radius = pt.elements[self.atom_number].covalent_radius


	def set_colour(self):
		c = data.ATOM_COLOURS[self.atom_number]
		self.colour = tuple([int(i) for i in c])


class Molecule:
	def __init__(self, name=[], elements=[], positions=[], charges=[], atoms=[], bonds=[]):
		self.name = name

		assert(len(elements) == len(positions))
		self.elements = elements
		self.positions = positions

		self.charges = charges

		#get atoms from elements, positions
		self.atoms = [Atom(elements[i], positions[i], label=i, index=i) for i in range(len(self.elements))]
		#If a list of atoms is already given, append it to the list just produced
		self.atoms += atoms

		self._guess_bond_order_iters = 0

		if len(self.atoms) < 200:
			if bonds == []:
				self.bonds = self.guess_bond_orders()
			else:
				self.bonds={self.atoms[a1]: {self.atoms[a2]: order  for a2, order in b.items()} for a1, b in bonds.items()}

			self.unique_bonds = self.get_unique_bonds()
			self.graph_representation = self.get_graph_representation()
			self.unique_pairs = self.get_unique_atom_pairs()
			self.unique_pairs_3 = self.get_unique_atom_pairs(3) #used in forcefields, vdw forces are usually taken only for atoms >= 3 bonds apart
			self.unique_bond_angles = self.get_unique_bond_angles()
			self.unique_torsion_angles = self.get_unique_torsion_angles()
			
			self.rings = self.detect_rings()
		else:
			self.bonds = self.initial_bonding()

		self.center()


	def __repr__(self):
		string = ''
		string += self.name + '\n'

		for a in self.atoms:
			string += f'{a.element:2s}\t{a.position[0]: .5f}\t{a.position[1]: .5f}\t{a.position[2]: .5f}\n'

		return string


	def copy(self):
		return Molecule(self.name, atoms=[a.copy() for a in self.atoms], bonds={self.atoms.index(a1): {self.atoms.index(a2): order  for a2, order in b.items()} for a1, b in self.bonds.items()})


	#### BASIS SET FUNCTIONS
	def set_basis_set(self, basis_name):
		self.basis_set = basissets.load_basis(basis_name)


	def evaluate_basis_set(self, p):
		b = self.basis_set['elements']
		for a in self.atoms:
			z = a.atom_number
			ce = b[str(z)]['electron_shells'][-1]
			l = ce['angular_momentum']
			coeff = ce['']


	#### MOLECULE MANIPULATION
	def get_coordinates(self):
		C = np.zeros((len(self.atoms), 3))
		for i, a in enumerate(self.atoms):
			C[i] = a.position

		return C


	def set_coordinates(self, C):
		for i, a in enumerate(self.atoms):
			a.position = C[i]


	def apply_gradient(self, grad, strength=1):
		for atom, grad in grad.items():
			atom.position = atom.position - grad * strength


	def rotate(self, rotation):
		r = rotation[0]
		Rx = np.array(([	  1, 	  0,	   0],
					   [	  0, cos(r), -sin(r)],
					   [      0, sin(r),  cos(r)]))

		r = rotation[1]
		Ry = np.array(([ cos(r),  	   0, sin(r)],
					   [ 	  0, 	   1,	   0],
					   [-sin(r), 	   0, cos(r)]))

		r = rotation[2]
		Rz = np.array(([ cos(r), -sin(r), 	   0],
					   [ sin(r),  cos(r), 	   0],
					   [ 	  0, 	   0, 	   1]))

		R = Rx @ Ry @ Rz

		for a in self.atoms:
			a.position = R @ a.position


	def center_of_mass(self):
		M = sum(a.mass for a in self.atoms)
		return sum(a.mass * a.position for a in self.atoms)/M


	def center(self, p=None):
		if p is None: p = self.center_of_mass()
		p = np.asarray(p)
		for a in self.atoms:
			a.position -= p


	def get_corners(self):
		mini = np.asarray([min(a.position[0] for a in self.atoms), min(a.position[1] for a in self.atoms), min(a.position[2] for a in self.atoms)])
		maxi = np.asarray([max(a.position[0] for a in self.atoms), max(a.position[1] for a in self.atoms), max(a.position[2] for a in self.atoms)])
		return mini, maxi


	def get_dimensions(self):
		'''
		Method that returns the dimensions of a square encompassing the molecule
		'''
		#get bottom left
		mini, maxi = self.get_corners()
		return maxi - mini


	def get_center(self):
		mini, maxi = self.get_corners()
		return mini + (maxi-mini)/2


	#### UTILITY FUNCTIONS
	def get_atoms_by_element(self, element, blacklist=False):
		'''
		Method that returns a list of atoms belonging to element
		if blacklist == True, return a list of all atoms NOT belonging to element

		element - string of element symbol
		blacklist - boolean
		'''
		if blacklist:
			return list(filter(lambda a: a.element != element, self.atoms))
		return list(filter(lambda a: a.element == element, self.atoms))


	def get_atoms_by_number(self, number, blacklist=False):
		'''
		Method that returns a list of atoms belonging to element
		if blacklist == True, return a list of all atoms NOT belonging to element

		element - string of element symbol
		blacklist - boolean
		'''
		return self.get_atoms_by_element(self.number_to_element(number), blacklist=blacklist)


	def element_to_number(self, element):
		return pt.elements.symbol(element).number


	def number_to_element(self, number):
		return pt.elements[number].symbol


	def remove_non_bonded_atoms(self):
		a = self.atoms.copy()
		for atom in self.atoms:
			if len(self.bonds[atom]) == 0:
				a.remove(atom)
		self.atoms = a


	def shake(self, strength):
		for atom in self.atoms:
			atom.position = atom.position + strength * np.random.randn(3)


	def get_graph_representation(self):
		g = nx.Graph()
		g.add_nodes_from(self.atoms)
		#bonds are given as a1, a2, order so only get index 0 and 1
		g.add_edges_from([b[0:2] for b in self.unique_bonds])

		return g

	def bond_distance(self, a1, a2):
		return shortest_path_length(self.graph_representation, a1, a2)


	#### BONDING FUNCTIONS	
	def get_unique_bonds(self):
		unique_bonds = []
		prev_atoms = []
		for a1 in self.atoms:
			prev_atoms.append(a1)
			for a2, order in self.bonds[a1].items():
				if not a2 in prev_atoms:
					# prev_atoms.append(a2)
					unique_bonds.append((a1,a2,order))

		return unique_bonds


	def get_unique_bond_angles(self, in_degrees=True):
		unique_bond_angles = []
		prev_angles = []
		for a1 in self.atoms:
			for a2 in self.bonds[a1]:
				for a3 in self.bonds[a2]:
					sorted_atoms = sorted((a1,a2,a3), key=lambda x: id(x))
					if not sorted_atoms in prev_angles and len(set((a1, a2, a3))) == 3:
						prev_angles.append(sorted_atoms)
						unique_bond_angles.append((a1, a2, a3))

		return unique_bond_angles


	def bond_angle(self, a1, a2, a3, in_degrees=True):
		u = a1.position - a2.position
		v = a3.position - a2.position

		#We know that cos(theta) = u @ v / (|u| * |v|)
		#function to calculate the magnitude of a vector
		mag = lambda x: np.sqrt(x @ x)

		#return the angle. If in_degrees is True multiply it by 180/pi, else multiply by 1
		return np.arccos((u @ v) / (mag(u) * mag(v))) * (1, 180/np.pi)[in_degrees]


	def get_unique_torsion_angles(self, in_degrees=True):
		'''
		Method that yields all unique torsion angles in the molecule along with the atoms over which the torsion angle is calculated.
		'''
		unique_torsion_angles = []
		prev_angles = []
		for a1 in self.atoms:
			for a2 in self.bonds[a1]:
				for a3 in self.bonds[a2]:
					for a4 in self.bonds[a3]:
						sorted_atoms = sorted((a1,a2,a3, a4), key=lambda x: id(x))
						if not sorted_atoms in prev_angles and len(set((a1, a2, a3, a4))) == 4:
							prev_angles.append(sorted_atoms)
							unique_torsion_angles.append((a1, a2, a3, a4))

		return unique_torsion_angles


	def torsion_angle(self, a1, a2, a3, a4, in_degrees=True):
		'''
		Method that returns the torsion angle or dihedral angle of the 
		a1 -- a2 -- a3 and a2 -- a3 -- a4 planes.

		a - atom object
		in_degrees - boolean specifying whether to return angle in degrees or radians
					 set to True for degrees or False for radians

		returns float
		'''

		#using method provided on https://en.wikipedia.org/wiki/Dihedral_angle
		b1 = a2.position - a1.position
		b2 = a3.position - a2.position
		b3 = a4.position - a3.position

		return math.atan2(np.dot(np.cross(np.cross(b1, b2), np.cross(b2, b3)), b2/np.linalg.norm(b2)), np.dot(np.cross(b1, b2), np.cross(b2, b3)))


	def get_unique_atom_pairs(self, min_bond_distance=0):
		unique_pairs = []
		for i in range(len(self.atoms)):
			for j in range(i+1, len(self.atoms)):
				if min_bond_distance > 0:
					if self.bond_distance(self.atoms[i], self.atoms[j]) >= min_bond_distance:
						unique_pairs.append(tuple(sorted((self.atoms[i], self.atoms[j]), key=id)))
				elif min_bond_distance == 0:
					unique_pairs.append(tuple(sorted((self.atoms[i], self.atoms[j]), key=id)))

		return unique_pairs



	#### BOND ORDER FUNCTIONS
	def initial_bonding(self):
		bonds = {a:{} for a in self.atoms}
		for a1 in self.atoms:
			for a2 in self.atoms:
				if a1 == a2:
					continue
				if a1.distance_to(a2) < a1.covalent_radius + a2.covalent_radius + 0.4:
					bonds[a1][a2] = 1
					bonds[a2][a1] = 1
				#prevent overbonding by breaking loop when too many bonds are formed
				if len(bonds[a1]) > a1.max_valence-1:
					break

		return bonds


	def get_bond_order(self, a1, a2):
		return self.bonds[a1][a2]


	def guess_bond_orders(self, max_iters=100):
		'''
		Method that guesses the bond orders of the molecule.
		
		Current strategy:
		- Sort elements from low valence to high valence (H < O < N < C, etc..) 
		  and loops over the elements.
			- Collect every atom of the element and checks its bond saturation.
			- If the atom is not saturated, loop over the atoms it is bonded to.
				- Check the saturation of the bonded atom. If the bonded atom 
				  is also not saturated, increase the bond order to that bond.
				- Terminate the loop if the current atom is saturated.
		'''
		#function to calculate saturation:
		def unsaturated(a):
			return sum(bonds[a].values()) < a.max_valence

		def hybridisation(a):
			c = len(bonds[a])
			if a.max_valence == 1: return 0

			elif a.max_valence == 2:
				if c == 2: return 3
				elif c == 1: return 2

			elif a.max_valence == 3:
				if c == 3: return 3
				elif c == 2: return 2
				elif c == 1: return 1

			elif a.max_valence == 4:
				if c == 4: return 3
				elif c == 3: return 2
				elif c == 2: return 1

			return 0
					

		### setup
		#get the element: valence pairs from data
		valences = list(data.MAX_VALENCE.items())
		#sort them from low to high valence
		valences = sorted(valences, key=lambda x: x[1])

		bonds = self.initial_bonding()
		for atom in self.atoms:
			atom.hybridisation = hybridisation(atom)

		### algorithm
		#loop over the elements and valences 
		for num, val in valences:
			#get all atoms of element
			curr_atoms = self.get_atoms_by_number(num)
			np.random.shuffle(curr_atoms)

			#loop over the atoms
			for a1 in curr_atoms:
				#calculate saturation
				if unsaturated(a1):
					neighbours = sorted(bonds[a1].copy(), key=lambda x: a1.distance_to(x))
					np.random.shuffle(neighbours)

					#if atom is sp, it must be bound with another sp atom with bond order 3
					if hybridisation(a1) == 1:
						for a2 in neighbours:
							if hybridisation(a2) == 1:
								bonds[a1][a2] = 3
								bonds[a2][a1] = 3

					if unsaturated(a1):
						for a2 in neighbours:
							if unsaturated(a1) and unsaturated(a2):
								bonds[a1][a2] += 1
								bonds[a2][a1] += 1

		### convergence
		#check if all atoms are saturated, if not recurse the function
		if self._guess_bond_order_iters < max_iters:
			if all(not unsaturated(a) for a in self.atoms):
				return bonds

			else:
				self._guess_bond_order_iters += 1
				return self.guess_bond_orders()
		else:
			return bonds


		#### RING DETECTION
	def detect_rings(self):
		'''
		Method that detects the rings in the molecule using
		networkx module. It also detects the type of ring
		(aliphatic (AL), aromatic (AR))
		'''

		#create a graph first

		#get the cycles
		cycles = nx.algorithms.cycles.minimum_cycle_basis(self.graph_representation)

		self.rings = []
		for cycle in cycles:
			#get some data on the atoms
			atom_sym = [a.element for a in cycle]
			carbon_hybrids = [a.hybridisation for a in cycle if a.element == 'C']
			nitrogen_bonds = [len(self.bonds[a]) for a in cycle if a.element == 'N']

			#carbon contributes 1 electron, oxygen 2, nitrogen with 3 bonds 2, nitrogens with 2 bonds 1
			ne = carbon_hybrids.count(2) + atom_sym.count('O')*2 + nitrogen_bonds.count(3)*2 + nitrogen_bonds.count(2)

			#apply kekule rule and check if all carbons are sp2
			if all([h == 2 for h in carbon_hybrids]) and ne%4==2:
				self.rings.append((cycle, 'AR'))
			else:
				self.rings.append((cycle, 'AL'))

		#give the atoms ring properties
		for atom in self.atoms:
			atom.ring = None
			for cycle, typ in self.rings:
				if not atom.ring == 'AR':
					if atom in cycle:
						atom.ring = typ
					else:
						atom.ring = 'NO'

	

			
# class MoleculeCopy:
# 	def __init__(self, mol):
# 		self.name = mol.name
# 		self.positions = [a.position for a in mol.atoms]
# 		self.elements = [a.element for a in mol.atoms]
# 		self.atom_number = [a.atom_number for a in mol.atoms]

# 		self.atoms = [Atom(self.elements[i], self.positions[i]) for i in range(len(self.elements))]
# 		self.bonds = [(mol.atoms.index()) for b in mol.bonds]

# 	def __repr__(self):
# 		string = ''
# 		string += self.name + '\n'

# 		for e, p in zip(self.elements, self.positions):
# 			string += f'{e:2s}\t{p[0]: .5f}\t{p[1]: .5f}\t{p[2]: .5f}\n'

# 		return string



##### ========================================== IF MAIN ========================================== ####

if __name__ == '__main__':
	structures_folder = os.getcwd() + '\\data\\resources\\xyz\\'
	m = load_mol('benzene')
	d = display.Display()
	# d.draw_molecule(m)
else:
	structures_folder = os.getcwd() + '\\pkg\\data\\resources\\xyz\\'