import numpy as np 
import pubchempy as pcp
import os
import periodictable as pt

if __name__ == '__main__':
	import data
else:
	import pkg.data as data



##### ========================================== FUNCTIONS FOR LOADING AND SAVING MOLECULES ========================================== ####

def load_from_file(file):
	name = file.split('/')[-1].strip('.xyz').capitalize()
	name = name.split('\\')[-1].strip('.xyz').capitalize()

	elements = list(np.loadtxt(file, skiprows=2, usecols=0, dtype=str))
	positions = list(np.loadtxt(file, skiprows=2, usecols=(1,2,3), dtype=float))
		

	return Molecule(name, elements, positions)


def get_from_pubchem(name):
	try:
		name = int(name)
	except:
		pass

	mol = pcp.get_compounds(name, ('name', 'cid')[type(name) is int], record_type='3d')
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


def find_mol(name, root=None, exact=True):
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
	get_from_pubchem(name)

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


def load_mol(name):
	path = find_mol(name.capitalize())
	mol = load_from_file(path)
	return mol




##### ========================================== ATOM AND MOLECULE CLASS ========================================== ####

class Atom:
	def __init__(self, element=None, position=None, charge=None):
		self.element = element
		self.atom_number = pt.elements.symbol(element).number
		self.position = position
		self.charge = charge

		self.set_max_valence()
		self.set_covalent_radius()


	def distance_to(self, p):
		if type(p) is Atom:
			return np.linalg.norm(self.position - p.position)
		return np.linalg.norm(self.position - p)


	def set_max_valence(self):
		#retrieve from data files
		try:
			self.max_valence = int(data.MAX_VALENCE[self.atom_number])
		#default to 1
		except:
			self.max_valence = 1


	def set_covalent_radius(self):
		self.covalent_radius = pt.elements[self.atom_number].covalent_radius


	def __repr__(self):
		return f'{self.element}({self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f})'



class Molecule:
	def __init__(self, name=[], elements=[], positions=[], charges=[], atoms=[]):
		self.name = name
		self.elements = elements
		self.positions = positions
		self.charges = charges


		assert(len(elements) == len(positions))

		#get atoms from elements, positions
		self.atoms = [Atom(elements[i], positions[i]) for i in range(len(self.elements))]
		#If a list of atoms is already given, append it to the list just produced
		self.atoms += atoms

		self.guess_bond_orders()


	#### UTILITY FUNCTIONS
	def get_atoms_by_element(self, element, blacklist=False):
		'''
		Method that returns a list of atoms belonging to element
		if blacklist == True, return a list of all atoms NOT belonging to element

		element - string of element symbol
		blacklist - boolean
		'''
		if blacklist:
			return filter(lambda a: a.element != element, self.atoms)
		return filter(lambda a: a.element == element, self.atoms)


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


	#### BOND ORDER FUNCTIONS
	def initial_bonding(self):
		bonds = {a:[] for a in self.atoms}

		for a1 in self.atoms:
			for a2 in self.atoms:
				if a1 == a2:
					continue
				if a1.distance_to(a2) < a1.covalent_radius + a2.covalent_radius + 0.4:
					bonds[a1].append((a1,1))

		return bonds 


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
			return sum([b[1] for b in bonds[a]]) < a.max_valence

		#get the element: valence pairs from data
		valences = list(data.MAX_VALENCE.items())
		#sort them from low to high valence
		valences = sorted(valences, key=lambda x: x[1])

		
		bonds = self.initial_bonding()

		[print(b) for b in bonds.values()]

		#loop over the elements and valences 
		for num, val in valences:
			#get all atoms of element
			curr_atoms = self.get_atoms_by_number(num)

			#loop over the atoms
			for a in curr_atoms:
				#calculate saturation
				if unsaturated(a):
					...

				
			

	def __repr__(self):
		string = ''
		string += self.name + '\n'

		for e, p in zip(self.elements, self.positions):
			string += f'{e:2s}\t{p[0]: .5f}\t{p[1]: .5f}\t{p[2]: .5f}\n'

		return string



##### ========================================== IF MAIN ========================================== ####

if __name__ == '__main__':
	structures_folder = os.getcwd() + '\\data\\resources\\xyz\\'
	m = load_mol('aspirin')
	# [print(a) for a in m.atoms]
else:
	structures_folder = os.getcwd() + '\\pkg\\data\\resources\\xyz\\'