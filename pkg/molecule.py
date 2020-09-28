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
		self.position = position
		self.charge = charge


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




	def get_bonds(self):
		pass
			

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
	[print(a) for a in m.atoms]
else:
	structures_folder = os.getcwd() + '\\pkg\\data\\resources\\xyz\\'