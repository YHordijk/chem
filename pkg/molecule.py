import numpy as np 
import pubchempy as pcp
import os
import periodictable as pt
import pygame as pg
from math import sin, cos
import math

try:
	import data
	import screen3D
except:
	import pkg.data as data
	import pkg.screen3D



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
	if get_from_pubchem(name) is None:
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
		self.set_mass()
		self.set_covalent_radius()
		self.set_colour()


	def __repr__(self):
		return f'{self.element}({self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f})'


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
			raise
			self.max_valence = 1

	def set_mass(self):
		self.mass = pt.elements[self.atom_number].mass


	def set_covalent_radius(self):
		self.covalent_radius = pt.elements[self.atom_number].covalent_radius


	def set_colour(self):
		c = data.ATOM_COLOURS[self.atom_number]
		self.colour = tuple([int(i) for i in c])


class Molecule:
	def __init__(self, name=[], elements=[], positions=[], charges=[], atoms=[]):
		self.name = name

		assert(len(elements) == len(positions))
		self.elements = elements
		self.positions = positions

		self.charges = charges

		#get atoms from elements, positions
		self.atoms = [Atom(elements[i], positions[i]) for i in range(len(self.elements))]
		#If a list of atoms is already given, append it to the list just produced
		self.atoms += atoms

		self._guess_bond_order_iters = 0
		self.bonds = self.guess_bond_orders()

		# [print(b) for b in self.bonds.items()]


	def __repr__(self):
		string = ''
		string += self.name + '\n'

		for e, p in zip(self.elements, self.positions):
			string += f'{e:2s}\t{p[0]: .5f}\t{p[1]: .5f}\t{p[2]: .5f}\n'

		return string


	#### MOLECULE MANIPULATION
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
					

		### setup
		#get the element: valence pairs from data
		valences = list(data.MAX_VALENCE.items())
		#sort them from low to high valence
		valences = sorted(valences, key=lambda x: x[1])

		bonds = self.initial_bonding()

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
		if not all(not unsaturated(a) for a in self.atoms) \
				and self._guess_bond_order_iters < max_iters:

			self._guess_bond_order_iters += 1
			self.guess_bond_orders()

		return bonds
			

##### ========================================== DISPLAY CLASS ========================================== ####

class Display(screen3D.Screen3D):
	def __init__(self, size=(1280,720), camera_position=np.array([0.,0.,20.]), camera_orientation=(0.,0.,0.), project_type="perspective", bkgr_colour=(0,0,0)):
		self.camera_position = np.asarray(camera_position)
		self.camera_orientation = np.asarray(camera_orientation)
		self.project_type = project_type
		self.bkgr_colour = bkgr_colour
			
		self._size = self.width, self.height = size
		self.set_projection_plane()
		

	def draw_bond(self, surf, a1, a2, order, width=3, outline_width=2):
		draw_line = pg.draw.line
		p1, p2 = self.atom_projections[a1], self.atom_projections[a2]

		if order == 1:
			draw_line(surf, (255,255,255),p1,p2, width)

		if order == 2:
			d = width
			poss = np.asarray([p1,p2])
			perp = poss - poss[0]
			perp = np.asarray([perp[0], (perp[1][1], -perp[1][0])])[1]
			perp = d * perp / np.linalg.norm(perp)

			draw_line(surf, (255,255,255), p1-perp, p2-perp, width)
			draw_line(surf, (255,255,255), p1+perp, p2+perp, width)



	def draw_atom(self, surf, a, size=300, outline_width=2):
		rad = int(a.covalent_radius/a.distance_to(self.camera_position) * size)
		p = self.atom_projections[a]
		pg.draw.circle(surf, (255,255,255), p, rad+outline_width)
		pg.draw.circle(surf, a.colour, p, rad)



	def handle_events(self, events, screen_params):
			'''
			Function that handles events during running
			'''
			for e in events:
				if e.type == pg.VIDEORESIZE:
					self.size = e.dict['size']
					self.set_projection_plane()

				if e.type == pg.QUIT:
					screen_params['run'] = False

			return screen_params


	def handle_keys(self, keys, screen_params):
			'''
			Function that handles key states during running
			'''
			if keys[pg.K_ESCAPE]:
				screen_params['run'] = False

			return screen_params


	def set_rotation_matrix(self):
		t = self.camera_orientation

		x_rot_mat = np.array([[1,0,0], [0, cos(t[0]), sin(t[0])], [0, -sin(t[0]), cos(t[0])]])
		y_rot_mat = np.array([[cos(t[1]), 0, -sin(t[1])], [0,1,0], [sin(t[1]), 0, cos(t[1])]])
		z_rot_mat = np.array([[cos(t[2]), sin(t[2]), 0], [-sin(t[2]), cos(t[2]), 0], [0,0,1]])
		z_rot_mat = np.array([[1,0,0], [0,1,0], [0,0,1]])

		self.rotation_matrix = x_rot_mat @ y_rot_mat @ z_rot_mat


	def set_projection_plane(self):
		e = np.array([self.width/2, self.height/2, 600])
		self.projection_plane = np.array([[1, 0, e[0]/e[2]], [0, 1, e[1]/e[2]], [0, 0, 1/e[2]]])


	def project(self, p):
		d = self.rotation_matrix @ (p - self.camera_position).T
		f = self.projection_plane @ d
		return int(round(f[0]/f[2])), int(round(f[1]/f[2]))


	def pre_update(self, screen_params, mol):
		self.set_rotation_matrix()
		self.atom_projections = {a:self.project(a.position) for a in mol.atoms}
		screen_params = self.handle_keys(pg.key.get_pressed(), screen_params)
		screen_params = self.handle_events(pg.event.get(), screen_params)

		return screen_params


	def draw_molecule(self, mol):
		'''
		Method that draws and displays the provided molecule
		'''
		disp = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)
		draw_surf = pg.surface.Surface(self.size)

		clock = pg.time.Clock()
		tick = clock.tick_busy_loop

		#set parameters for the screen
		screen_params = {}
		screen_params['run'] = True
		screen_params['FPS'] = 500
		screen_params['updt'] = 0
		screen_params['time'] = 0

		cam_dist = lambda a: np.linalg.norm(a.position - self.camera_position)

		atoms = mol.atoms
		bonds = mol.bonds
		# print(bonds)

		#update loop
		while screen_params['run']:
			screen_params['updt'] += 1
			dT = tick(screen_params['FPS'])/1000
			screen_params['time'] += dT

			self.pre_update(screen_params, mol)

			#clear screen
			draw_surf.fill(self.bkgr_colour)

			mol.rotate((0,.01,0))

			#sort atoms by distance to camera and invert
			sorted_atoms = sorted([(a, cam_dist(a)) for a in atoms], key=lambda x: x[1], reverse=True)

			#### drawing of molecule
			for a, d in sorted_atoms:
				#draw furthest atoms first, then bonds to neighbouring atoms
				self.draw_atom(draw_surf, a)
				for b, order in bonds[a].items():
					if d > cam_dist(b):
						self.draw_bond(draw_surf, a, b, order)


			disp.blit(draw_surf, (0,0))
			pg.display.update()







##### ========================================== IF MAIN ========================================== ####

if __name__ == '__main__':
	structures_folder = os.getcwd() + '\\data\\resources\\xyz\\'
	m = load_mol('hexabenzocoronene')
	d = Display()
	d.draw_molecule(m)
else:
	structures_folder = os.getcwd() + '\\pkg\\data\\resources\\xyz\\'