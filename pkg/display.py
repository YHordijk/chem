import numpy as np
import pygame as pg
import pygame.locals as locals
from math import sin, cos
import math



##### ========================================== DISPLAY CLASS ========================================== ####

class Display:
	def __init__(self, size=(1280,720), camera_position=np.array([0.,0.,20.]), camera_orientation=(0.,0.,0.), project_type="perspective", bkgr_colour=(0,0,0)):
		self.camera_position = np.asarray(camera_position)
		self.camera_orientation = np.asarray(camera_orientation)
		self.project_type = project_type
		self.bkgr_colour = bkgr_colour
			
		self.size = self.width, self.height = size
		self.set_projection_plane()
		

	def draw_bond(self, surf, a1, a2, order, width=125, outline_width=3):
		draw_line = pg.draw.line
		p1, p2 = self.atom_projections[a1], self.atom_projections[a2]
		m = (p1+p2)/2

		width = int(width/a1.distance_to(self.camera_position))

		if order == 1:
			draw_line(surf, self.bkgr_colour, p1, p2, width + outline_width)

			draw_line(surf, a1.colour,p1,m, width)
			draw_line(surf, a2.colour,m,p2, width)

		elif order == 2:
			poss = np.asarray([p1,p2])
			if not np.array_equal(p1,p2):
				perp = poss - poss[0]
				perp = np.asarray([(0,0), (perp[1][1], -perp[1][0])])[1]
				perp = width * perp / np.linalg.norm(perp)

				draw_line(surf, self.bkgr_colour, p1-perp, p2-perp, width + outline_width)
				draw_line(surf, a1.colour,p1-perp,m-perp, width)
				draw_line(surf, a2.colour,m-perp,p2-perp, width)

				draw_line(surf, self.bkgr_colour, p1+perp, p2+perp, width + outline_width)
				draw_line(surf, a1.colour,p1+perp,m+perp, width)
				draw_line(surf, a2.colour,m+perp,p2+perp, width)


		elif order == 3:
			poss = np.asarray([p1,p2])
			if not np.array_equal(p1,p2):
				perp = poss - poss[0]
				perp = np.asarray([perp[0], (perp[1][1], -perp[1][0])])[1]
				perp = 1.5 * width * perp / np.linalg.norm(perp)

				draw_line(surf, self.bkgr_colour, p1-perp, p2-perp, width + outline_width)
				draw_line(surf, a1.colour,p1-perp,m-perp, width)
				draw_line(surf, a2.colour,m-perp,p2-perp, width)

				draw_line(surf, self.bkgr_colour, p1, p2, width + outline_width)
				draw_line(surf, a1.colour,p1,m, width)
				draw_line(surf, a2.colour,m,p2, width)

				draw_line(surf, self.bkgr_colour, p1+perp, p2+perp, width + outline_width)
				draw_line(surf, a1.colour,p1+perp,m+perp, width)
				draw_line(surf, a2.colour,m+perp,p2+perp, width)

		elif order == 1.5:
			NotImplemented



	def draw_atom(self, surf, a, size=300, outline_width=2):
		rad = int(a.covalent_radius/a.distance_to(self.camera_position) * size)
		p = self.atom_projections[a]
		pg.draw.circle(surf, self.bkgr_colour, p, rad+outline_width)
		pg.draw.circle(surf, a.colour, p, rad)



	def handle_events(self, events, keys, screen_params):
			'''
			Function that handles events during running
			'''
			if keys[pg.K_ESCAPE]:
				screen_params['run'] = False

			for e in events:
				if e.type == pg.VIDEORESIZE:
					self.size = e.dict['size']
					self.draw_surf = pg.transform.scale(self.draw_surf, self.size)
					self.set_projection_plane()

				elif e.type == pg.QUIT:
					screen_params['run'] = False


				elif e.type == pg.MOUSEBUTTONDOWN:
					if e.button == 4:
							screen_params['zoom'] = -screen_params['dT'] * self.camera_position[2] * 3
					elif e.button == 5:
							screen_params['zoom'] = screen_params['dT'] * self.camera_position[2] * 3


			move = pg.mouse.get_rel()
			if keys[pg.K_LCTRL] or keys[pg.K_RCTRL]:
				if pg.mouse.get_pressed()[2]:
						self.camera_position[0] += move[0]/50
						self.camera_position[1] += move[1]/50

				if pg.mouse.get_pressed()[0]:
					screen_params['rot'] = np.array([move[1]/150, -move[0]/150, 0])

			[m.rotate(screen_params['rot']) for m in screen_params['mols']]
			screen_params['rot'] = screen_params['rot'] * 0.8


			self.camera_position[2] += screen_params['zoom']
			if self.camera_position[2] > 40: self.camera_position[2] = 40
			if self.camera_position[2] < 3: self.camera_position[2] = 3
			screen_params['zoom'] *= 0.8

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
		return np.asarray((int(round(f[0]/f[2])), int(round(f[1]/f[2]))))


	def pre_update(self, screen_params, mol):
		self.set_rotation_matrix()
		self.atom_projections = {a:self.project(a.position) for a in mol.atoms}
		screen_params = self.handle_events(pg.event.get(), pg.key.get_pressed(), screen_params)

		return screen_params


	def draw_molecule(self, mol, draw_hydrogens=True, draw_atoms=True):
		'''
		Method that draws and displays the provided molecule
		'''
		disp = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)
		self.draw_surf = pg.surface.Surface(self.size)

		clock = pg.time.Clock()
		tick = clock.tick_busy_loop

		#set parameters for the screen
		screen_params = {}
		screen_params['run'] = True
		screen_params['FPS'] = 500
		screen_params['updt'] = 0
		screen_params['time'] = 0
		screen_params['zoom'] = 0
		screen_params['move'] = (0,0)
		screen_params['rot'] = np.array([0,0,0])
		screen_params['mol'] = mol
		screen_params['draw_hydrogens'] = draw_hydrogens
		screen_params['draw_atoms'] = draw_atoms

		cam_dist = lambda a: np.linalg.norm(a.position - self.camera_position)

		atoms = mol.atoms
		bonds = mol.bonds

		#update loop
		while screen_params['run']:
			screen_params['updt'] += 1
			screen_params['dT'] = tick(screen_params['FPS'])/1000
			screen_params['time'] += screen_params['dT']

			self.pre_update(screen_params, mol)

			#clear screen
			self.draw_surf.fill(self.bkgr_colour)

			# mol.rotate((0,.01,0))
			# self.camera_position = np.array([0,0, 20+10*math.sin(screen_params['time'])])

			#sort atoms by distance to camera and invert
			sorted_atoms = sorted([(a, cam_dist(a)) for a in atoms], key=lambda x: x[1], reverse=True)

			#### drawing of molecule

			for a, d in sorted_atoms:
				#draw furthest atoms first, then bonds to neighbouring atoms
				if screen_params['draw_hydrogens']:
					for b, order in bonds[a].items():
						if d > cam_dist(b):
							self.draw_bond(self.draw_surf, a, b, order)
					if screen_params['draw_atoms']:
						self.draw_atom(self.draw_surf, a)

				else:
					if a.element == 'H':
						continue
					for b, order in bonds[a].items():
						if d > cam_dist(b) and not b.element == 'H':
							self.draw_bond(draw_surf, a, b, order)
					if screen_params['draw_atoms']:
						self.draw_atom(draw_surf, a)

			disp.blit(self.draw_surf, (0,0))
			pg.display.update()


	def draw_molecule_animation(self, mols, animation_speed=4, draw_hydrogens=True, draw_atoms=True):
		'''
		Method that draws and displays the provided molecule
		'''
		disp = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)
		self.draw_surf = pg.surface.Surface(self.size)

		clock = pg.time.Clock()
		tick = clock.tick_busy_loop

		#set parameters for the screen
		screen_params = {}
		screen_params['run'] = True
		screen_params['FPS'] = 120
		screen_params['updt'] = 0
		screen_params['time'] = 0
		screen_params['zoom'] = 0
		screen_params['move'] = (0,0)
		screen_params['rot'] = np.array([0,0,0])
		screen_params['mols'] = mols
		screen_params['draw_hydrogens'] = draw_hydrogens
		screen_params['draw_atoms'] = draw_atoms

		cam_dist = lambda a: np.linalg.norm(a.position - self.camera_position)

		atoms_collection = [mol.atoms for mol in mols]
		bonds_collection = [mol.bonds for mol in mols]

		
		curr_mol_index = 0
		mol = mols[0]
		screen_params['mol'] = mol

		#update loop
		while screen_params['run']:
			screen_params['updt'] += 1
			screen_params['dT'] = tick(screen_params['FPS'])/1000
			screen_params['time'] += screen_params['dT']

			if screen_params['updt']%animation_speed == 0:
				curr_mol_index += 1
				curr_mol_index = curr_mol_index%len(mols)
				mol = mols[curr_mol_index]
				screen_params['mol'] = mol

			self.pre_update(screen_params, mol)

			#clear screen
			self.draw_surf.fill(self.bkgr_colour)



			atoms = atoms_collection[curr_mol_index]
			bonds = bonds_collection[curr_mol_index]

			#sort atoms by distance to camera and invert
			sorted_atoms = sorted([(a, cam_dist(a)) for a in atoms], key=lambda x: x[1], reverse=True)

			#### drawing of molecule
			for a, d in sorted_atoms:
				#draw furthest atoms first, then bonds to neighbouring atoms
				if screen_params['draw_hydrogens']:
					for b, order in bonds[a].items():
						if d > cam_dist(b):
							self.draw_bond(self.draw_surf, a, b, order)
					if screen_params['draw_atoms']:
						self.draw_atom(self.draw_surf, a)

				else:
					if a.element == 'H':
						continue
					for b, order in bonds[a].items():
						if d > cam_dist(b) and not b.element == 'H':
							self.draw_bond(draw_surf, a, b, order)
					if screen_params['draw_atoms']:
						self.draw_atom(draw_surf, a)


			disp.blit(self.draw_surf, (0,0))
			pg.display.update()
