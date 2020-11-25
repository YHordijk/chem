import numpy as np
import pygame as pg
import pygame.gfxdraw as pgfx
import pygame.locals as locals
from math import sin, cos
import math
# import mcubes
from time import perf_counter as pc

import pkg.colour_maps as cmap



##### ========================================== DISPLAY CLASS ========================================== ####

class Display:
	def __init__(self, size=(1280,720), camera_position=np.array([0.,0.,20.]), camera_orientation=(0.,0.,0.), project_type="perspective", bkgr_colour=(0,0,0)):
		self.camera_position = np.asarray(camera_position)
		self.camera_orientation = np.asarray(camera_orientation)
		self.project_type = project_type
		self.bkgr_colour = bkgr_colour
		self.complementary_colour = [255-x for x in bkgr_colour]
			
		self.size = self.width, self.height = size
		self.set_projection_plane()
		

	def draw_bond(self, surf, a1, a2, order, width=125, outline_width=3, draw_background=False):
		draw_line = pg.draw.line
		p1, p2 = self.atom_projections[a1], self.atom_projections[a2]
		m = (p1+p2)/2

		width = int(width/a1.distance_to(self.camera_position))

		rects = []

		if order == 1:
			if draw_background:
				rects.append(draw_line(surf, self.complementary_colour, p1, p2, width + outline_width))
			else:
				rects.append(draw_line(surf, a1.colour,p1,m, width))
				rects.append(draw_line(surf, a2.colour,m,p2, width))

		elif order == 2:
			poss = np.asarray([p1,p2])
			if not np.array_equal(p1,p2):
				perp = poss - poss[0]
				perp = np.asarray([(0,0), (perp[1][1], -perp[1][0])])[1]
				perp = width * perp / np.linalg.norm(perp)
				
				if draw_background:
					rects.append(draw_line(surf, self.complementary_colour, p1-perp, p2-perp, width + outline_width))
				else:
					rects.append(draw_line(surf, a1.colour,p1-perp,m-perp, width))
					rects.append(draw_line(surf, a2.colour,m-perp,p2-perp, width))

				if draw_background:
					rects.append(draw_line(surf, self.complementary_colour, p1+perp, p2+perp, width + outline_width))
				else:
					rects.append(draw_line(surf, a1.colour,p1+perp,m+perp, width))
					rects.append(draw_line(surf, a2.colour,m+perp,p2+perp, width))


		elif order == 3:
			poss = np.asarray([p1,p2])
			if not np.array_equal(p1,p2):
				perp = poss - poss[0]
				perp = np.asarray([perp[0], (perp[1][1], -perp[1][0])])[1]
				perp = 1.5 * width * perp / np.linalg.norm(perp)
				width = round(width/1.5)
				
				if draw_background:
					rects.append(draw_line(surf, self.complementary_colour, p1-perp, p2-perp, width + outline_width))
				else:
					rects.append(draw_line(surf, a1.colour,p1-perp,m-perp, width))
					rects.append(draw_line(surf, a2.colour,m-perp,p2-perp, width))

				if draw_background:
					rects.append(draw_line(surf, self.complementary_colour, p1, p2, width + outline_width))
				else:
					rects.append(draw_line(surf, a1.colour,p1,m, width))
					rects.append(draw_line(surf, a2.colour,m,p2, width))

				if draw_background:
					rects.append(draw_line(surf, self.complementary_colour, p1+perp, p2+perp, width + outline_width))
				else:
					rects.append(draw_line(surf, a1.colour,p1+perp,m+perp, width))
					rects.append(draw_line(surf, a2.colour,m+perp,p2+perp, width))

		elif order == 1.5:
			NotImplemented

		return rects


	def draw_atom(self, surf, a, size=300, outline_width=2):
		rad = int(a.covalent_radius/a.distance_to(self.camera_position) * size)
		p = self.atom_projections[a]
		rect = pg.draw.circle(surf, self.complementary_colour, p, rad+outline_width)
		pg.draw.circle(surf, a.colour, p, rad)
		return [rect]



	def handle_events(self, params):
			'''
			Function that handles events during running
			'''
			if params['keys'][pg.K_ESCAPE]:
				params['run'] = False


			for e in params['events']:
				if e.type == pg.VIDEORESIZE:
					self.size = self.width, self.height = e.dict['size']
					params['draw_surf'] = pg.transform.scale(params['draw_surf'], self.size)
					params['disp'] = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)
					self.set_projection_plane()

				elif e.type == pg.QUIT:
					params['run'] = False


				elif e.type == pg.MOUSEBUTTONDOWN:
					if e.button == 4:
							params['zoom'] = -params['dT'] * self.camera_position[2] * 3
					elif e.button == 5:
							params['zoom'] = params['dT'] * self.camera_position[2] * 3


			move = pg.mouse.get_rel()
			if params['keys'][pg.K_LCTRL] or params['keys'][pg.K_RCTRL]:
				if pg.mouse.get_pressed()[2]:
						self.camera_position[0] += move[0]/50
						self.camera_position[1] += move[1]/50

				if pg.mouse.get_pressed()[0]:
					params['rot'] = np.array([move[1]/150, -move[0]/150, 0])

			try:
				[m.rotate(params['rot']) for m in params['mols']]
			except:
				params['mol'].rotate(params['rot'])

			

			self.camera_position[2] += params['zoom']
			if self.camera_position[2] > 40: self.camera_position[2] = 40
			if self.camera_position[2] < 3: self.camera_position[2] = 3
			
			return params


	def look_at(self, p):
		''' 
		Method that orients the camera to look at a point p
		'''

		P = p - self.camera_position
		self.camera_orientation[0] = math.atan2(P[1],P[2])
		self.camera_orientation[1] = math.atan2(P[0],P[2])


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


	def project_array(self, array):
		#accepts array in the form:
		'''
		array = [x0, y0, z0]
				[x1, y1, z1]
					...
				[xn, yn, zn]
		'''

		d = self.rotation_matrix @ (array - self.camera_position).T
		f = self.projection_plane @ d

		return np.vstack((f[0]/f[2], f[1]/f[2])).T.astype(int)


	def pre_update(self, params):
		self.set_rotation_matrix()
		self.atom_projections = {a:self.project(a.position) for a in params['mol'].atoms}
		params['keys'] = pg.key.get_pressed()
		params['events'] = pg.event.get()
		params = self.handle_events(params)

		return params


	def post_update(self, params):
		params['zoom'] *= 0.8
		params['rot'] = params['rot'] * 0.8


	def update(self, params):
		pass


	def draw_3dpoints(self, params, P, array, colour_map=cmap.BlueRed()):
		projected_P = self.project_array(P)
		surf = params['draw_surf']
		[pg.draw.circle(surf, ((0,0,255), (255,0,0))[float(a)>0], p, min(r*2,8)) for a, p, r in zip(array, projected_P, params['R']) if r>0]


	def draw_mesh(self, params, mesh, fill=True):
		vert, tria = mesh.vertices, mesh.triangles
		vert_p = self.project_array(vert)

		surf = params['draw_surf']

		dist = lambda x: np.linalg.norm(self.camera_position - vert[x[0]])
		if fill:
			for i, t in enumerate(sorted(tria, key=dist, reverse=True)):
				# d = dist(t)
				c = (255,255,255)
				try:
					pgfx.trigon(surf, *vert_p[t[0]], *vert_p[t[1]], *vert_p[t[2]], c)
				except:
					pass



	def draw_molecule(self, mol, draw_hydrogens=True, draw_atoms=True, draw_bonds=True):
		'''
		Method that draws and displays the provided molecule
		'''

		params = {}
		params['draw_surf'] = pg.surface.Surface(self.size)

		clock = pg.time.Clock()
		tick = clock.tick_busy_loop

		#set parameters for the screen
		params['run'] = True
		params['FPS'] = 500
		params['updt'] = 0
		params['time'] = 0
		params['zoom'] = 0
		params['move'] = (0,0)
		params['rot'] = np.array([0.,0.,0.])
		params['mol'] = mol
		params['draw_hydrogens'] = draw_hydrogens
		params['draw_atoms'] = draw_atoms
		params['draw_bonds'] = draw_bonds
		params['disp'] = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)

		cam_dist = lambda a: np.linalg.norm(a.position - self.camera_position)

		atoms = mol.atoms
		bonds = mol.bonds

		#update loop
		while params['run']:
			params['updt'] += 1
			params['dT'] = tick(params['FPS'])/1000
			params['time'] += params['dT']
			params['draw_surf'].fill(self.bkgr_colour)

			params['rects'] = []
			self.pre_update(params)
			self.update(params)


			#sort atoms by distance to camera and invert
			sorted_atoms = sorted([(a, cam_dist(a)) for a in atoms], key=lambda x: x[1], reverse=True)

			#### drawing of molecule

			for a, d in sorted_atoms:
				#draw furthest atoms first, then bonds to neighbouring atoms
				if params['draw_hydrogens']:
					if params['draw_bonds']:
						for b, order in bonds[a].items():
							if d > cam_dist(b):
								params['rects'] += self.draw_bond(params['draw_surf'], a, b, order, draw_background=True)
					if params['draw_atoms']:
						params['rects'] += self.draw_atom(params['draw_surf'], a)
					if params['draw_bonds']:
						for b, order in bonds[a].items():
							if d > cam_dist(b):
								params['rects'] += self.draw_bond(params['draw_surf'], a, b, order, draw_background=False)

				else:
					if a.element == 'H':
						continue
					for b, order in bonds[a].items():
						if d > cam_dist(b) and not b.element == 'H':
							params['rects'] += self.draw_bond(draw_surf, a, b, order)
					if params['draw_atoms']:
						params['rects'] += self.draw_atom(draw_surf, a)
			
			params['disp'].blit(params['draw_surf'], (0,0))
			self.post_update(params)
			# pg.display.update(params['rects'])
			pg.display.update()

			# print('Total  time (s):', params['dT'])


	def draw_molecule_animation(self, mols, animation_speed=4, draw_hydrogens=True, draw_atoms=True, draw_bonds=True):
		'''
		Method that draws and displays the provided molecule
		'''
		params = {}
		params['draw_surf'] = pg.surface.Surface(self.size)



		clock = pg.time.Clock()
		tick = clock.tick_busy_loop

		#set parameters for the screen
		params['run'] = True
		params['FPS'] = 120
		params['updt'] = 0
		params['time'] = 0
		params['zoom'] = 0
		params['move'] = (0,0)
		params['rot'] = np.array([0,0,0])
		params['mols'] = mols
		params['draw_hydrogens'] = draw_hydrogens
		params['draw_atoms'] = draw_atoms
		params['draw_bonds'] = draw_bonds
		params['disp'] = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)

		cam_dist = lambda a: np.linalg.norm(a.position - self.camera_position)

		atoms_collection = [mol.atoms for mol in mols]
		bonds_collection = [mol.bonds for mol in mols]

		
		params['curr_mol_idx'] = 0
		mol = mols[0]
		params['mol'] = mol

		#update loop
		while params['run']:
			params['updt'] += 1
			params['dT'] = tick(params['FPS'])/1000
			params['time'] += params['dT']

			if params['updt']%animation_speed == 0:
				params['curr_mol_idx'] += 1
				params['curr_mol_idx'] = params['curr_mol_idx']%len(mols)
				mol = mols[params['curr_mol_idx']]
				params['mol'] = mol

			self.pre_update(params)
			self.update(params)

			#clear screen
			params['draw_surf'].fill(self.bkgr_colour)

			atoms = atoms_collection[params['curr_mol_idx']]
			bonds = bonds_collection[params['curr_mol_idx']]

			#sort atoms by distance to camera and invert
			sorted_atoms = sorted([(a, cam_dist(a)) for a in atoms], key=lambda x: x[1], reverse=True)

			#### drawing of molecule
			for a, d in sorted_atoms:
				#draw furthest atoms first, then bonds to neighbouring atoms
				if params['draw_hydrogens']:
					for b, order in bonds[a].items():
						if d > cam_dist(b):
							self.draw_bond(params['draw_surf'], a, b, order)
					if params['draw_atoms']:
						self.draw_atom(params['draw_surf'], a)

				else:
					if a.element == 'H':
						continue
					for b, order in bonds[a].items():
						if d > cam_dist(b) and not b.element == 'H':
							self.draw_bond(draw_surf, a, b, order)
					if params['draw_atoms']:
						self.draw_atom(draw_surf, a)


			params['disp'].blit(params['draw_surf'], (0,0))
			self.post_update(params)
			pg.display.update()
