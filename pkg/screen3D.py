import pygame as pg
import pygame.locals as pglc
import numpy as np
from math import cos, sin, sqrt
import math, time
from scipy.spatial.distance import euclidean
from time import perf_counter


try:
	import colour_maps as cmap
except:
	import pkg.colour_maps as cmap






class Screen3D:
	def __init__(self, size, camera_position=np.array([0.,0.,30.]), camera_orientation=(0.,0.,0.), project_type="perspective", bkgr_colour=(0,0,0)):
		self.camera_position = np.asarray(camera_position)
		self.camera_orientation = np.asarray(camera_orientation)
		self.project_type = project_type
		self.bkgr_colour = bkgr_colour
		
		self._size = self.width, self.height = size
		self.disp = pg.display.set_mode(self.size, pglc.HWSURFACE|pglc.DOUBLEBUF|pglc.RESIZABLE)
		self.disp.fill(self.bkgr_colour)

		self._dens_pos = {}
		self._dens_colours = {}


	@property
	def size(self):
		return self._size

	@size.setter
	def size(self, val):
		self._size = val
		self.disp = pg.display.set_mode(val, pglc.HWSURFACE|pglc.DOUBLEBUF|pglc.RESIZABLE)


	def follow(self, target, offset=0):
		delta = (target.position - self.camera_position + offset)/4
		self.camera_position += delta


	def display_text(self, text, pos):
		f = pg.font.Font(pg.font.get_default_font(), 20)
		surf = f.render(text, True, (255,255,255))
		self.disp.blit(surf, pos)


	def project(self, coord):
		a = coord.T
		c = self.camera_position.T
		t = self.camera_orientation
		e = np.array([self.width/2, self.height/2, 600])

		x_rot_mat = np.array([[1,0,0], [0, cos(t[0]), sin(t[0])], [0, -sin(t[0]), cos(t[0])]])
		y_rot_mat = np.array([[cos(t[1]), 0, -sin(t[1])], [0,1,0], [sin(t[1]), 0, cos(t[1])]])
		z_rot_mat = np.array([[cos(t[2]), sin(t[2]), 0], [-sin(t[2]), cos(t[2]), 0], [0,0,1]])
		z_rot_mat = np.array([[1,0,0], [0,1,0], [0,0,1]])

		d = x_rot_mat @ y_rot_mat @ z_rot_mat @ (a - c)

		f = np.array([[1, 0, e[0]/e[2]], [0, 1, e[1]/e[2]], [0, 0, 1/e[2]]]) @ d
		return int(round(f[0]/f[2])), int(round(f[1]/f[2]))


	def project_array(self, array):
		#accepts array in the form:
		'''
		array = [x0, y0, z0]
				[x1, y1, z1]
					...
				[xn, yn, zn]
		'''

		a = array
		c = self.camera_position.T
		t = self.camera_orientation
		e = np.array([self.width/2, self.height/2, 600])

		x_rot_mat = np.array([[1,0,0], [0, cos(t[0]), sin(t[0])], [0, -sin(t[0]), cos(t[0])]])
		y_rot_mat = np.array([[cos(t[1]), 0, -sin(t[1])], [0,1,0], [sin(t[1]), 0, cos(t[1])]])
		z_rot_mat = np.array([[cos(t[2]), sin(t[2]), 0], [-sin(t[2]), cos(t[2]), 0], [0,0,1]])
		z_rot_mat = np.array([[1,0,0], [0,1,0], [0,0,1]])


		d = x_rot_mat @ y_rot_mat @ z_rot_mat @ (a - c).T

		f = np.array([[1, 0, e[0]/e[2]], [0, 1, e[1]/e[2]], [0, 0, 1/e[2]]]) @ d
		return np.vstack((f[0]/f[2], f[1]/f[2])).T.astype(int)

	def draw_pixel(self, pos, colour=(255,255,255)):
		try:
			pos = self.project(pos)
			self.disp.set_at(pos, colour)
		except Exception as e:
			pass
	
	def draw_pixels(self, poss, colour=(255,255,255), colour_func=None, colour_array=None):
		set_at = self.disp.set_at
		if type(colour_array) is np.ndarray:
			if type(poss) is np.ndarray:
				poss = self.project_array(poss)
				[set_at(pos, clr) for pos, clr in zip(poss, colour_array)]
			else:
				draw = self.draw_pixel
				[draw(pos, colour_func(pos)) for pos in poss]

		elif colour_func != None:
			if type(poss) is np.ndarray:
				poss = self.project_array(poss)
				[self.disp.set_at(pos, colour_func(pos)) for pos in poss]
			else:
				draw = self.draw_pixel
				[draw(pos, colour_func(pos)) for pos in poss]

		else:
			if type(poss) is np.ndarray:
				poss = self.project_array(poss)
				[set_at(pos, colour) for pos in poss]
			else:
				draw = self.draw_pixel
				[draw(pos, colour_func(pos)) for pos in poss]


	def draw_line(self, poss, colour=(255,255,255)):
		try:
			poss = [self.project(pos) for pos in poss]
			pg.draw.aaline(self.disp, colour, poss[0], poss[1])
		except:
			raise


	def draw_lines(self, poss, colour=(255,255,255), closed=True, width=1):
		try:
			proj = self.project
			poss = [proj(pos) for pos in poss]
			pg.draw.aalines(self.disp, colour, closed, poss, width)

		except:
			raise


	def draw_single_bond(self, poss, colour=(255,255,255), width=1):
		try:
			h = self.height + 200
			w = self.width + 200
			poss = [self.project(pos) for pos in poss]
			if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
				pg.draw.line(self.disp, self.contrast_colour, poss[0], poss[1], width+3)
				pg.draw.line(self.disp, colour, poss[0], poss[1], width)
		except:
			raise


	def draw_double_bond(self, poss, colour=(255,255,255), width=1):
		try:
			h = self.height + 200
			w = self.width + 200
			poss = np.asarray([self.project(pos) for pos in poss])
			if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
				d = width
				perp = poss - poss[0]
				
				perp = np.asarray([perp[0], (perp[1][1], -perp[1][0])])[1]
				perp = d * perp / np.linalg.norm(perp)

				pg.draw.line(self.disp, self.contrast_colour, poss[0]-perp, poss[1]-perp, d+3)
				pg.draw.line(self.disp, self.contrast_colour, poss[0]+perp, poss[1]+perp, width+3)

				pg.draw.line(self.disp, colour, poss[0]-perp, poss[1]-perp, d)
				pg.draw.line(self.disp, colour, poss[0]+perp, poss[1]+perp, d)
		except:
			pass

	def draw_triple_bond(self, poss, colour=(255,255,255), width=1):
		try:
			h = self.height + 200
			w = self.width + 200
			poss = np.asarray([self.project(pos) for pos in poss])
			if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
				d = width
				perp = poss - poss[0]
				
				perp = np.asarray([perp[0], (perp[1][1], -perp[1][0])])[1]
				perp = 1.5 * d * perp / np.linalg.norm(perp)

				pg.draw.line(self.disp, self.contrast_colour, poss[0]-perp, poss[1]-perp, d+3)
				pg.draw.line(self.disp, self.contrast_colour, poss[0], poss[1], d+3)
				pg.draw.line(self.disp, self.contrast_colour, poss[0]+perp, poss[1]+perp, width+3)

				pg.draw.line(self.disp, colour, poss[0]-perp, poss[1]-perp, d)
				pg.draw.line(self.disp, colour, poss[0], poss[1], d)
				pg.draw.line(self.disp, colour, poss[0]+perp, poss[1]+perp, d)
		except:
			pass


	def draw_circle(self, center, radius, colour=(255,255,255), width=0):
		pos = self.project(np.asarray(center))
		pg.draw.circle(self.disp, colour, pos, radius, width)


	def draw_polygon(self, points, colour=(255,255,255)):
		try:
			proj = self.project
			dist = int((self.camera_position[2] - points[0][1])*2)
			points = [proj(point) for point in points]
			colour = (255-dist,255-dist,255-dist)
			pg.draw.polygon(self.disp, colour, points)
		except:
			raise


	def update(self):
		pg.display.update()


	def get_atom_at_pos(self, pos):
		pos = np.asarray(pos)
		dists = sorted([(euclidean(pos, p), a) for a, p in self.atom_draw_pos.items()], key=lambda x: x[0])
		atom = None
		for dist, a in dists:
			if dist < self.atom_draw_rad[a]:
				try:
					if a.distance_to(self.camera_position) < atom.distance_to(self.camera_position):
						atom = a
				except:
					atom = a

		return atom


	def pre_render_densities(self, orbitals, points=50000, colour_map=cmap.BlueRed(posneg_mode=True)):
		utils.message(f'Pre-rendering {len(orbitals)} orbitals ({points} points):')
		for i, orbital in enumerate(orbitals):

			utils.message(f'	Progress: {i+1}/{len(orbitals)} = {round((i+1)/len(orbitals)*100,2)}%')
			samples = 10*points
			ranges = orbital.ranges

			x, y, z = ((np.random.randint(ranges[0]*10000, ranges[1]*10000, size=samples)/10000), (np.random.randint(ranges[2]*10000, ranges[3]*10000, size=samples)/10000), (np.random.randint(ranges[4]*10000, ranges[5]*10000, size=samples)/10000))
			d = orbital.evaluate(np.asarray((x, y, z)), True).flatten()

			index = abs(d**2).argsort()[::-1]
			colours = colour_map[d].T

			x, y, z, colours = x[index][0:points], y[index][0:points], z[index][0:points], colours[index][0:points]
			dens_pos = self.rotate(np.asarray((x, y, z)).T, orbital.molecule.rotation)

			self._dens_pos[orbital], self._dens_colours[orbital] = dens_pos, colours
			orbital.molecule._dens_pos[orbital], orbital.molecule._dens_colours[orbital] = dens_pos, colours

		utils.message(f'Orbitals prepared. Please use Screen3D.draw_density() to display the orbitals.')


	def draw_mesh(self, mesh, colour=(255,255,255, 200), lighting=(1,0,0), fill=True, lighting_colour=(255,255,255)):
		'''
		Method that draws a mesh from a 3d matrix, where the rows represent the triangles,
		the rows in the rows represent the verteces of the triangles. The elements are the coordinates
		'''

		mesh = np.asarray(mesh)
		project = self.project_array
		draw_polygon = pg.draw.polygon

		order = []

		for t in mesh:
			order.append((t, max(euclidean(t[0], self.camera_position), euclidean(t[1], self.camera_position), euclidean(t[2], self.camera_position))))


		for triangle, _ in reversed(sorted(order, key=lambda x: x[1])):
			# tri = self.rotate(triangle, self.camera_orientation)
			tri = triangle
			p1, p2 = (tri[1]-tri[0]), (tri[2]-tri[0])
			norm = np.cross(p1, p2)
			norm /= np.linalg.norm(norm)

			light_direction = lighting - np.mean(triangle, axis=1)

			angle = np.arccos(np.clip(np.dot(norm, -np.asarray(light_direction)), -0.9, 0.9))/math.pi
			# print(angle)

			triangle = project(triangle)
			
			colour = (angle * np.array([255,255,255])).astype(int)
			colour -= (angle * np.asarray(lighting_colour)/2).astype(int)
			if fill:
				draw_polygon(self.disp, colour, triangle)
			else:
				draw_polygon(self.disp, colour, triangle, 1)


	def draw_density(self, orbital, points=50000, colour_map=cmap.BlueRed(posneg_mode=True), grid_mode=False):
		if not orbital in self._dens_pos:
			samples = 50*points
			ranges = orbital.ranges
			if grid_mode:
				x, y, z = np.linspace(ranges[0], ranges[1], points//3), np.linspace(ranges[2], ranges[3], points//3), np.linspace(ranges[4], ranges[5], points//3)
				x, y, z = np.meshgrid(x, y, z)
				x, y, z = x.flatten(), y.flatten(), z.flatten()


				d = orbital.evaluate(np.asarray((x, y, z))).flatten()
				colours = colour_map[d].T

			else:
				x, y, z = ((np.random.randint(ranges[0]*10000, ranges[1]*10000, size=samples)/10000), (np.random.randint(ranges[2]*10000, ranges[3]*10000, size=samples)/10000), (np.random.randint(ranges[4]*10000, ranges[5]*10000, size=samples)/10000))
			
				d = orbital.evaluate(np.asarray((x, y, z))).flatten()

				index = abs(d**2).argsort()[::-1]
				colours = colour_map[d].T

				x, y, z, colours = x[index][0:points], y[index][0:points], z[index][0:points], colours[index][0:points]
			dens_pos = self.rotate(np.asarray((x, y, z)).T, orbital.molecule.rotation)

			self._dens_pos[orbital], self._dens_colours[orbital] = dens_pos, colours
			orbital.molecule._dens_pos[orbital], orbital.molecule._dens_colours[orbital] = dens_pos, colours

		self.draw_pixels(orbital.molecule._dens_pos[orbital], colour_array=orbital.molecule._dens_colours[orbital])


	def draw_electrostatic_potential(self, molecule, points=50000, colour_map=cmap.ElectroStat()):
		if not hasattr(molecule, '_elec_stat_pos'):
			utils.message(f'Calculating electrostatic potential of {molecule.name} ...')
			samples = 50*points
			rang = np.amax([np.abs(atom.coords) for atom in molecule.atoms]) + 4

			x, y, z = ((np.random.randint(-rang*10000, rang*10000, size=samples)/10000), (np.random.randint(-rang*10000, rang*10000, size=samples)/10000), (np.random.randint(-rang*10000, rang*10000, size=samples)/10000))
			d = molecule.electrostatic_potential(np.asarray((x, y, z)).T).flatten()
			index = abs(d).argsort()[::-1]
			d = np.maximum(d, np.mean(d)*4)
			colours = colour_map[d].T

			x, y, z, colours = x[index][0:points], y[index][0:points], z[index][0:points], colours[index][0:points]
			molecule._elec_stat_pos, molecule._elec_stat_colours = np.asarray((x, y, z)).T, colours

		self.draw_pixels(molecule._elec_stat_pos, colour_array=molecule._elec_stat_colours)


	def draw_shape(self, shape, colour=(255,255,255), double_sided=False, mode="fill", draw_atoms=True, draw_bonds=True, colour_bonds=True, draw_hydrogens=True, wireframe=False, high_contrast_mode=True):
		if shape.type == 'flat':
			if mode == "fill":
				faces = shape.faces(double_sided=double_sided)
				[self.draw_polygon(face, colour) for face in faces]
			if mode == "lines":
				self.draw_lines(shape.points, colour, shape.closed)

		elif shape.type == 'molecule':
			if not high_contrast_mode:
				contrast_colour = self.contrast_colour = self.bkgr_colour
			else:
				contrast_colour = self.contrast_colour = (255-self.bkgr_colour[0], 255-self.bkgr_colour[1], 255-self.bkgr_colour[2])

			atom_draw_pos = {}
			atom_draw_rad = {}
			p = shape.position

			cam_pos = self.camera_position #reference to camera position
			if draw_hydrogens:
				atoms = shape.atoms #reference to the atoms
			else:
				atoms = shape.get_by_element('H', blacklist=True)
			coords = np.asarray([a.coords for a in atoms]) #coordinates converted to np array

			dists = np.asarray([a.distance_to((cam_pos)) for a in atoms])#calculate dists to determine order of drawing
			indices = np.argsort(dists)[::-1] #determine order of drawing by sorting the dists and reversing
			atoms = np.asarray(atoms)[indices] #sort atoms by distance to cam_pos
			dists = dists[indices]
			deltas = coords - cam_pos
			deltas = deltas[indices]

			d2 = lambda x, y: (x - y)/2

			h = self.height + 200
			w = self.width + 200

			disp = self.disp
			single = self.draw_single_bond
			double = self.draw_double_bond
			triple = self.draw_triple_bond
			line = pg.draw.line
			project = self.project

			prev_indices = []
			for i, a1 in enumerate(atoms):
				width = max(1, int(50/dists[i]))
				# if deltas[i][2] < 0:
				if True:
					if not wireframe:
						prev_indices.append(a1)
						c1 = a1.coords
						if draw_bonds:
							if colour_bonds:
								for a2 in a1.bonds:
									if not a2.symbol == 'H' or draw_hydrogens:
										if not a2 in prev_indices:
											c2 = a2.coords
											if a1.bond_orders[a2] == 1:
												# single((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
												
												poss = [project(pos) for pos in (c1 + p, c1 + p + d2(c2,c1))]
												if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
													line(disp, contrast_colour, poss[0], poss[1], width+3)
													# line(disp, a1.draw_colour, poss[0], poss[1], width)

												poss = [project(pos) for pos in (c2 + p - d2(c2,c1), c2 + p)]
												if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
													line(disp, contrast_colour, poss[0], poss[1], width+3)
													# line(disp, a2.draw_colour, poss[0], poss[1], width)

											elif a1.bond_orders[a2] == 2:
												double((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
											elif a1.bond_orders[a2] == 3:
												triple((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)


						if draw_atoms:
							rad = int(a1.radius/dists[i] * shape.scale)
							self.draw_circle(c1+p, rad+1, contrast_colour)
							self.draw_circle(c1+p, rad, a1.draw_colour)
							atom_draw_pos[a1] = project(c1+p)
							atom_draw_rad[a1] = rad*1.1

						if draw_bonds:
							if colour_bonds:
								for a2 in a1.bonds:
									if not a2.symbol == 'H' or draw_hydrogens:
										if not a2 in prev_indices:
											c2 = a2.coords
											if a1.bond_orders[a2] == 1:
												# single(, width=width, colour=a2.draw_colour)	
												poss = [project(pos) for pos in (c2 + p - d2(c2,c1), c2 + p)]
												if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
													# line(disp, contrast_colour, poss[0], poss[1], width+3)
													line(disp, a2.draw_colour, poss[0], poss[1], width)

												poss = [project(pos) for pos in (c1 + p, c1 + p + d2(c2,c1))]
												if (-200 <= poss[0][1] <= h and -200 <= poss[0][0] <= w and -200 <= poss[1][1] <= h and -200 <= poss[1][0] <= w):
													# line(disp, contrast_colour, poss[0], poss[1], width+3)
													line(disp, a1.draw_colour, poss[0], poss[1], width)

											elif a1.bond_orders[a2] == 2:
												double((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
											elif a1.bond_orders[a2] == 3:
												triple((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
					
					elif wireframe:
							c1 = a1.coords
							for a2 in a1.bonds:
								if not a2 in prev_indices:
									c2 = a2.coords
									if a1.bond_orders[a2] == 1:
										single((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
										single((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)
									elif a1.bond_orders[a2] == 2:
										double((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
										double((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
									elif a1.bond_orders[a2] == 3:
										triple((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
										triple((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)


			self.atom_draw_pos = atom_draw_pos
			self.atom_draw_rad = atom_draw_rad


		elif shape.type == 'molecule':
			if not high_contrast_mode:
				self.contrast_colour = self.bkgr_colour
			else:
				self.contrast_colour = (255-self.bkgr_colour[0], 255-self.bkgr_colour[1], 255-self.bkgr_colour[2])

			atom_draw_pos = {}
			atom_draw_rad = {}
			p = shape.position

			cam_pos = self.camera_position #reference to camera position
			if draw_hydrogens:
				atoms = shape.atoms #reference to the atoms
			else:
				atoms = shape.get_by_element('H', blacklist=True)
			coords = np.asarray([a.coords for a in atoms]) #coordinates converted to np array

			dists = np.asarray([a.distance_to((cam_pos)) for a in atoms])#calculate dists to determine order of drawing
			indices = np.argsort(dists)[::-1] #determine order of drawing by sorting the dists and reversing
			atoms = np.asarray(atoms)[indices] #sort atoms by distance to cam_pos
			dists = dists[indices]
			deltas = coords - cam_pos
			deltas = deltas[indices]

			d2 = lambda x, y: (x - y)/2

			prev_indices = []
			for i, a1 in enumerate(atoms):
				width = max(1, int(50/dists[i]))
				# if deltas[i][2] < 0:
				if True:
					if not wireframe:
						prev_indices.append(a1)
						c1 = a1.coords
						if draw_bonds:
							if colour_bonds:
								for a2 in a1.bonds:
									if not a2.symbol == 'H' or draw_hydrogens:
										if not a2 in prev_indices:
											c2 = a2.coords
											if a1.bond_orders[a2] == 1:
												self.draw_single_bond((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
											elif a1.bond_orders[a2] == 2:
												self.draw_double_bond((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
											elif a1.bond_orders[a2] == 3:
												self.draw_triple_bond((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)


						if draw_atoms:
							rad = int(a1.radius/dists[i] * shape.scale)
							self.draw_circle(c1+p, rad+1, self.contrast_colour)
							self.draw_circle(c1+p, rad, a1.draw_colour)
							atom_draw_pos[a1] = self.project(c1+p)
							atom_draw_rad[a1] = rad*1.1

						if draw_bonds:
							if colour_bonds:
								for a2 in a1.bonds:
									if not a2.symbol == 'H' or draw_hydrogens:
										if not a2 in prev_indices:
											c2 = a2.coords
											if a1.bond_orders[a2] == 1:
												self.draw_single_bond((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
											elif a1.bond_orders[a2] == 2:
												self.draw_double_bond((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
											elif a1.bond_orders[a2] == 3:
												self.draw_triple_bond((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
					
					elif wireframe:
							c1 = a1.coords
							for a2 in a1.bonds:
								if not a2 in prev_indices:
									c2 = a2.coords
									if a1.bond_orders[a2] == 1:
										self.draw_single_bond((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
										self.draw_single_bond((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)
									elif a1.bond_orders[a2] == 2:
										self.draw_double_bond((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
										self.draw_double_bond((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)	
									elif a1.bond_orders[a2] == 3:
										self.draw_triple_bond((c1 + p, c1 + p + d2(c2,c1)), width=width, colour=a1.draw_colour)
										self.draw_triple_bond((c2 + p - d2(c2,c1), c2 + p), width=width, colour=a2.draw_colour)


			self.atom_draw_pos = atom_draw_pos
			self.atom_draw_rad = atom_draw_rad


	def draw_axes(self, length=1):
		self.draw_line([np.asarray((0,0,0)),np.asarray((length,0,0))], colour=(255,0,0))
		self.draw_line([np.asarray((0,0,0)),np.asarray((0,length,0))], colour=(0,255,0))
		self.draw_line([np.asarray((0,0,0)),np.asarray((0,0,length))], colour=(0,0,255))

	def clear(self):
		self.disp.fill(self.bkgr_colour)
		# img = pg.image.load(r"C:\Users\Yuman\AppData\Local\Temp\logo voor yuman.png")
		# img = pg.transform.scale(img, self.size)
		# self.disp.blit(img, (0,0))

	def move(self, vector):
		a = np.asarray(vector).T
		t = self.camera_orientation
		c = self.camera_position.T
		x_rot_mat = np.array([[1,0,0], [0, cos(t[0]), sin(t[0])], [0, -sin(t[0]), cos(t[0])]])
		y_rot_mat = np.array([[cos(t[1]), 0, -sin(t[1])], [0,1,0], [sin(t[1]), 0, cos(t[1])]])
		z_rot_mat = np.array([[cos(t[2]), sin(t[2]), 0], [-sin(t[2]), cos(t[2]), 0], [0,0,1]])
		z_rot_mat = np.array([[1,0,0], [0,1,0], [0,0,1]])
		result = x_rot_mat @ y_rot_mat @ z_rot_mat @ (a)
		self.camera_position += result.T

	def look_at(self, point=None, obj=None):
		if point is None:
			point = obj.position

		point = np.asarray(point)

		x, y, z = (point - self.camera_position)/np.linalg.norm(point - self.camera_position)
		# print(y/z)
		# rotx = math.asin(y/z)
		# if z >= 0:
		# 	roty = -math.atan2(x * math.cos(rotx), z)
		# else:
		# 	roty = math.atan2(x * math.cos(rotx), -z)

		rotx = math.atan2( y, z )
		roty = math.atan2( x * math.cos(rotx), z )
		rotz = math.atan2( math.cos(rotx), math.sin(rotx) * math.sin(roty) )

		self.camera_orientation = np.array([rotx, roty, 0])
		return self.camera_orientation



	def rotate(self, array, rotation):
		'''
		Method that rotates atom coordinates by rotation
		'''

		r = rotation[0]
		Rx = np.array(([	  1, 	  0,	   0],
					   [	  0, cos(r), -sin(r)],
					   [      0, sin(r),  cos(r)]))

		r = rotation[1]
		Ry = np.array(([ cos(r),  	   0, sin(r)],
					   [ 	  0, 	   1,	   0],
					   [-sin(r), 	   0, cos(r)]))

		return (Rx @ Ry @ array.T).T
			