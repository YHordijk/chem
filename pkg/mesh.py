import numpy as np
import mcubes
from math import cos, sin




class Mesh:
	def __init__(self, vertices=[], triangles=[]):
		self.vertices = vertices
		self.triangles = triangles




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

		self.vertices = (R @ self.vertices.T).T


def create_surface(f, resolution=0.5, thresh=0.3, ranges=((-1,1),(-1,1),(-1,1))):
	x, y, z = (np.arange(ranges[0][0], ranges[0][1], resolution), 
			   np.arange(ranges[1][0], ranges[1][1], resolution), 
			   np.arange(ranges[2][0], ranges[2][1], resolution))

	X, Y, Z = np.meshgrid(x, y, z)
	u = f(X,Y,Z)

	vertices, triangles = mcubes.marching_cubes(u, thresh)

	return Mesh(vertices, triangles)


def molecule_surface(mol, resolution=0.5, thresh=0.3):
	def f(X,Y,Z):
		u = np.sqrt((X)**2 + (Y)**2 + (Z)**2)
		for a in mol.atoms:
			u = np.minimum(u, np.sqrt((X-a.position[0])**2 + (Y-a.position[1])**2 + (Z-a.position[2])**2))
		return u

	dims = mol.get_center() + thresh * 2
	mini, maxi = mol.get_corners()

	ranges = ((mini[0]-thresh-resolution, maxi[0]+thresh+resolution),
			  (mini[1]-thresh-resolution, maxi[1]+thresh+resolution),
			  (mini[2]-thresh-resolution, maxi[2]+thresh+resolution))

	return create_surface(f, resolution, thresh=thresh, ranges=ranges)