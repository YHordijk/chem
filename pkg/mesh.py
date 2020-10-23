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


		self.vertices = np.asarray([R@v for v in self.vertices])


def molecule_surface(mol, resolution=0.5, dist_thresh=0.3):
	dims = mol.get_center() + dist_thresh * 2
	mini, maxi = mol.get_corners()

	x, y, z = (np.arange(mini[0]-dist_thresh-resolution, maxi[0]+dist_thresh+resolution, resolution), 
			   np.arange(mini[1]-dist_thresh-resolution, maxi[1]+dist_thresh+resolution, resolution), 
			   np.arange(mini[2]-dist_thresh-resolution, maxi[2]+dist_thresh+resolution, resolution))

	X, Y, Z = np.meshgrid(x, y, z)

	u = np.sqrt((X)**2 + (Y)**2 + (Z)**2)
	for a in mol.atoms:
		u = np.minimum(u, np.sqrt((X-a.position[0])**2 + (Y-a.position[1])**2 + (Z-a.position[2])**2))


	vertices, triangles = mcubes.marching_cubes(u, dist_thresh)

	# print(vertices)
	# print(triangles)
	return Mesh(vertices, triangles)