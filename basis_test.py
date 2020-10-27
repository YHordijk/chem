import pkg.molecule as mol
import pkg.basissets as basis
import pkg.display as disp
import pkg.mesh as mesh
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

def rotate(array, rotation):
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

		return R @ array.T


def update(self, params=None):
	if params['updt'] == 1:
		params['3d_pos'] = np.hstack((X,Y,Z))
	params['3d_pos'] = rotate(params['3d_pos'], params['rot']).T
	self.draw_3dpoints(params, params['3d_pos'], psi)
	# params['mesh'] = surface_mesh
	# params['mesh'].rotate(params['rot'])
	# self.draw_mesh(params, surface_mesh, fill=True)


# disp.Display.update = update
d = disp.Display(bkgr_colour=(0, 0, 0))

m = mol.load_mol('_test')

d.draw_molecule(m, draw_atoms=True, draw_bonds=True)
a = m.atoms
b = basis.Basis()

b.load_molecule(m)




dims = (15,15,15)
dims_prod = dims[0]*dims[1]*dims[2]
ranges = ((-2,2),(-2,2),(-2,2))


X, Y, Z = np.meshgrid(np.linspace(*ranges[0],dims[0]), np.linspace(*ranges[1],dims[1]), np.linspace(*ranges[2],dims[2]))

X = X.reshape(dims_prod,1)
Y = Y.reshape(dims_prod,1)
Z = Z.reshape(dims_prod,1)


def f(X,Y,Z):

	# dims = X.size, Y.size, Z.size
	# dims_prod = dims[0]*dims[1]*dims[2]


	# X = X.reshape(dims_prod,1)
	# Y = Y.reshape(dims_prod,1)
	# Z = Z.reshape(dims_prod,1)

	u = b.basis_functions[a[0]][2][0][0](X,Y,Z) + b.basis_functions[a[1]][2][0][0](X,Y,Z)

	# u = u.reshape((dims[0], dims[1], dims[2]))

	return abs(u)


psi = b.basis_functions[a[0]][2][1][0](X,Y,Z) + b.basis_functions[a[0]][2][0][0](X,Y,Z)
# psi = psi.reshape((dims[0], dims[1], dims[2]))


# surface_mesh = mesh.create_surface(f, resolution=3, thresh=0.5, ranges=ranges)




d.draw_molecule(m, draw_atoms=True, draw_bonds=True)