import pkg.molecule as mol
import pkg.basisset as basis
import pkg.display as disp
import pkg.mesh as mesh
import pkg.integral as integral
import pkg.extended_huckel as eh
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
		params['R'] = np.floor(psi**2*100).astype(int)
		

	params['3d_pos'] = rotate(params['3d_pos'], params['rot']).T
	for ev in params['events']:
		if ev.type == pg.KEYDOWN:
			if ev.key == pg.K_RIGHT:
				psi_index = psis.index(psi)
				psi = psis[(psi_index+1)%len(psis)]
				params['R'] = np.floor(psi**2*100).astype(int)
			if ev.key == pg.K_LEFT:
				psi_index = psis.index(psi)
				psi = psis[(psi_index-1)%len(psis)]
				params['R'] = np.floor(psi**2*100).astype(int)


	self.draw_3dpoints(params, params['3d_pos'], psi)
	# params['mesh'] = surface_mesh
	# params['mesh'].rotate(params['rot'])
	# self.draw_mesh(params, surface_mesh, fill=True)


disp.Display.update = update
d = disp.Display(bkgr_colour=(0, 0, 0))

m = mol.load_mol('Phosphonitrilic chloride trimer')

a = m.atoms

aos = basis.load_molecule(m)

aos_list = [a.orbital_list for a in aos.values()]
aos_list = sum(aos_list, [])


aos_list_valence = [list(filter(lambda x: x.n == a[1].max_n, a[1].orbital_list)) for a in aos.items()]
aos_list_valence = sum(aos_list_valence, [])



# integral.hartree_fock(aos_list, m.atoms)


dims = (20,20,20)
dims_prod = dims[0]*dims[1]*dims[2]
ranges = ((-4,4),(-4,4),(-4,4))


X, Y, Z = np.meshgrid(np.linspace(*ranges[0],dims[0]), np.linspace(*ranges[1],dims[1]), np.linspace(*ranges[2],dims[2]))

X = X.reshape(dims_prod,1)
Y = Y.reshape(dims_prod,1)
Z = Z.reshape(dims_prod,1)

# print(X.shape)

def f(X,Y,Z):

	dims = X.size, Y.size, Z.size
	dims_prod = dims[0]*dims[1]*dims[2]


	X = X.reshape(dims_prod,1)
	Y = Y.reshape(dims_prod,1)
	Z = Z.reshape(dims_prod,1)

	u = aos[a[0]][1,0,0](X,Y,Z) + aos[a[1]][1,0,0](X,Y,Z)

	# u = u.reshape((dims[0], dims[1], dims[2]))

	return abs(u)

# psi = aos[a[1]][1,0,0](X,Y,Z) + aos[a[0]][1,0,0](X,Y,Z) + aos[a[2]][1,0,0](X,Y,Z)
# psi = psi.reshape((dims[0], dims[1], dims[2]))

energies, coeffs = eh.extended_huckel(aos_list_valence)
psis = [sum(c*ao(X,Y,Z) for c, ao in zip(coeff, aos_list_valence)) for coeff in coeffs.T]
psi = psis[0]

d.draw_molecule(m, draw_atoms=False, draw_bonds=True)

