import pkg.molecule as mol
import pkg.basisset as basis
import pkg.display as disp
import pkg.mesh as mesh
import pkg.integral as integral
import pkg.extended_huckel as eh
import pkg.minimizers as mini
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin
import pygame as pg



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
		params['psis'] = psis
		params['psi'] = params['psis'][0]
		params['psi_idx'] = 0
		params['3d_pos'] = np.hstack((X,Y,Z))
		params['R'] = np.floor(params['psi']*2*200).astype(int)

		
	params['3d_pos'] = rotate(params['3d_pos'], params['rot']).T
	for ev in params['events']:
		if ev.type == pg.KEYDOWN:
			if ev.key == pg.K_RIGHT:
				params['psi_idx'] = (params['psi_idx'] + 1) % len(params['psis'])
				params['psi'] = params['psis'][params['psi_idx']]
				params['R'] = np.floor(params['psi']**2*200).astype(int)
			if ev.key == pg.K_LEFT:
				params['psi_idx'] = (params['psi_idx'] - 1) % len(params['psis'])
				params['psi'] = params['psis'][params['psi_idx']]
				params['R'] = np.floor(params['psi']**2*200).astype(int)


	self.draw_3dpoints(params, params['3d_pos'], params['psi'])
	# params['mesh'] = surface_mesh
	# params['mesh'].rotate(params['rot'])
	# self.draw_mesh(params, surface_mesh, fill=False)



d = disp.Display(bkgr_colour=(0, 0, 0))
uff = mini.UFF(verbose=False)
m = mol.load_mol('water')
m.center(m.atoms[0].position)

print(uff.get_energy(m))
print(m)
m.shake(0)
mollist = mini.minimize_molecule(m)
energies = [uff.get_energy(x) for x in mollist]
plt.plot(range(len(mollist)), energies)
plt.show()
d.draw_molecule_animation([m, mollist[-1]], draw_atoms=True, draw_bonds=True, animation_speed=30)

# print(uff.get_energy(m))

a = m.atoms

aos = basis.load_molecule(m)

aos_list = [a.orbital_list for a in aos.values()]
aos_list = sum(aos_list, [])


aos_list_valence = [list(filter(lambda x: x.n == a[1].max_n, a[1].orbital_list)) for a in aos.items()]
aos_list_valence = sum(aos_list_valence, [])



# integral.hartree_fock(aos_list, m.atoms)


dims = (20,20,20)
dims_prod = dims[0]*dims[1]*dims[2]
ranges = ((-5,5),(-5,5),(-5,5))


X, Y, Z = np.meshgrid(np.linspace(*ranges[0],dims[0]), np.linspace(*ranges[1],dims[1]), np.linspace(*ranges[2],dims[2]))
X = X.reshape(dims_prod,1)
Y = Y.reshape(dims_prod,1)
Z = Z.reshape(dims_prod,1)


def f(x,y,z):
	# dims = x.size, y.size, z.size
	# dims_prod = dims[0]*dims[1]*dims[2]
	# x = x.reshape(dims_prod,1)
	# y = y.reshape(dims_prod,1)
	# z = z.reshape(dims_prod,1)

	u = sum(c*ao(x,y,z) for c, ao in zip(coeffs[0], aos_list_valence))
	# u = u.reshape((dims[0], dims[1], dims[2]))

	return abs(u)

# psi = aos[a[1]][1,0,0](X,Y,Z) + aos[a[0]][1,0,0](X,Y,Z) + aos[a[2]][1,0,0](X,Y,Z)



energies, coeffs = eh.extended_huckel(aos_list_valence)
psis = [sum(c*ao(X,Y,Z) for c, ao in zip(coeff, aos_list_valence)) for coeff in coeffs.T]
# surface_mesh = mesh.create_surface(f)

# psi = psis[0]

disp.Display.update = update
d.draw_molecule(m, draw_atoms=True, draw_bonds=True)

