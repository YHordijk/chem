import pkg.molecule as mol
import pkg.display as disp
import pkg.minimizers as minimizers
import pkg.mesh as mesh
import copy, math
import numpy as np
from time import perf_counter as pc



def update(self, params=None):
	# start = pc()
	params['mesh'] = surface_mesh
	params['mesh'].rotate(params['rot'])
	self.draw_mesh(params, surface_mesh, fill=True)
	# print('Mesh drawing (s):', pc() - start)
	# self.camera_position[2] = 10*math.sin(4*params['time'])
	# self.camera_position[1] = 10*math.cos(4*params['time'])

	# self.look_at(params['mols'][0].center_of_mass())


# disp.Display.update = update



m = mol.load_mol('Octamethylcyclotetrasiloxane', download_from_pdb=False)
d = disp.Display(bkgr_colour=(0, 0, 0))


m.remove_non_bonded_atoms()
# m.shake(1)
# m.center()

surface_mesh = mesh.molecule_surface(m, resolution=0.5, thresh=1)
d.draw_molecule(m, draw_atoms=True, draw_bonds=True)

# mini = minimizers.UFF()
# mini.get_energy(m, verbose=True)



# mols = [m.copy()]
# for i in range(5000):
# 	# print(i)
# 	grad = mini.get_gradient(m)
# 	m.apply_gradient(grad, 0.00001)
# 	if i%30==0:
# 		# print(i)
# 		# print(m.bond_angle(m.atoms[0], m.atoms[1], m.atoms[2])/math.pi) 
# 		mols.append(m.copy())

# # # [ for m in mols]

# mini.get_energy(m, verbose=True)

# d.draw_molecule_animation(mols, draw_hydrogens=True, draw_atoms=True)


