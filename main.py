import pkg.molecule as mol
import pkg.display as disp
import pkg.minimizers as minimizers
import copy, math


m = mol.load_mol('Pentacynium')
d = disp.Display(bkgr_colour=(0, 0, 0))

m.remove_non_bonded_atoms()
d.draw_molecule(m)
# m.shake(1)
m.center()
mini = minimizers.UFF()
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


