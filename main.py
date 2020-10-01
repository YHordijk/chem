import pkg.molecule as mol
import pkg.display as disp
import pkg.minimizers as minimizers
import copy


m = mol.load_mol('methane')
d = disp.Display(bkgr_colour=(0, 0, 0))
m.remove_non_bonded_atoms()
m.shake(0.1)
m.center()


mini = minimizers.UFF()
mini.get_energy(m, verbose=False) * 4.182

mols = [m.copy()]
for i in range(20000):
	# print(i)
	grad = mini.get_gradient(m)
	m.apply_gradient(grad, 0.000001)
	if i%100==0:
		print(i)
		mols.append(m.copy())

# [print(m.bond_angle(m.atoms[0], m.atoms[1], m.atoms[2])) for m in mols]

mini.get_energy(m, verbose=True) * 4.182

d.draw_molecule_animation(mols, draw_hydrogens=True, draw_atoms=True)


