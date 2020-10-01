import pkg.molecule as mol

m = mol.load_mol('4avd')
d = mol.Display(bkgr_colour=(100, 100, 190))
m.remove_non_bonded_atoms()
d.draw_molecule(m, draw_hydrogens=True, draw_atoms=False)

