import pkg.molecule as mol

m = mol.load_mol('benzene')
d = mol.Display()
d.draw_molecule(m)

