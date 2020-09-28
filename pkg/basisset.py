# import pkg.data as data
import periodictable as pt

help(pt.elements.symbol('H'))

class BasisSet:
	def __init__(self, molecule):
		self.molecule = molecule
		self.n_basis_funcs = sum(data.MAX_PRINCIPAL_QUANTUM_NUMBER[pt] for i in molecule.elements)