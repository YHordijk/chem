import numpy as np
import pkg.integral as integral
import pkg.data as data


def extended_huckel(aos, K=1.75):
	n = len(aos)
	S = integral.overlap_matrix(aos)
	H = np.eye(n)

	for i in range(n):
		H[i][i] = -float(data.IONISATION_ENERGIES[aos[i].atom_number][0])/10.364

	for i in range(n):
		for j in range(i+1, n):
			H[i][j] = H[j][i] = K * S[i][j] * (H[i][i] + H[j][j])/2

	E, C = np.linalg.eigh(H)
	return E, C