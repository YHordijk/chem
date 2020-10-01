
import os

def csv_to_dict(file):
	d = {}
	with open(file, 'r') as f:
		for row in f.readlines():
			row = row.strip('\n')
			row = row.split(', ')
			d[int(row[0])] = row[1:]

	return d



RESOURCES_DIR = os.getcwd() + r'\pkg\data\resources'
if  not os.path.exists(RESOURCES_DIR):
	RESOURCES_DIR = os.getcwd() + r'\data\resources'

#element parameters
MAX_PRINCIPAL_QUANTUM_NUMBER = csv_to_dict(rf'{RESOURCES_DIR}\elements\max_quantum_number.csv')
MAX_VALENCE = csv_to_dict(rf'{RESOURCES_DIR}\elements\max_valence.csv')
ATOM_COLOURS = csv_to_dict(rf'{RESOURCES_DIR}\elements\colours.csv')


#forcefield parameters
def load_uff_parameters(path):
	uff_params = {}
	uff_params['valence_bond'] = {}
	uff_params['valence_angle'] = {}
	uff_params['nonbond_distance'] = {}
	uff_params['nonbond_scale'] = {}
	uff_params['nonbond_energy'] = {}
	uff_params['effective_charge'] = {}
	uff_params['sp3_torsional_barrier_params'] = {}
	uff_params['sp2_torsional_barrier_params'] = {}
	uff_params['electro_negativity'] = {}

	with open(path, 'r') as f:
		for row in f.readlines():
			parts = row.split()

			if len(parts) == 0:
				continue

			if parts[0] == 'param':
				el = parts[1]
				uff_params['valence_bond'][el] = float(parts[2])
				uff_params['valence_angle'][el] = float(parts[3])
				uff_params['nonbond_distance'][el] = float(parts[4])
				uff_params['nonbond_scale'][el] = float(parts[6])
				uff_params['nonbond_energy'][el] = float(parts[5])
				uff_params['effective_charge'][el] = float(parts[7])
				uff_params['sp3_torsional_barrier_params'][el] = float(parts[8])
				uff_params['sp2_torsional_barrier_params'][el] = float(parts[9])
				uff_params['electro_negativity'][el] = float(parts[10])

	return uff_params


FF_UFF_PARAMETERS = load_uff_parameters(rf'{RESOURCES_DIR}\forcefields\UFF\UFF.prm')