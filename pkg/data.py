
import os

def csv_to_dict(file):
	d = {}
	with open(file, 'r') as f:
		for row in f.readlines():
			row = row.strip('\n')
			row = row.split(', ')
			d[int(row[0])] = row[1]

	return d



try:
	RESOURCES_DIR = os.getcwd() + r'\pkg\data\resources'
	MAX_PRINCIPAL_QUANTUM_NUMBER = csv_to_dict(rf'{RESOURCES_DIR}\elements\max_quantum_number.csv')
except:
	RESOURCES_DIR = os.getcwd() + r'\data\resources'
	MAX_PRINCIPAL_QUANTUM_NUMBER = csv_to_dict(rf'{RESOURCES_DIR}\elements\max_quantum_number.csv')