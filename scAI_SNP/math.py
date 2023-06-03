import numpy as np
import pandas as pd
from returns import returns

# variables and constants
n_mut = 4586890

@returns(int)
def div_int(x, y):
    return x / y

def center(x, y):
	return round(x - y, 2)

def cmd_center(args=None):
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('x', type=float)
	parser.add_argument('y', type=float)
	parsed_args = parser.parse_args(args)
	print(center(parsed_args.x, parsed_args.y))

# a function that reads a file
def read_validate(file):
	df = pd.read_csv(file, header = None)

	# checking number of rows
	if df.shape[0] != n_mut:
		print(f"ERROR (INPUT): The number of rows is not {n_mut}, but {df.shape[0]}")
		return None

	# replacing quoted and 'NA' values to correct numeric values
	df.replace(
		{
			'"0"': 0, 
			'"0.0"': 0, 
			'"1"': 1, 
			'"1.0"': 1, 
			'"0.5"': 0.5, 
			'NA': np.nan
		}, inplace=True
	)

	# file conversion to numpy
	data_values = df.values.astype(float)
    
	# checking the values using numpy functions
	allowed_values = np.array([0.0, 0.5, 1.0, np.nan])
	mask = np.isin(data_values, allowed_values, assume_unique=True, invert=True)
	if mask.any():
		print(f"ERROR (INPUT): Found invalid values, entries must be {0, 0.5, 1, NA}")
		return None

	return data_values
