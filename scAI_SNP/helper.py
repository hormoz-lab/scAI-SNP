import sys
import os
import numpy as np
import pandas as pd
#from returns import returns

# variables and constants
n_mut = 4586890

#@returns(int)

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
			'NA': np.nan,
			'Na': np.nan,
			'na': np.nan
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

# a function that reads mean genotype of the training data
def read_center():
	try:
		file_mean = 'data/genotype_mean.col'
		return pd.read_csv(file_mean, header = None)
	except FileNotFoundError:
		print("ERROR (PACKAGE): mean genotype of training not found")
		print(f"current directory: {os.getcwd()}")
		sys.exit(1)

# a function from scikit-learn-1.2.0/sklearn/utils/extmath.py
def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret
