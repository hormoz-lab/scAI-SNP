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
	df = pd.read_csv(file, header = None, sep = '\t', encoding='utf-8', dtype=str)
	print(df)
	print(type(df))
	print(df.shape)
	# checking number of rows
	if df.shape[0] != n_mut:
		print(f"ERROR (INPUT): The number of rows is not {n_mut}, but {df.shape[0]}")
		print(f"current directory: {os.getcwd()}")
		return None

	# replacing quoted and 'NA' values to correct numeric values
	df.replace(
		{
			'"0"': 0, 
			'"-0"': 0, 
			'"0.0"': 0, 
			'"-0.0"': 0, 
			'"1"': 1, 
			'"1.0"': 1, 
			'"0.5"': 0.5, 
			'NA': np.nan,
			'Na': np.nan,
			'na': np.nan,
			'NaN': np.nan,
			'nan': np.nan
		}, inplace=True
	)

	all_values = df.values.flatten()
	# checking the values
	print('all values')
	print(pd.unique(all_values))
	# file conversion to numpy
	data_values = df.values.astype(float)
    
	# checking the values using numpy functions
	allowed_values = np.array([-0.0, 0.0, 0.5, 1.0, np.nan])
	# return true if the values are in the allowed_values
	mask_num = np.isin(data_values, allowed_values, assume_unique=True)
	# return true if the values are nan
	mask_nan = np.isnan(data_values)
	# return true if the values are not in the allowed_values and not nan
	mask_not_allowed = ~mask_num & ~mask_nan
	if mask_not_allowed.any():
		invalid_values = data_values[mask_not_allowed]
		print(f"ERROR (INPUT): Found invalid values: {invalid_values}, entries must be {0, 0.5, 1, 'NA'}")
		print(f"function classify will now terminate, please review the input file")
		# return None TEMP
	
	return data_values

# a function that reads mean genotype of the training data
def read_center():
	try:
		file_mean = 'data/genotype_mean.col'
		return pd.read_csv(file_mean, header = None).to_numpy().flatten()
	except FileNotFoundError:
		print("ERROR (PACKAGE): mean genotype of training not found")
		print(f"current directory: {os.getcwd()}")
		sys.exit(1)

# a function that subtracts the mean genotype from the input data
def center_scale_input(mat_input, col_mean):
	# count missing values in each column
	vec_n_missing = np.sum(np.isnan(mat_input), axis = 0)
	vec_inverse_present_data = 1/(1 - vec_n_missing/n_mut)

	# count the number of columns in the input
	n_col_input = mat_input.shape[1]
	
	## initialize centered input data	
	mat_input_centered_scaled = np.empty_like(mat_input)

	# centering the input data
	for COL in range(n_col_input):
		col_center = mat_input[:, COL] - col_mean
		mask = np.isnan(col_center)
		col_center[mask] = 0
		# also scale by the inverse of the fraction of present data
		mat_input_centered_scaled[:, COL] = col_center * vec_inverse_present_data[COL]

	return mat_input_centered_scaled, vec_n_missing

# a function that gets the sample name
def get_name_input(input, name_input, n_sample):
	if (name_input is None):
		vec_name_input = [os.path.splitext(os.path.basename(input))[0] + '_'+ str(i) for i in range(1, n_sample + 1)]
		print(f"using default input name: {vec_name_input}")
	else:
		df_string = pd.read_csv(name_input, header = None, sep = '\t')
		print(f"reading input name file from {name_input}")
		print(f"shape of the input name file: {df_string.shape}")

		if(df_string.shape[1] != 1):
			print(f"WARNING (INPUT): input name file must have only one column, only the first column will be used")

		if (df_string.shape[0] != n_sample):
			print(f"WARNING (INPUT): input name file must have the same number of rows as the input file, the first {n_sample} rows will be used")
		
		vec_name_input = df_string.iloc[:n_sample, 0].to_list()
	return vec_name_input

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
