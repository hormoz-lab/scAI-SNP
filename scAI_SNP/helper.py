import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# variables and constants
n_mut = 4586890

#@returns(int)

# a function that checks if a string is a valid path
def is_valid_path(path_str):
	invalid_chars = set('<>:|?"*')
	return bool(path_str) and not any(ch in invalid_chars for ch in path_str)

# a function that ensures a directory exists
def ensure_directory_exists(path_str):
	print(f"NOTE: Output directory '{path_str}' will be created if it does not exist.")
	if not is_valid_path(path_str):
		raise ValueError(f"'{path_str}' is not a valid path string.")

	if not os.path.exists(path_str):
		os.makedirs(path_str)
		print(f"NOTE: Output directory '{path_str}' created.")
	else:
		print(f"NOTE: Output directory '{path_str}' already exists.")

# a function that ensures a path ends with a slash
def ensure_trailing_slash(path):
    if not path.endswith('/'):
        return path + '/'
    return path

# a function that reads a file
def read_validate(file):
	print(f"reading input file from {file}")
	df = pd.read_csv(file, header = None, sep = '\t', encoding='utf-8', dtype=str)
	
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
	print(f"printing all unique values in the input:")
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
	else:
		print("SUCCESS: all input values are valid")
	
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
def center_is_valid_pathle_input(mat_input, col_mean):
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

# a function that saves the plot of the input
def save_prob_plot(df_prob, vec_name_input, n_sample, path_output):
	# data objects needed for plotting
	vec_pop_ordered = ['ACB','ASW','ESN','GWD','LWK',
					'MSL','YRI','CLM','MXL','PEL',
					'PUR','CDX','CHB','CHS','JPT',
					'KHV','CEU','FIN','GBR','IBS',
					'TSI','BEB','GIH','ITU','PJL',
					'STU']

	df_meta_unique = pd.DataFrame({
		'SUP': vec_pop_ordered,
		'POP': ['AFR'] * 7 + ['AMR'] * 4 + ['EAS'] * 5 + ['EUR'] * 5 + ['SAS'] * 5
	})

	vec_sup_full = ['African Ancestry', 
					'American Ancestry', 
					'East Asian Ancestry', 
					'European Ancestry', 
					'South Asian Ancestry']

	vec_sup = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']

	dict_color_super = {
		'AFR': '#FBB6B1',
		'AMR': '#C8C967',
		'EAS': '#47D1A0',
		'EUR': '#76D4F9',
		'SAS': '#F0A0F7'}

	colors_bar = plt.cm.viridis(np.linspace(0, 1, n_sample))
	df_prob_plot = df_prob.T[vec_pop_ordered].T
	ax = df_prob_plot.plot(
		kind = 'bar', 
		color = colors_bar,
		figsize = (20, 10))
		
	plt.title('Predicted Probabilities of the Input using Convex Optimization \n(post PCA transformation)',
		fontsize = 30)
	plt.xlabel('populations', fontsize = 20)
	plt.ylabel('probabilities', fontsize = 20)
	plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

	# get handles and labels for the subpopulation
	handles_sub, labels_sub = ax.get_legend_handles_labels()

	# create handles and labels for the population at continent level
	handles_pop = [patches.Rectangle((0,0),1,1, color=dict_color_super[region_i]) for region_i in vec_sup]
		
	handles = handles_sub + handles_pop
		
	# create legends
	## legened for the subpopulation
	legend_bar = ax.legend(handles, vec_name_input, 
						loc='upper left', fontsize=20, bbox_to_anchor = (1,1))
	legend_bar.set_title("Legend (Bar)", prop={"size": 25}) 
	ax.add_artist(legend_bar)

	## legend for the continent
	for label in ax.get_xticklabels():  
		pop_temp = label.get_text()
		color_temp = dict_color_super[df_meta_unique[df_meta_unique['SUP'] == pop_temp]['POP'].values[0]]
		label.set_bbox(dict(facecolor = color_temp, edgecolor='None', alpha=0.5))

	legend_box = ax.legend(handles_pop, vec_sup_full, 
						loc = 'lower left', bbox_to_anchor = (1, 0), fontsize = 20)
	legend_box.set_title("Legend (Population)", prop={"size": 25})
	ax.add_artist(legend_box)
	
	plt.tight_layout()
	path_plot = ensure_trailing_slash(path_output)
	plt.savefig(path_plot + 'barplot_probabilities.jpg', bbox_inches = 'tight', pad_inches = 0.5)
	# plt.show()
	print("SUCCESS: plot saved!")