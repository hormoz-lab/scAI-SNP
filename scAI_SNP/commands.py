import sys
import typer
import numpy as np
import pandas as pd
import cvxpy as cp
import time
import pyarrow

from scAI_SNP.helper import (read_center, read_validate, center_scale_input, get_name_input, save_prob_plot,
							 ensure_directory_exists, n_mut, is_valid_path, ensure_trailing_slash)


app = typer.Typer(
	help = "Command line tool to extract genetic population classification of mutation data "
		"using HGDP project"
)

@app.command(short_help="classify the data")
def classify(file_input, path_output, name_input = None, bool_save_plot = True):
	now = time.time()
	print(f"starting ancestry classification for scAI-SNP")
	print(f"python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
	print(f"input file path: {file_input}")

	# ensure the output directory exists and creates it if needed
	ensure_directory_exists(path_output)

	# read the input data
	mat_input = read_validate(file_input)
	
	# count the number of columns in the input
	n_sample = mat_input.shape[1]
	print(f"found {n_sample} inputs in the input file")

	# get the sample name
	vec_name_input = get_name_input(file_input, name_input, n_sample)

	# read the mean genotype of the training data
	col_mean = read_center()

	# centering the input data and count the number of missing values
	mat_input_centered_scaled, vec_n_missing = center_scale_input(mat_input, col_mean)

	print(f"input has {n_sample} samples (columns)")
	print(f"each sample is missing {np.around(vec_n_missing/n_mut * 100, 2)} % genotypes")
	print("SUCCESS: centering complete!") 
	print("reading mean PCA matrix...")
	n_PC = 600
	mat_mean_PC = pd.read_csv(f'data/mat_GT_PCA_projected_mean.tsv', sep = '\t', header = None).values
	mat_mean_PC = mat_mean_PC[0:n_PC,:]	
	print(f'shape of mat_mean_P ({n_PC} by 26): {mat_mean_PC.shape}')
	print("SUCCESS: mean PCA matrix loaded!")

	list_pca_projected_input = []
	now = time.time()
	print(f'version of pyarrow: {pyarrow.__version__}')
	for index_PC in range(1, int(int(n_PC/100) + 1)):
		print("reading PCA projection matrix...")
		mat_proj_pca = pd.read_parquet(
			f'data/proj_PCA/mat_proj_PCA_cc{index_PC}_2s.parquet', 
			engine = 'pyarrow').values
		print(f'shape of mat_proj_pca (4.5M by 100): {mat_proj_pca.shape}')
		print(f"SUCCESS: PCA projection matrix loaded! ({index_PC}/{int(int(n_PC/100))})")	
	
		print("applying PCA...")
		print(f'shape of mat_input_centered_scaled (4.5M by n_sample): {mat_input_centered_scaled.shape}')
		list_pca_projected_input.append(mat_input_centered_scaled.T @ mat_proj_pca)
		print(f"SUCCESS: PCA applied for ({index_PC}/{int(n_PC/100)})")
		print(f"PCA ({index_PC}/{int(n_PC/100)}) took {round((time.time() - now)/60, 2)} minutes")

	pca_projected_input = np.concatenate(list_pca_projected_input, axis = 1)
	print(f"shape of pca_projected_input: {pca_projected_input.shape}")

	vec_population = pd.read_csv('model/population.tsv', header = None).to_numpy().flatten()
	 
	def predict_convex(test_vec, mean_vectors = mat_mean_PC):
		# Define the optimization variable
		X = cp.Variable(26)
		# Define the objective function (minimize the least squares error)
		objective = cp.Minimize(cp.norm(mean_vectors @ X - test_vec, 'fro'))
		# Define the constraints
		constraints = [X >= 0, cp.sum(X) == 1]

		# Define and solve the problem
		problem = cp.Problem(objective, constraints)
		problem.solve()
		this_predicted = X.value

		# Use this_predicted to construct a prediction vector from the mean_vectors
		vec_predicted = mean_vectors @ this_predicted

		# Compute the cosine similarity between this_vector and test_vec
		cos_sim = np.dot(vec_predicted, test_vec) / (np.linalg.norm(vec_predicted) * np.linalg.norm(test_vec))  # cosine similarity is computed using this single line of code ***

		return this_predicted, cos_sim

	print("applying classification...")
	list_predicted = []
	list_angle = []
	for index_sample in range(n_sample):
		vec_prob, angle = predict_convex(pca_projected_input[index_sample,:].T)
		list_predicted.append(vec_prob)
		list_angle.append(angle)
	
	print("SUCCESS: classification done!")
	print(f"classify took {round((time.time() - now)/60, 2)} minutes")
	print("Saving probabilities...")

	if (n_sample == 1):
		prob = vec_prob.reshape(1, -1)
	else:
		prob = np.array(list_predicted)

	df_prob = pd.DataFrame(prob.T, index = vec_population, columns = vec_name_input)
	print(f"saving probabilities to {ensure_trailing_slash(path_output)}probabilities.tsv")
	df_prob.to_csv(sep = '\t', path_or_buf = ensure_trailing_slash(path_output) + 'df_probabilities.tsv', index = True)
	print("SUCCESS: probabilities saved!")
	pd.DataFrame(
     {'angle': list_angle}, index = vec_name_input).to_csv(
         ensure_trailing_slash(path_output) + 'cos_angle.tsv',
         index = True, sep = '\t')

	if (bool_save_plot):
		save_prob_plot(df_prob, vec_name_input, n_sample, path_output)
	return df_prob


def cmd_classify(args=None):
	import argparse
	parser = argparse.ArgumentParser(
		description = "Command line tool to extract genetic population classification of mutation data"
	)
	
	# required arguments
	parser.add_argument('file_input', help = "input genotype file path")
	parser.add_argument('path_output', help = "output genotype folder path")
	
	# optional arguments
	parser.add_argument('--name_input', default = None, help= "input sample names (file path) (default: None).")
	parser.add_argument('--bool_save_plot', type = bool, default = True, help = "Flag to save plot (default: True).")

	parsed_args = parser.parse_args(args)
	
	classify(parsed_args.file_input, parsed_args.path_output, parsed_args.name_input, parsed_args.bool_save_plot)
