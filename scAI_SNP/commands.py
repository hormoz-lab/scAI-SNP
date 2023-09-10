import sys
import typer
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from scAI_SNP.helper import read_center
from scAI_SNP.helper import read_validate
from scAI_SNP.helper import safe_sparse_dot


app = typer.Typer(
	help = "Command line tool to extract genetic population classification of mutation data "
		"using HGDP project"
)

@app.command(short_help="classify the data")
def classify(input):
	now = time.time()
	print(f"starting ancestry classification for scAI-SNP")
	print(f"python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
	print(f"input file name: {input}")

	col_mean = read_center()
	mat_input = read_validate(input)

	# count the number of columns in the input
	n_col_input = mat_input.shape[1]
	## initialize centered input data	
	mat_center = np.empty_like(mat_input)
	
	for COL in range(n_col_input):
		col_center = mat_input[:, COL] - col_mean
		mask = np.isnan(col_center)
		col_center[mask] = col_mean
		mat_center[:, COL] = col_center


	# script when the input can only be one column
	# centered column that ignores NA (np.nan)
	# col_center = col_input - col_mean
	# mask = np.isnan(col_center)
	# col_center[mask] = col_mean[mask]

	print(f"input has {n_col_input} columns")
	print(f"input has {np.sum(np.isnan(mat_input), axis = 0)} NA values")
	print("SUCCESS: centering complete!")
	
	print("reading LDA projection matrix...")
	mat_proj_lda = pd.read_csv('data/proj_lda.tsv.gz', sep = '\t', compression = 'gzip', header = None).values
	print("SUCCESS: projection matrix loaded!")
	print("applying classification...")	

	lda_proj_input = mat_center.T @ mat_proj_lda
	
	coef_lr = pd.read_csv('model/LR/LR_coef.tsv', sep = '\t', header = None).to_numpy()
	intercept_lr = pd.read_csv('model/LR/LR_intercept.tsv', sep = '\t', header = None).to_numpy()
	vec_population = pd.read_csv('model/LR/population.tsv', header = None).to_numpy()
	
	scores = lda_proj_input @ coef_lr.T + intercept_lr.T
	scores = scores.to_numpy()
	print(f"scores dimension: {scores.shape}")
	index_max = np.argmax(scores[0])
	prob = np.exp(scores[0]) / np.sum(np.exp(scores[0]), axis = 0)

	print(f"population: {vec_population[index_max]}")
	print(f"max index: {index_max}")
	print(f"prob: {prob[index_max]}")

	print("SUCCESS: classification done!")
	print(f"classify took {round((time.time() - now)/60, 2)} minutes")
	print("plotting and saving probabilities...")

	df_prob = pd.DataFrame(prob.reshape(1, -1), columns = vec_population)
	df_prob.T.plot(kind = 'bar', color = ['red'], figsize = (16, 10))
	plt.legend(['Input'], loc = 'upper right',
		fontsize = 20)
	plt.title('Population Probabilities of the Input using LR classification \n(post PCA-LDA transformation)',
		fontsize = 25)
	plt.xlabel('populations', fontsize = 20)
	plt.ylabel('probabilities', fontsize = 20)
	plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

	path_plot = 'output/figure/'
	plt.savefig(path_plot + 'probabilities.jpg')
	plt.show()
	print("SUCCESS: plot saved!")
	return None


def cmd_classify(args=None):
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input_file')
	parsed_args = parser.parse_args(args)
	
	classify(parsed_args.input_file)
