import typer
import numpy as np
import pandas as pd

from scAI_SNP.math import read_center
from scAI_SNP.math import read_validate
from scAI_SNP.math import safe_sparse_dot


app = typer.Typer(
	help = "Command line tool to extract genetic population classification of mutation data "
		"using HGDP project"
)

@app.command(short_help="classify the data")
def classify(input):

	print(f"input file name: {input}")

	col_mean = read_center()
	col_input = read_validate(input)
	
	# centered column that ignores NA (np.nan)
	col_center = col_input - col_mean
	mask = np.isnan(col_center)
	col_center[mask] = col_mean[mask]

	print(f"input has {np.sum(np.isnan(col_input))} NA values")
	print(f"centered input has {np.sum(np.isnan(col_center))} NA values")

	print(f"input has {np.nanmean(col_input)} average")
	print(f"centered input has {np.nanmean(col_center)} average")
	print(f"SUCCESS: centering complete!")

	mat_proj_lda = pd.read_csv('data/proj_lda.tsv.gz', sep = '\t', compression = 'gzip', header = None).values
	print(f"the shape of mat_proj is {mat_proj_lda.shape}")

	lda_proj_input = col_center.T @ mat_proj_lda
	print(f"the shape of projected input is {lda_proj_input.shape}")
	print(f"average is {np.mean(lda_proj_input, axis = 0)}")
	print(f"average is {np.mean(lda_proj_input, axis = 1)}")
	
	coef_lr = pd.read_csv('model/LR/LR_coef.tsv', sep = '\t', header = None).values
	intercept_lr = pd.read_csv('model/LR/LR_intercept.tsv', sep = '\t', header = None).values
	
	print(f"shape of coef_lr: {coef_lr.shape}")
	print(f"shape of intercept_lr: {intercept_lr.shape}")
	
	scores = safe_sparse_dot(coef_lr, lda_proj_input.T, dense_output = True) + intercept_lr
	decision = scores.reshape(1, -1)

	print(f"shape of scores {scores.shape}")
	print(f"shape of decision {decision.shape}"
	
	#return(None)

	#return {'probabilities': ,
	#	'population': softmax(decision, copy = False}

def cmd_classify(args=None):
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input_file')
	parsed_args = parser.parse_args(args)
	
	classify(parsed_args.input_file)

#if __name__ == '__main__':
#	cmd_classify()
