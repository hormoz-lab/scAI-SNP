import typer
import numpy as np
import pandas as pd

from scAI_SNP.math import read_center
from scAI_SNP.math import read_validate

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

	print(f"the input had {np.isnan(col_input)} NA values")
	print(f"the output had {np.isnan(col_center)} NA values")

	print(f"the input had {np.nanmean(col_input)} average")
	print(f"the output had {np.nanmean(col_center)} average")

	mat_proj_lda = pd.read_csv('data/proj_lda.tsv.gz', sep = '\t', compression = 'gzip').values
	print(f"the shape of mat_proj is {mat_proj_lda.shape}")

	return None

def cmd_classify(args=None):
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input_file')
	parsed_args = parser.parse_args(args)
	
	classify(parsed_args.input_file)

#if __name__ == '__main__':
#	cmd_classify()
