import typer
import numpy as np

from scAI_SNP.math import read_center
from scAI_SNP.math import read_validate

app = typer.Typer(
	help = "Command line tool to extract genetic population classification of mutation data "
		"using HGDP project"
)

@app.command(short_help="classify the data")
def classify(input):

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



	return None
