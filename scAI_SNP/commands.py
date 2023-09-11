import sys
import typer
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scAI_SNP.helper import read_center
from scAI_SNP.helper import read_validate
from scAI_SNP.helper import center_scale_input
from scAI_SNP.helper import get_name_input
from scAI_SNP.helper import safe_sparse_dot

n_mut = 4586890

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

app = typer.Typer(
	help = "Command line tool to extract genetic population classification of mutation data "
		"using HGDP project"
)

@app.command(short_help="classify the data")
def classify(file_input, name_input = None, bool_save_plot = True):
	now = time.time()
	print(f"starting ancestry classification for scAI-SNP")
	print(f"python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
	print(f"input file path: {file_input}")	

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

	# script when the input can only be one column
	# centered column that ignores NA (np.nan)
	# col_center = col_input - col_mean
	# mask = np.isnan(col_center)
	# col_center[mask] = col_mean[mask]
	

	print(f"input has {n_sample} columns")
	print(f"input has {vec_n_missing} NA values")
	print("SUCCESS: centering complete!")
	
	print("reading LDA projection matrix...")
	mat_proj_lda = pd.read_csv('data/proj_lda.tsv.gz', sep = '\t', compression = 'gzip', header = None).values
	print("SUCCESS: projection matrix loaded!")
	print("applying classification...")
	
	lda_proj_input = mat_input_centered_scaled.T @ mat_proj_lda
	
	coef_lr = pd.read_csv('model/LR/LR_coef.tsv', sep = '\t', header = None).to_numpy()
	intercept_lr = pd.read_csv('model/LR/LR_intercept.tsv', sep = '\t', header = None).to_numpy()
	vec_population = pd.read_csv('model/LR/population.tsv', header = None).to_numpy().flatten()
	
	scores = lda_proj_input @ coef_lr.T + intercept_lr.T
	# scores = scores.to_numpy()
	print(f"scores dimension: {scores.shape}")
	index_max = np.argmax(scores, axis = 1)
	prob = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)

	for sample in range(n_sample):
		print(f"For sample {sample + 1}:")
		print(f"classified population: {vec_population[index_max[sample]]}")
		print(f"prob: {prob[sample, index_max[sample]]}")

	print("SUCCESS: classification done!")
	print(f"classify took {round((time.time() - now)/60, 2)} minutes")
	print("plotting and saving probabilities...")

	if (n_sample == 1):
		prob = prob.reshape(1, -1)

	df_prob = pd.DataFrame(prob.T, index = vec_population, columns = vec_name_input)
	print(f"saving probabilities to output/probabilities.tsv")
	df_prob.to_csv(sep = '\t', path_or_buf = 'output/probabilities.tsv', index = True)
	print("SUCCESS: probabilities saved!")

	if (bool_save_plot):
		colors_bar = plt.cm.viridis(np.linspace(0, 1, n_sample))
		df_prob_plot = df_prob.T[vec_pop_ordered].T
		ax = df_prob_plot.plot(
			kind = 'bar', 
			color = colors_bar,
			figsize = (20, 10))
		
		plt.title('Population Probabilities of the Input using LR classification \n(post PCA-LDA transformation)',
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
		path_plot = 'output/figure/'
		plt.savefig(path_plot + 'probabilities.jpg', bbox_inches = 'tight', pad_inches = 0.5)
		plt.show()
		print("SUCCESS: plot saved!")
	return df_prob


def cmd_classify(args=None):
	import argparse
	parser = argparse.ArgumentParser(
		description = "Command line tool to extract genetic population classification of mutation data"
	)
	
	# required arguments
	parser.add_argument('file_input', help = "input genotype file path")
	
	# optional arguments
	parser.add_argument('--name_input', default = None, help= "input sample names (file path) (default: None).")
	parser.add_argument('--bool_save_plot', type = bool, default=True, help = "Flag to save plot (default: True).")

	parsed_args = parser.parse_args(args)
	
	classify(parsed_args.file_input, parsed_args.name_input, parsed_args.bool_save_plot)

#import os
#path_working = '/homes1/shong/tool/scAI_SNP/'
#os.chdir(path_working)
#print(os.getcwd())

#vec_input_uncenter = ['210521_white_MNC', '211008_asian_MNC', '211102_indian_old', '220111_asian_MNC', 'ET1_scRNA_190114', 'ET1_vcf', 'ET1_WGS', 'ET2_scRNA_190311_old', 'ET2_WGS']
#for sample in vec_input_uncenter:
#	uncenter('/homes1/shong/tool/scAI_SNP/data/vec_bone-marrow_4.5M/genotype_' + sample + '.col')

# classify('/homes1/shong/tool/scAI_SNP/data/genotype_test_mixed.col')
# classify('/homes1/shong/tool/scAI_SNP/data/genotype_two_col.tsv')
#result = classify(
#	'/homes1/shong/tool/scAI_SNP/data/genotype_combined_v4.tsv',
#	name_input = '/homes1/shong/tool/scAI_SNP/data/sample_input_name.txt')
# classify('/homes1/shong/tool/scAI_SNP/data/vec_bone-marrow_4.5M_uncentered/genotype_210521_white_MNC.col')