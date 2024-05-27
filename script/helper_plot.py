####################################################################################################
############################################# libraries ############################################
####################################################################################################

import sys
print(f"<helper_plot> version: {sys.version}")

import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# for plotting
import matplotlib.patches as patches # for plotting figure
from matplotlib.patches import Patch
import gc # for collecting garbage
import seaborn as sns

# for figures
import matplotlib.font_manager as fm
import matplotlib as mpl

####################################################################################################
########################################## basic functions #########################################
####################################################################################################

def fn_path_exists(path):
    """
    throws an error if a given path does not exist
    :param path: string object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'<fn_path_exists> the following path does not exist: {path}')
    
def fn_file_exists(file):
    """
    throws an error if a given file does not exist
    :param file: string object
    """
    fn_path_exists(file) # first check if the directory exists
    
    if not os.path.isfile(file):
        raise FileNotFoundError(f'<fn_file_exists> the following file does not exist: {file}')

def fn_ensure_slash(path):
    """
    Ensures that the input path string ends with the appropriate directory separator.

    :param path: The file path as a string.
    :return: The path string with a trailing directory separator if it was missing.
    """
    
    # make sure path exists
    fn_path_exists(path)
    
    # os.sep provides the correct directory separator based on the operating system
    if not path.endswith(os.sep):
        return path + os.sep
    return path

def title_case(text):
    """
    Converts a string to title case, capitalizing the first letter of each word except for certain
    conjunctions, prepositions, and articles.

    :param text: The string to be converted to title case.
    :return: A string in title case.
    """
    # Define a set of words that should remain lowercase unless they are the first word in the title
    lowercase_words = {
        'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up',
        'down', 'as', 'so', 'than', 'yet', 'is', 'if'
    }
    
    # Split the input text into a list of words
    words = text.split()
    
    # Capitalize the first word regardless of whether it is in lowercase_words
    # For the rest of the words, capitalize them only if they are not in lowercase_words
    new_words = [words[0].capitalize()] + [
        word if word.lower() in lowercase_words else word.capitalize() for word in words[1:]
    ]
    
    # Join the list of words back into a single string with spaces between words
    return ' '.join(new_words)

def fn_print(expression):
    """
    Evaluates a given Python expression and prints both the expression and its result.
    If an error occurs during evaluation, it prints an error message.

    :param expression: A string representing a Python expression to be evaluated.
    """
    try:
        # Evaluate the expression passed as a string and store the result.
        # The eval function interprets a string as Python code.
        # This can be dangerous if used with untrusted input.
        result = eval(expression)

        # Print the original expression and its evaluated result.
        # The f-string format is used for string interpolation.
        print(f'{expression}: {result}')
    except Exception as e:
        # If an exception occurs during the evaluation of the expression,
        # catch the exception and print an error message.
        # The exception message is included in the output.
        print(f'<fn_print> error: {e}')
        
####################################################################################################
########################################### plt setting ############################################
####################################################################################################

# Path to custom font
font_path = '../font/Arial.ttf'
fn_file_exists(font_path)
font_bold_path = '../font/Arial_Bold.ttf'
fn_file_exists(font_bold_path)

params = {'pdf.fonttype': 42}  # or pdf.fonttype: 3 for TrueType font
plt.rcParams.update(params) # this is for PDF to save text as text

# Load your custom font
fm.fontManager.addfont(font_path)
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial' # Or the exact name of the font as it appears in the font editor
font_arial = fm.FontProperties(fname = font_path)
font_arial_bold = fm.FontProperties(fname = font_bold_path)
sns.set(font = "Arial")

plt.rcParams['figure.dpi'] = 1200

size_small = 8
size_medium = 8
size_big = 8
size_font = 5

plt.rc('font', size = size_small)          # controls default text sizes
plt.rc('axes', titlesize = size_small)     # fontsize of the axes title
plt.rc('axes', labelsize = size_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = size_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize = size_small)    # fontsize of the tick labels
plt.rc('legend', fontsize = size_small)    # legend fontsize
plt.rc('figure', titlesize = size_big)     # fontsize of the figure title
plt.rc('axes', titlesize = size_big)       # fontsize of the axes title


####################################################################################################
############################################ varialbes #############################################
####################################################################################################

path_plot = '../figure/'
fn_path_exists(path_plot)

# size of plots
size_onecolumn = 3.38583
size_twocolumn = 7.00787

# size of fonts
size_font_small = 5
size_font_medium = 8
size_font_large = 12

vec_sup = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']

vec_sup_fullname = ['African Ancestry', 
                    'American Ancestry', 
                    'East Asian Ancestry', 
                    'European Ancestry', 
                    'South Asian Ancestry']

para_scatter_train = {
    'c': 'r',
    'marker': 's',
    'alpha': 0.25,
    'lw': 0,
    'label': 'train'
}

para_scatter_test = {
    'c': 'b',
    'marker': 'o',
    'alpha': 0.25,
    'lw': 0,
    'label': 'test'
}

para_scatter_test_inflated = {
    'c': 'g',
    'marker': 'v',
    'alpha': 0.25,
    'lw': 0,
    'label': 'test_inflated'
}

colors_bar = ['#B9314F', '#D5A18E', '#F0A202', '#006D77', 'purple',
              '#83C5BE', '#246EB9', 'black']

dict_color_super = {
    'AFR': '#FBB6B1',
    'AMR': '#C8C967',
    'EAS': '#47D1A0',
    'EUR': '#76D4F9',
    'SAS': '#F0A0F7'}

dict_color_subpop = {
    # Africa
    'ACB': '#fee9e8', 'ASW': '#fddbd8', 'ESN': '#fcccc8', 'GWD': '#FBB6B1',
    'LWK': '#e2a49f', 'MSL': '#c9928e', 'YRI': '#b07f7c',
    # America
    'CLM': '#e3e4b3', 'MXL': '#cece76', 'PEL': '#C8C967', 'PUR': '#b4b55d',
    # East Asia
    'CDX': '#c8f1e2', 'CHB': '#91e3c6', 'CHS': '#47D1A0', 'JPT': '#40bc90', 'KHV': '#39a780',
    # Europe
    'CEU': '#d6f2fd', 'FIN': '#ade5fb', 'GBR': '#76D4F9', 'IBS': '#6abfe0', 'TSI': '#5eaac7',
    # Southeast Asia
    'BEB': '#fae2fd', 'GIH': '#f8d0fb', 'ITU': '#F0A0F7', 'PJL': '#d890de', 'STU': '#c080c6'
}

dict_color_marker = {
    # Africa
    'ACB': 'o', 'ASW': 'v', 'ESN': '^', 'GWD': 's',
    'LWK': 'P', 'MSL': 'd', 'YRI': '*',
    # America
    'CLM': 'o', 'MXL': 'v', 'PEL': '^', 'PUR': 's',
    # East Asia
    'CDX': 'o', 'CHB': 'v', 'CHS': '^', 'JPT': 's', 'KHV': 'P',
    # Europe
    'CEU': 'o', 'FIN': 'v', 'GBR': '^', 'IBS': 's', 'TSI': 'P',
    # Southeast Asia
    'BEB': 'o', 'GIH': 'v', 'ITU': '^', 'PJL': 's', 'STU': 'P'
}

dict_sup_total = {'AFR': 7,
                  'EUR': 5,
                  'EAS': 5,
                  'SAS': 5,
                  'AMR': 4}

# meta data
n_mut = 4586890

file_meta_raw = '../data/meta_merged.csv'
fn_file_exists(file_meta_raw)
df_meta_raw = pd.read_csv(file_meta_raw)

df_meta_sorted = df_meta_raw.sort_values(['POP', 'SUP']).reset_index()
df_meta_unique = df_meta_raw[['SUP', 'POP']].drop_duplicates()
vec_label_sorted = df_meta_sorted.loc[0:3200,:][['SUP','POP']].drop_duplicates()['SUP'].values
vec_pop_ordered = vec_label_sorted

#vec_pop_ordered = ['ACB','ASW','ESN','GWD','LWK',
#                   'MSL','YRI','CLM','MXL','PEL',
#                   'PUR','CDX','CHB','CHS','JPT',
#                   'KHV','CEU','FIN','GBR','IBS',
#                   'TSI','BEB','GIH','ITU','PJL',
#                   'STU']

vec_label_super_sorted = df_meta_sorted.loc[0:3200,:][['POP']].drop_duplicates()['POP'].values

dict_super = {}
for i, row in df_meta_unique.iterrows():
    dict_super[row['SUP']] = row['POP']
    
####################################################################################################
############################################ functions #############################################
####################################################################################################

def fn_pie_category(df, col_groupby, name, size_pie = 3):    
    # Data
    first_col = df.columns[0]

    labels = list(df.groupby(col_groupby).count()[first_col].index)
    sizes = list(df.groupby(col_groupby).count()[first_col])
    
    # sort by sizes
    # Zip labels and sizes together
    zipped_lists = sorted(zip(sizes, labels))

    # Unzip the sorted lists
    sorted_sizes, sorted_labels = zip(*zipped_lists)

    # If you want the results as lists
    sizes = list(sorted_sizes)
    labels = list(sorted_labels)

    # Total for calculating percentages
    total = sum(sizes)

    # Creating a custom label list
    custom_labels = [label if (size/total)*100 >= 2 else '' for label, size in zip(labels, sizes)]
    custom_autopct = lambda p: ('%.1f%%' % p) if p >= 2 else ''

    # Create pie chart
    plt.figure(figsize=(size_pie, size_pie))  # Optional: set the figure size
    plt.pie(sizes, labels=custom_labels, autopct=custom_autopct, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(name)  # Optional: Add a title
    plt.show()
    
# Sample data
def fn_box_GTRead(df, col_group, col_y = 'GT.read', name = 'Box plot (High to low median)', 
                  angle_x = 0, name_x = '', plot_size_x = 10, plot_size_y = 6):
    
    if (name_x == ''):
        name_x = col_group
    # Order the Site categories by their median in descending order
    sorted_groups = df.groupby(col_group).median().sort_values(col_y, ascending=False).index

    # Plot
    plt.figure(figsize=(plot_size_x, plot_size_y))
    sns.boxplot(x = col_group, y = col_y, data = df, order = sorted_groups)

    # Rotate x-axis tick labels
    plt.xticks(rotation = angle_x)

    plt.title(name)
    plt.xlabel(name_x)
    plt.ylabel(col_y)
    plt.grid(axis = 'y')
    plt.show()
    
# barplot
def fn_hist_GT_count(data, name, bin_width = 1000, x_min = -1, x_max = -1, y_max = -1):
    bin_edges = np.arange(start=data.min(), stop=data.max() + bin_width, step=bin_width)

    # Plot
    plt.hist(data, bins=bin_edges)

    middle_value = data.median()
    plt.axvline(middle_value, color='red', linestyle='dotted', label='Middle')

    plt.title(f'Histogram of Mutation Reads per Sample \n({name}) (median: {round(middle_value)}) \n (binsize: {bin_width})')
    plt.ylabel('Number of Samples')
    plt.xlabel('Number of Found Sites')
    
    if (x_min != -1):
        plt.xlim(x_min, x_max)
        
    if (y_max != -1):
        plt.ylim(0, y_max)
    
    plt.show()

def get_matching_id(df_meta, col_id_ind, col_id_sample, index_pt = None, id_ind = ""):
    dict_result = {}
    
    # if id_ind is blank, then get it by index
    if ((id_ind == "") and (index_pt is not None)):
        list_pt = sorted(list(set(df_meta[col_id_ind])))
        dict_result['pt'] = list_pt[index_pt]
        dict_result['list_id'] = list(df_meta[df_meta[col_id_ind] == dict_result['pt']][col_id_sample])
    
    # otherwise, get the matching by row value
    elif ((id_ind != "") and (index_pt is None)):
        dict_result['pt'] = id_ind
        dict_result['list_id'] = list(df_meta[df_meta[col_id_ind] == dict_result['pt']][col_id_sample])
    else:
        print(f'<get_matching_id> ERROR: either index_pt or id_ind must be provided')
        return 1
    return dict_result

def get_label(df_meta, col_id_sample, sample, list_meta_imp):
    df_meta_one = df_meta[df_meta[col_id_sample] == sample][list_meta_imp]
    return '_'.join(df_meta_one.iloc[0].astype(str))

def get_df_prob(df_prob, df_meta, col_id_ind, col_id_sample, list_meta_imp, index_pt):
    result_temp = get_matching_id(df_meta, col_id_ind, col_id_sample, index_pt)
    id_subset = result_temp['list_id']
    id_patient = result_temp['pt']

    df_prob = df_prob[id_subset]
    df_prob = df_prob.loc[vec_pop_ordered]

    vec_bool = [(i in list(df_prob.columns)) for i in df_meta[col_id_sample]]
    df_meta = df_meta[vec_bool]
    df_meta = df_meta[[col_id_sample] + list_meta_imp]
    df_meta = df_meta.set_index(col_id_sample)
    
    return(df_prob, df_meta, id_patient)

def get_df_prob_two(df_prob_1, df_meta_1, col_id_ind_1, col_id_sample_1, list_meta_imp_1,
                    df_prob_2, df_meta_2, col_id_ind_2, col_id_sample_2, list_meta_imp_2,
                    index_pt = None, id_ind = ""):
    
    result_temp_1 = get_matching_id(df_meta_1, col_id_ind_1, col_id_sample_1, index_pt, id_ind)
    result_temp_2 = get_matching_id(df_meta_2, col_id_ind_2, col_id_sample_2, index_pt, id_ind)
    print(result_temp_2)
    id_subset_1 = result_temp_1['list_id']
    id_patient_1 = result_temp_1['pt']
    
    id_subset_2 = result_temp_2['list_id']
    id_patient_2 = result_temp_2['pt']

    if(len(id_subset_1) != 0):
        df_prob_1 = df_prob_1[id_subset_1]
        df_prob_1 = df_prob_1.loc[vec_pop_ordered]
        vec_bool_1 = [(i in list(df_prob_1.columns)) for i in df_meta_1[col_id_sample_1]]
        df_meta_1 = df_meta_1[vec_bool_1]
        df_meta_1 = df_meta_1[[col_id_sample_1] + list_meta_imp_1]
        df_meta_1 = df_meta_1.set_index(col_id_sample_1)
    
    if(len(id_subset_2) != 0):
        df_prob_2 = df_prob_2[id_subset_2]
        df_prob_2 = df_prob_2.loc[vec_pop_ordered]
        vec_bool_2 = [(i in list(df_prob_2.columns)) for i in df_meta_2[col_id_sample_2]]
        df_meta_2 = df_meta_2[vec_bool_2]
        df_meta_2 = df_meta_2[[col_id_sample_2] + list_meta_imp_2]
        df_meta_2 = df_meta_2.set_index(col_id_sample_2)

    if((len(id_subset_1) != 0) and (len(id_subset_2) != 0)):
        df_prob = pd.concat([df_prob_1, df_prob_2], axis = 1)
        df_meta = pd.concat([df_meta_1, df_meta_2], axis = 0, sort = False)
        id_patient = id_patient_1
    else:
        if(len(id_subset_1) != 0):
            df_prob = df_prob_1
            df_meta = df_meta_1
            id_patient = id_patient_1
        else:
            df_prob = df_prob_2
            df_meta = df_meta_2
            id_patient = id_patient_2
    return(df_prob, df_meta, id_patient)

def get_super(sub):
    return(df_meta_unique[df_meta_unique['SUP'] == sub]['POP'].iloc[0])

def get_barplot(df_prob, 
                df_meta, 
                id_patient, 
                list_meta_imp, 
                col_id_sample, 
                path_plot,
                bool_save_plot = False):
    n_sample = df_prob.shape[1]
    ax = df_prob.plot(kind = 'bar', color = plt.cm.viridis(np.linspace(0, 1, n_sample)), figsize = (16, 10))
    plt.title(f'Predicted Probabilities of patient {id_patient}', fontsize = 25)
    plt.xlabel('populations', fontsize = 20)
    plt.ylabel('probabilities', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 20)

    # get handles and labels from the old legend
    old_handles, old_labels = ax.get_legend_handles_labels()

    # create new handles for the new legend entries
    new_handles = [patches.Rectangle((0, 0), 1, 1, color = dict_color_super[region_i]) for region_i in vec_sup]
    new_labels = vec_sup_fullname

    labels_donor = [get_label(df_meta, col_id_sample, sample, list_meta_imp) 
                    for sample in df_prob.columns]
    
    # combine old and new handles and labels
    handles = old_handles + new_handles
    # labels = labels_donor + new_labels
    labels = labels_donor[:n_sample]

    # create the new legend
    legend_bar = ax.legend(handles, labels, 
                               loc = 'upper left', fontsize = 20, bbox_to_anchor = (1, 1))
    legend_bar.set_title("Legend (Bar)", prop={"size": 25}) 
    ax.add_artist(legend_bar)

    for label in ax.get_xticklabels():  
        pop_temp = label.get_text()
        color_temp = dict_color_super[df_meta_unique[df_meta_unique['SUP'] == pop_temp]['POP'].values[0]]
        label.set_bbox(dict(facecolor = color_temp, edgecolor = 'None', alpha = 0.5))  # change color as needed

    legend_box = ax.legend(new_handles, new_labels, 
                               loc = 'lower left', bbox_to_anchor = (1, 0), fontsize = 20)
    legend_box.set_title("Legend (Population)", prop={"size": 25}) 
    ax.add_artist(legend_box)

    plt.tight_layout()
    plt.show()
    if (bool_save_plot):
        plt.savefig(path_plot + id_patient + '.png', dpi = 300)
        
def get_data_barh(df_prob, df_meta, col_id_ind, col_id_sample, list_meta_imp, thres_GT = None):
    # get number of individuals for the dataset
    n_ind = len(df_meta[col_id_ind].unique())
    
    # dict_count_all: dictionary of all individuals' classification counts
    dict_count_all = {}
    
    n_sample_subset = 0
    
    # for each individaul
    for i_pt in range(n_ind):
        
        # df_prob_temp: probability matrix of 26 x n_sample
        # id_patient_temp: name of the individual
        df_prob_temp, df_meta_temp, id_patient_temp = get_df_prob(df_prob,
                                                                  df_meta,
                                                                  col_id_ind,
                                                                  col_id_sample,
                                                                  list_meta_imp,
                                                                  index_pt = i_pt)

        # subset by threshold
        if (thres_GT is None):
            vec_index_thres_pass = df_meta_temp.index
        else:
            vec_index_thres_pass = df_meta_temp[df_meta_temp['GT.read'] >= thres_GT].index
        df_prob_temp = df_prob_temp[vec_index_thres_pass]
        n_sample_subset += df_prob_temp.shape[1]
        df_meta_temp = df_meta_temp.loc[vec_index_thres_pass]
        
        # dict_count_max: count of each individual's samples' classification
        dict_count_max = Counter(df_prob_temp.idxmax())

        # df_temp: 26 x 1 dataframe of count of classification of subpopulation
        df_temp = pd.DataFrame(0, index = vec_pop_ordered, columns = [id_patient_temp])
        for key, value in dict_count_max.items():
            if key in df_temp.index:
                df_temp.loc[key, id_patient_temp] = value
        
        dict_count_all[id_patient_temp] = df_temp
    
    print(f'number of samples before subset: {df_prob.shape[1]}')
    print(f'number of samples after subset: {n_sample_subset}')
    return(pd.concat(dict_count_all.values(), axis=1).T.sort_index(), n_sample_subset)

def get_stacked_barh(df, n_sample_subset, title = ""):
    # Calculate proportions for each individual
    df = df.div(df.sum(axis=1), axis=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initial position of the left edge of the bar
    lefts = [0] * len(df)
    
    label_legend = list(dict_color_super.keys())
    color_legend = list(dict_color_super.values())
    
    for column in df.columns:
        widths = df[column]
        temp_super = get_super(column)
        
        rects = ax.barh(df.index, 
                widths, 
                left=lefts, 
                label=column, 
                color = dict_color_super[temp_super], 
                edgecolor = 'black')
        # Update the left edge for the next segment
        lefts = [left + value for left, value in zip(lefts, widths)]
        
        for rect in rects:
            width = rect.get_width()
            if width > 0:  # only if there's a visible segment
                ax.text(rect.get_x() + width/2, rect.get_y() + rect.get_height()/2,
                        column, 
                        ha='center', 
                        va='center', 
                        fontsize=10, 
                        color='white',
                       fontweight = 'bold')

    ax.set_xlabel("Proportion")
    if title == "":
        ax.set_title(f"Stacked Proportions of Predicted Subpopulation (N_Sample: {n_sample_subset})")
    else:
        ax.set_title(f"Stacked Proportions of Predicted Subpopulation (N_Sample: {n_sample_subset})\n({title})")
    ax.set_ylim(0 - 0.5, df.shape[0] - 0.5)

    legend_handles = [Patch(
        facecolor = color, 
        edgecolor='black', 
        label = label) for color, label in zip(color_legend, label_legend)]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()

# origin: plot_bone-marrow
def center_input(mat_input, col_mean):
	# count missing values in each column
	vec_n_missing = np.sum(np.isnan(mat_input), axis = 0)
	vec_inverse_present_data = 1/(1 - vec_n_missing/n_mut)
	# count the number of columns in the input
	n_col_input = mat_input.shape[1]
	## initialize centered input data
	mat_center = np.empty_like(mat_input)

	# centering the input data
	for COL in range(n_col_input):
		col_center = mat_input[:, COL] - col_mean
		mask = np.isnan(col_center)
		col_center[mask] = 0
		mat_center[:, COL] = col_center
	return mat_center

def uncenter_input(mat_input, col_mean):
	# count missing values in each column
	vec_n_missing = np.sum(np.isnan(mat_input), axis = 0)
	vec_inverse_present_data = 1/(1 - vec_n_missing/n_mut)
    
	# count the number of columns in the input
	n_col_input = mat_input.shape[1]
    
	## initialize centered input data
	mat_uncenter = np.empty_like(mat_input)

	# centering the input data
	for COL in range(n_col_input):
		col_uncenter = mat_input[:, COL] + col_mean
		mask = (mat_input[:, COL] == 0)
		col_uncenter[mask] = np.nan
		mat_uncenter[:, COL] = col_uncenter
	return np.around(mat_uncenter, decimals = 2)

####################################################################################################
############################################# reviewed #############################################
####################################################################################################

def calculate_scale_factor(fig, ax):
    """
    Calculate scale factor to convert desired bar height in inches to data units.
    :param ax: Matplotlib Axes object.
    :param fig: Matplotlib Figure object.
    :return: Scale factor to be used as bar height in data units.
    """
    # Get the size of the figure in inches and the limits of the y-axis
    fig_height_inch = fig.get_size_inches()[1]
    ylim = ax.get_ylim()
    yrange_data_units = ylim[1] - ylim[0]

    # Calculate the total height available for all bars in inches
    total_height_available_inch = fig_height_inch

    # Calculate scale factor
    scale_factor = 1 / total_height_available_inch * yrange_data_units
    return scale_factor

def adjust_left_margin(fig, ax, fixed_margin=0.2):
    """
    Adjust the left margin of the plot area to a fixed size.
    :param fig: Matplotlib Figure object.
    :param ax: Matplotlib Axes object.
    :param fixed_margin: Fixed margin size as a fraction of figure width.
    """
    fig_width, fig_height = fig.get_size_inches()
    left_margin = fixed_margin / fig_width  # Convert to fraction of figure width

    # Get current axes position in figure coordinates
    pos = ax.get_position()

    # Adjust left position of the axes
    ax.set_position([left_margin, pos.y0, pos.width - left_margin, pos.height])

def format_to_three_significant_digits(number):
    """
    Formats a number to have three significant digits.
    :param number: The number to format.
    :return: A string representation of the number with three significant digits.
    """
    # Check if the number is zero
    if number == 0:
        return "0.00"

    # Determine the number of digits before the decimal point
    int_part_length = len(str(int(abs(number))))

    # If there are more than three digits before the decimal point, format as a regular number
    if int_part_length >= 3:
        return f"{number:.0f}"

    # Otherwise, format to have three significant digits
    decimal_places = 3 - int_part_length
    format_string = "{:." + str(decimal_places) + "f}"
    return format_string.format(number)

def get_barhplot(df_prob_input, df_meta_input, label_tick = 'GT.read',
                 min_percent_visible_pop = 0.09,
                 bool_sort = False, type_GTread = "percent",
                 bool_save_plot = False, labelpad_y = 50,
                 size_plot_x = 3, size_plot_y = 1.5, 
                 name_plot = "", name_plot_i = "", path_plot = ""):
    """
    Generates a horizontally stacked bar plot from the given dataframes and saves the plot if required.

    :param df_prob_input: DataFrame (sample x 26 populations) containing probability data for plotting.
    :param df_meta_input: DataFrame (sample x metadata) containing metadata associated with the probability data.
    :param label_tick: Column name in df_meta_input to use for y-axis tick labels.
    :param min_percent_visible_pop: minimum percentage that shows the population label
    :param bool_sort: boolean indicating whether to sort the samples by label_tick
    :param type_GTread: type of GTread which is either percent or integer
    :param bool_save_plot: Boolean indicating whether to save the plot to a file.
    :param labelpad_y: Padding for the y-axis labels (not used in this function).
    :param size_plot_x: Width of the plot figure.
    :param size_plot_y: Height of the plot figure.
    :param name_plot: Base name for the saved plot file.
    :param name_plot_i: Additional identifier for the saved plot file.
    :param path_plot: Path where the plot file will be saved.
    """
    # Check if the indices of both input dataframes are equal
    test_bool_1 = (df_prob_input.index == df_meta_input.index).all()
    if (not(test_bool_1)):
        print(f'<get_barhplot> WARNING: index of df_prob_input not df_meta_input equal')
    
    # Identify rows that do not contain all NA values
    vec_bool_row_not_NA = ~df_prob_input.isna().all(axis = 1)
    n_row_NA = df_prob_input.shape[0] - sum(vec_bool_row_not_NA)
    if (n_row_NA != 0):
        print(f'<get_barhplot> WARNING: number of NA rows: {n_row_NA}')
    
    # Filter out rows with all NA values
    df_prob_input = df_prob_input[vec_bool_row_not_NA]
    df_meta_input = df_meta_input[vec_bool_row_not_NA]
    
    # reverse order
    df_meta_input = df_meta_input.iloc[::-1]
    df_prob_input = df_prob_input.iloc[::-1]
    
    # sort df_prob_input and df_meta_input by GT.read
    if (bool_sort):
        vec_index_sorted = df_meta_input.sort_values('GT.read').index
        df_prob_input = df_prob_input.loc[vec_index_sorted]
        df_meta_input = df_meta_input.loc[vec_index_sorted]
    
    # number of samples
    n_sample = df_prob_input.shape[0]
    
    # Create a figure and an axes
    fig, ax = plt.subplots(figsize=(size_plot_x, size_plot_y))
    adjust_left_margin(fig, ax, fixed_margin=0.2)

    # Set the background color
    fig.set_facecolor('white')  # Set the figure background to white
    ax.set_facecolor('white')   # Set the axes background to white
    
    # Initial position of the left edge of the bar
    df = df_prob_input
    lefts = [0] * len(df)
    
    # set limits of x, y-axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0 - 0.5, df.shape[0] - 0.5)
    
    # calculate the scale factor and adjust bar height
    scale_factor = calculate_scale_factor(fig, ax)
    height_bar = 0.12 * scale_factor
    vec_coord_y = np.arange(n_sample) * height_bar * 1.25

    # loop through each column and plot it
    for i_sample, column in enumerate(df.columns):
        widths = df[column]
        temp_super = get_super(column)
        rects = ax.barh(vec_coord_y, widths, left = lefts, height = height_bar, label=column, 
                        color = dict_color_super[temp_super], edgecolor = 'black')
        
        # Update the left edge for the next segment
        lefts = [left + value for left, value in zip(lefts, widths)]
        
        # add text labels to the bars if the segment is wide enough
        for rect in rects:
            width = rect.get_width()
            if width > min_percent_visible_pop:  # only if there's a visible segment
                ax.text(rect.get_x() + width/2, rect.get_y() + rect.get_height()/2,
                        column, ha = 'center', va = 'center', fontsize = size_font, 
                        color = 'white')
    
    # Add percentage labels to the end of the bars
    for idx, value in enumerate(lefts):
        label = df_meta_input['GT.read'].iloc[idx]
        
        if (type_GTread == "percent"):
            ax.text(value + 0.01, vec_coord_y[idx], f'{label}%', ha = 'left', va = 'center', fontsize = size_font, 
                    color = 'black')
        
        elif (type_GTread == "integer"):
            ax.text(value + 0.02, idx, f'{round(100 * label/n_mut, 2)}%',
                    ha = 'left', va = 'center', fontsize = size_font, color = 'black')
        else:
            print(f"<get_barhplot> type_GTread must be percent or integer")
            return(None)
    
    # Set the y-axis tick positions and labels
    ax.set_yticks(vec_coord_y)
    ax.set_yticklabels(list(df_meta_input[label_tick]), fontsize = size_font)
    ax.set_xticks([])
    
    # Adjust the padding for the tick labels
    ax.tick_params(axis = 'y', which = 'major', pad = 0.01)
    ax.tick_params(axis = 'x', which = 'major', pad = 0.01)
    
    # adjust the position of x-axis tick labels
    for label in ax.get_yticklabels():
        label.set_x(label.get_position()[0])
    
    # Hide the top, right, and bottom spines (borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Ensure the layout fits well within the figure size
    plt.tight_layout()
    
    # Save the plot to a file if requested
    if (bool_save_plot):
        # check if path exists
        fn_path_exists(path_plot)
        plt.savefig(f'{fn_ensure_slash(path_plot)}{name_plot}_bar_horizontal_{name_plot_i}.pdf', 
                    format = 'pdf', dpi = 1200, bbox_inches = 'tight')
        plt.savefig(f'{fn_ensure_slash(path_plot)}{name_plot}_bar_horizontal_{name_plot_i}.jpeg', 
                    format = 'jpeg', dpi = 1200, bbox_inches = 'tight')

def plot_pie_chart(name_sample, vec_freq, name_plot, 
                   size_pie=0.9, size_font=5, 
                   bool_save_plot=False, path_plot=""):
    """
    Plots a pie chart for the given frequency data and saves the plot if required.

    :param name_sample: The name of the sample being plotted.
    :param vec_freq: A pandas Series containing frequency data for each population.
    :param name_plot: The base name for the saved plot file.
    :param size_pie: The size of the pie chart figure.
    :param size_font: The font size for text within the pie chart.
    :param bool_save_plot: Boolean indicating whether to save the plot to a file.
    :param path_plot: Path where the plot file will be saved.
    """
    
    # Create a figure with specified size
    plt.figure(figsize=(size_pie, size_pie))
    
    # Filter out categories with zero frequency
    vec_freq_filtered = vec_freq[vec_freq > 0]
    
    # Generate colors for each category based on a predefined dictionary (dict_color_super)
    # and a mapping from category to super-category (dict_super)
    colors = [dict_color_super[dict_super[category]] for category in vec_freq_filtered.index]
        
    # Plot the pie chart with the filtered frequency data
    plt.pie(vec_freq_filtered, labels=None, startangle=0,
            textprops={'fontsize': size_font},
            wedgeprops=dict(width=0.6),
            colors=colors)

    # Add labels inside the pie chart
    total = vec_freq_filtered.sum()
    if vec_freq_filtered.shape[0] == 1:
        # If there's only one category, place the label in the center
        angle = np.pi / 2
        x = 0.7 * np.cos(angle)
        y = 0.7 * np.sin(angle)
        plt.text(x, y, vec_freq_filtered.index[0], ha='center', va='center', 
                 fontsize=size_font, color='white')
    else:
        # For multiple categories, calculate the position for each label
        for i, (category, value) in enumerate(vec_freq_filtered.items()):
            angle = sum(vec_freq_filtered[:(i + 1)]) / total - (vec_freq_filtered[i] / total) * 0.5
            angle = 2 * np.pi * angle
            x = 0.7 * np.cos(angle)
            y = 0.7 * np.sin(angle)
            plt.text(x, y, category, ha='center', va='center', 
                     fontsize=size_font, color='white')

    # Add a center text with the total number of samples
    plt.text(0, 0, f'n = {total}', ha='center', va='center', fontsize=size_font)
    
    # Ensure the pie chart is circular
    plt.axis('equal')
    
    # Save the plot to a file if requested
    if bool_save_plot:
        # Ensure the path exists
        fn_path_exists(path_plot)

        # Ensure the path ends with a slash
        path_plot_with_slash = fn_ensure_slash(path_plot)

        # Save the plot as a PDF
        plt.savefig(f'{path_plot_with_slash}{name_plot}_pie_{name_sample}.pdf', 
                    format='pdf', bbox_inches='tight')
        # Save the plot as a high-resolution JPEG
        plt.savefig(f'{path_plot_with_slash}{name_plot}_pie_{name_sample}.jpeg', 
                    format='jpeg', dpi=1200, bbox_inches='tight')
    
    # Display the plot
    plt.show()