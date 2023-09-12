[![Build Status](https://travis-ci.com/hongdavid94/ancestry.svg?branch=main)](https://travis-ci.com/hongdavid94/ancestry)
[![Coverage Status](https://coveralls.io/repos/github/hongdavid94/ancestry/badge.svg?branch=main)](https://coveralls.io/github/hongdavid94/ancestry?branch=main)

# ancestry-informative SNP scAI-SNP

## installation

Because this repository includes large files (over 100MB), you may use [git-lfs](https://git-lfs.com/) to install these large files in the repository. Here is the link that can direct you to the installation [instructions](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing). Here are some helpful instructions.

### Step 0: Make sure you have the prerequisites 
- pip
- python 3.7+

### Step 1: installation of scAI-SNP

```{bash}
git clone https://github.com/hongdavid94/scAI_SNP.git
cd scAI_SNP
pip install .
```

### Step 2A (you may instead do Step 2B): installation of git-lfs

1. install the appropriate binary package in this [list](https://github.com/git-lfs/git-lfs/releases) under "Assets"
2. untar the file and move the folder to an appropriate path of your choice
3. if you don't have write access or do not prefer that the executable file be automatically installed under your directory /usr/bin, modify the install.sh file by changing its prefix to a directory of your choice (make sure this directory is on your $PATH). If you don't any issue with the installation at /usr/bin, skip this step
4. run the installation by command `./install.sh`
5. go to the directory where you have cloned the repository
6. use command `git lfs install` to apply git-lfs to the repository
7. use command `git lfs ls-files` to make sure the large files of the repository are listed in the terminal output
8. use command `git lfs pull` to convert git-lfs tagged files to their full size (this will download about ~1.2GB of memory)

### Step 2B (instead of following Step 2A): download large files using dropbox links

Use the following dropbox link to download the large files needed for the package [link](https://www.dropbox.com/sh/t8asohtbg6y8y8i/AABgztiVy4LlZ5DEwR4UZLi_a?dl=0). Anyone with the link can download the files. Make sure all the files are located probably such that
data/ and model/ files are in your scAI_SNP folder

## running the classification
### Overview
the classification will take a column or columns of 4.5 million genotypes and conduct necssary data-processing, dimension reduction, and classification prediction. More details can be found on the paper
#### data-processing
the data will be centered by the mean genotype of the training data such that all the missing genotypes will be non-informative to the model and that the other genotypes are centerred appropriatly for the subsequent steps. 
#### dimension reduction
the centered data will also be going through dimension reduction through PCA, condensing 4.5M rows of data into 1,000 rows by using the first 1000 PCs. Afterwards, LDA will be conducted to further reduce the dimensions into 25 to maximize the differences across the 26 population groups. The model was created using the training dataset and this github repository conducts the dimension reduction by multiplying the centered input by the composite projection matrix that will do both PCA and LDA simulatneously. The data will also be scaled by multiplying a scalar, inverse of present genotype data (if only 10% genotypes are present in the input, the components will be multiplied by `1/0.1` or `10` to account for missing data). 
#### classification
the reduced dimensions would then be applied to a logistic regression model that outputs the probabilities that the sample belongs to any of the 26 groups.

### Command Line Interface
```{bash}
scAI_SNP_classify <input_genotype_file> <output_directory> --name_input <input_name_file> --bool_save_plot <True or False>
```
running the command above would produce probabilities of the samples belonging into the 26 groups. The `name_input` and `bool_save_plot` are optional parameters but you must provide the `input_genotype_file`

### File Format
#### <input_genotype_file>
`input_genotype_file` must be a tab-separated text file of exactly 4.5 million (4,586,890 genotypes) rows which would correspond, in order, to the genotypes of 4.5 million SNPs [here](https://www.dropbox.com/scl/fi/65sn4qinedwsd6sh6eu4f/snp_meta_4.5M.col?rlkey=ncscgtr4p65ll46itn9fjkvy9&dl=0) of your input. There **must be no header row** and you may have multiple columns of the data in which multiple columns correspond to multiple samples. Each entry of the genotype must be `{NA, 0, 0.5, 1}`, which represents missing genotype, homozygous reference, heterozygous mutation, and homozygous mutation genotype. 

For example, for these three SNPs listed, '1:13649:G:C', '1:13868:A:G', and '1:14464:A:T', correspond to SNPs at chromosome 1 at position 13649, 13868, and 14464, respectively (using the Human genome reference GRCh38 or hg38). For a sample, if the read of the first SNP is G/G, then the genotype would be homozygous reference because both match the reference and its corresponding data value would be 0. For the second SNP, if the observed genotype is A/G, then the corresponding data value would be 0.5. And if the genotype of the third SNP is not obtainable, then the corresponding data value must be `'NA'`. To reiterate, all data values in `input_genotype_file` must be `{NA, 0, 0.5, 1}` with allowing exceptions of `{Na, na, NaN, nan}` as `NA`, `{-0, 0.0, -0.0}` as `0`, `{1.0}` as `1`.

#### <input_name_file>
`input_name_file` is an optional parameter and a text file in which you can specify the name of the sample. For each column of `input_genotype_file`, from left to right, you can write down the sample name for each row. If the number of rows in `input_name_file` and the number of columns in `input_genotype_file` do not match or if this input is not given, this parameter will be ignored and a default naming will be given. The default name would be the file name of `input_genotype_file` followed by `_#` where `#` will range from 1 to the number of samples (columns in `input_genotype_file`).

#### <bool_save_plot>
`bool_save_plot` is an optional parameter that controls where the command would create a resulting plot or not. It is recommended to generate your plot using the probability output if you have more than 8 samples as the plot will not be able to scale well with more than 8 samples. The plot will be a bar plot of probabilities of the samples belonging to the 26 population groups.

### Output
More details about the three letter population code (e.g ACB) can be found in a tab-separated file `data/meata_population_summarized.tsv`. The output will be saved in <output_directory> and if such directory does not exist, it will be made along with its parent directories. **The filenames do not change so make sure to specify a different output directory for each run of your samples**.

#### Probabilities in table
The output will be saved on your given `<output_directory>` and there are one or two files that will result from running `scAI_SNP_classify`. You will always get a probability text file `<output_directory>/probabilities.tsv` which will have 26 rows (excluding the header) of probabilities, corresponding to the alphabetically listed 26 population groups. The header row will either show the default or the input name you had provided, corresponding to the sample names. The file will have as many columns and the number of columns of the `input_genotype_file` you had provided.
#### Probabilities in barplot
The barplot will be saved on `<output_directory>/barplot_probabilities.jpg` and will show a probability bar plot. In the x-axis, represented are the 26 population groups, which are also colored based on which continent (African, American, East Asian, European, and South Asian) the group is from. As mentioned before, the number of samples should not exceed 8 as the plot does not scale well with high number of samples. You may generate your own plot using the probability text file.

## generating genotype input file

[SComatic](https://github.com/cortes-ciriano-lab/SComatic) is a github repository that enables users to extract genotypes of germline mutations (and also somatic mutations) which are required for this package. Please refer to this github link if you need its further assistance.

