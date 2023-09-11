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

scAI_SNP_classify <input_genotype_file> --<input_name_file>
