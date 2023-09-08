[![Build Status](https://travis-ci.com/hongdavid94/ancestry.svg?branch=main)](https://travis-ci.com/hongdavid94/ancestry)
[![Coverage Status](https://coveralls.io/repos/github/hongdavid94/ancestry/badge.svg?branch=main)](https://coveralls.io/github/hongdavid94/ancestry?branch=main)

# ancestry-informative SNP scAI-SNP

## installation

Because this repository includes large files (over 100MB), using scAI-SNP requires installation of [git-lfs](https://git-lfs.com/). Here is the link that can direct you to the installation [instructions](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing). Here are some helpful instructions.

### installation of git-lfs
#### linux

1. install the appropriate binary package in this [list](https://github.com/git-lfs/git-lfs/releases) under "Assets"
2. untar the file and move the folder to an appropriate path of your choice
3. if you don't have write access or do not prefer that the executable file be automatically installed under your directory /usr/bin, modify the install.sh file by changing its prefix to a directory of your choice (make sure this directory is on your $PATH). If you don't any issue with the installation at /usr/bin, skip this step
4. run the installation by command `./install.sh`
5. go to the directory where you have cloned the repository
6. use command `git lfs install` to apply git-lfs to the repository
7. use command `git lfs ls-files` to make sure the large files of the repository are listed in the terminal output
8. use command `git lfs pull` to convert git-lfs tagged files to their full size (this will download about ~1.2GB of memory)

### installation of scAI-SNP

1. 

pip install scAI_SNP

or

git clone
cd scAI_SNP
pip install .

## running the classification

scAI_SNP_classify <input_genotype_file> <input_name_file>
