# Genotyping with cellsnp-lite and publicly available data and 

After using the given input file for installation verification, users may refer to this document to see how the repository could be used. This documentation will be using publically available data from 10X Genomics [website](https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-targeted-immunology-panel-3-1-standard-4-0-0). For this example, we will be looking at a cellranger output from a healthy donor (more details about the sample is in the provided link)

## Cellsnp-lite 

As stated in the github page, installation for this package is quite simple, and it is recommended to use conda to set up using cellsnp-lite.

```{bash}
conda install -c bioconda cellsnp-lite
```

Here is an example script that uses cellsnp-lite to genotype the data by providing the 4.5 million SNP information as a `vcf` file. As this sample's chromosome annotation follows the `chr#` format, we will be using `snp_4.5M_w_chr.vcf.gz` provided in the repository.

```{bash}
#!bin/bash
#https://cellsnp-lite.readthedocs.io/en/latest/main/manual.html

file_bam=../input/Targeted_NGSC3_DI_PBMC_Immunology_possorted_genome_bam.bam
file_vcf=../input/snp_4.5M_w_chr.vcf.gz
path_output=../output/

# parameters
# --minMAF 0 (Minimum Minor Allele Frequency)
# this should be set to 0 so that we can capture homozygous mutation (wild type)

# --minCOUNT 20 (Minimum UMI Count)
# Only keeps SNPs that have at least 20 UMIs (or reads) covering that position
# This parameter ensures sufficient depth for reliable genotype calling and can be adjusted for the user's need

# -p 16 (Number of Threads)
# depending on the user's resource, the number of threads can be set to parallize genotyping task

cellsnp-lite -s $file_bam -O $path_output -R $file_vcf -p 30 --minMAF 0 --cellTAG None --minCOUNT 20 --genotype --gzip
```

## Converting cellsnp-lite results to be compatible with scAI-SNP

The output `cellSNP.cells.vcf.gz` includes genotype result for each given SNP. A user may use different text-wrangling softwares such as python or R for this process. If `bcftools` is available, a user can refer to the following script to extract genotypes from cellsnp-lite output.

```{bash}
#!/bin/bash

file_input='../input/cellSNP.cells.vcf.gz'
file_output='../output/cellSNP_for_scAI-SNP_temp.tsv'

bcftools view -H ${file_input} | awk '
BEGIN { OFS="\t" } 
{
    if ($0 !~ /^#/) {
        split($9, format, ":");
        split($10, sample, ":");
        for (i in format) {
            if (format[i] == "GT") {
                print $1, $2, $4, $5, sample[i];
                break;
            }
        }
    }
}' > $file_output
```

The content of this file (which has found 4,859 genotypes) will look like the following
```{bash}
(env_cellsnp) [shong@leo output]$ head -n 5 cellSNP_for_scAI-SNP_temp.tsv
chr1    1014228 G       A       1/0
chr1    1014274 A       G       1/1
chr1    1014545 C       T       1/1
chr1    1203533 T       C       0/0
chr1    1203822 T       C       0/0
```

As a next step, the user must fill in `NA` values for SNP genotypes not observed in the correct order. Python or R is better suited for this process (i.e. left-join this output to the table of 4.5 million SNP table). And lastly, convert 0/0 to 0, 0/1 or 1/0 to 0.5, and 1/1 to 1.