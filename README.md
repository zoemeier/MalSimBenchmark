Benchmarking the simulation of malignant scRNA-seq data
------------------------------------------------------------------------
This repository contains all the code for the benchmarking framework I developed for my bachelor thesis at the Boeva Lab at ETH Zurich (July 2023).

The benchmark takes 2 .h5ad files of 2 scRNA-seq datasets (one simulated dataset and one reference dataset to compare it to), infers CNVs, calculates various cancer-specific and 'normal' cell and gene-wise metrics and plots them. Finally it assesses distributional similarity of these metrics using the Kolmogorov-Smirnov distance. 

The quality of a simulated datasets can either be assessed visually, by looking at the plots comparing the metrics of both datasets in: `plots/comparison_DATASET1_vs_DATASET2/` or by assessing the Kolmogorov-Smirnov distance, which can be found in: `data/comparison_DATASET1_vs_DATASET2/`. 

Metrics
------------------
Cancer-specific metrics:
- `cnvregion_{gain/loss}_size_{bp/gene}`: distribution of the size of consecutive regions of gain or losses respectively over all cells, meassured in bp or genes
- `cnvregion_size_{bp/gene}`: distribution of the size of all consecutive regions over all cells, measured in bp or genes.
- `total_{gains/losses}_per_{subclone/cell}_{bp/gene}`: total size of sum of all CNV regions of type gain or loss, per cell or subclone respectively, measured in bp or genes
- `num_{gain/loss}_region_per_{subclone/cell}`: number of gain or loss cnv regions per subclone or cell
- `{min/mean/median/max}_{gain/loss}_region_per_{subclone/cell}`: min (or mean, median, max) size of consecutive CNV regions of type gain or loss per cell (or subclone), measured in bp

Gene metrics:
- `mean_logCPM_{all/malignant}`: mean of gene expression over all cells or only over malignant cells respectively
- `var_logCPM_{all/malignant}` : variance of gene expression over all cells or only over malignant cells respectively
- `cv_{all/malignant}` : expression variability relative its mean (coefficient of variation) over all cells or only over malignant cells respectively
- `zeroes_cells_{all/malignant}` : gene detection frequency, i.e. the number of cells this gene is expressed in divided by the number of all cells, calculated over all cells or only over malignant cells respectively
- `gene_corr_{he/hv}_{all/malignant}`: gene-to-gene correlation over top highly expressed or highly variable genes, over all cells or over malignant cells only


Cell metrics:
- `log1p_total_counts_{all/malignant}`: log library size of all cells or only of malignant cells respectively
- `zeroes_gene_{all/malignant}` : cell detection frequency, i.e. the number of genes expressed per cell divided by the number of all genes,
        of all cells or only malignant cells 
- `cell_corr_{hv/all}_{all/malignant}`: cell-to-cell correlation over top highly variable genes or over all genes, over all cells or over malignant cells only

(Additionally: Location of the CNVs per chromosome visualized in plots)

Reproduction Guide
------------------
The data used in this benchmark is available on the Boeva Lab shared drive, together with the code. To run with your own dataset: 
1. Use the Dockerfile to create a Docker image (or alternatively install a python venv from the requirements.txt)
2. Add your two .h5ad files to `data/DATASET1/` and `data/DATASET2/` 
3. For each dataset: update parameters in the notebooks (`notebook_DATASET1`,`notebook_DATASET2`) and prepare data to fit into requirements (columnnames etc.) as described in the notebooks. 
4. Run both notebooks, it will save results and plots into the automatically generated directories
5. Update the parameters in `notebook_comparison_DATASET1_DATASET2` and run the notebook, it will save KS results and plots into the automatically generated directories. 



Project Organization
--------------------
- `data/`
    - `data/external/` externally downloaded data
    - `data/DATASET1/` (e.g. here DATASET1 = breast)
        - `data/DATASET1/interim/` folder for saving interim results needed for calculations
        - `data/DATASET1/results/` folder for saving results 
        - `data/DATASET1/FILENAME.h5ad` anndata object of scRNA-seq data to be analyzed 
    - `data/DATASET2/`
        - `data/DATASET2/interim/` folder for saving interim results needed for calculations
        - `data/DATASET2/results/` folder for saving results 
        - `data/DATASET2/FILENAME.h5ad` anndata object of scRNA-seq data to be analyzed 
    - `data/comparison_DATASET1_vs_DATASET2/`
        -  `data/comparison_DATASET1_vs_DATASET2/results` folder for results comparing the two datasets
- `benchmark/`
    - `benchmark/base/` common functions
    - `benchmark/analysis/` functions for cnv inferring & analysis
    - `benchmark/metrics/` functions for calculating various metrics
    - `benchmark/plots/` plotting functions  
- `plots/` 
    - `plots/DATASET1/` plots for dataset1      
    - `plots/DATASET2/` plots for dataset2
    - `plots/comparison_DATASET1_vs_DATASET2/` plots comparing dataset1 and dataset2     
- `notebook_DATASET1` Jupyter notebooks to calculate the metrics and produce plots for dataset1
- `notebook_DATASET2` Jupyter notebooks to calculate the metrics and produce plots for dataset2 
- `notebook_comparison_DATASET1_vs_DATASET2` Jupyter notebooks to compare the two datasets, produce similarity metrics and plots

