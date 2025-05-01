#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os 
from scipy import stats
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse  # pytype: disable=import-error
import benchmark.base.utils as utils


def set_num_genes_by_counts(adata):
    '''
    Calculates number of genes expressed per cell and writes result to adata.obs in column 'n_genes_by_counts'.
    
    Args: 
        adata: anndata object (annotated data matrix) with raw counts
    '''
    
    arr_start = adata.X.indptr[:-1]
    arr_end = adata.X.indptr[1:]
    num_genes = arr_end - arr_start 
    adata.obs['n_genes_by_counts'] = num_genes
    
def set_zeroes_genes(adata):
    '''
    Calculates cell detection frequency, i.e. the number of genes expressed per cell divided by the number of all genes
    and writes result to adata.var in column 'zeroes_genes'. (= Inverse of proportion of zeroes per cells) 
    
    Args: 
        adata: anndata object (annotated data matrix) with raw counts
    '''
   
    total_genes = len(adata.var)
    adata.obs['zeroes_genes'] = adata.obs['n_genes_by_counts'] / total_genes
    

def get_cell_cell_correlation_all(adata): 
    '''
    Calculates cell-to-cell correlation using Pearson correlation for 1000 cells over all genes.
    
    Args: 
        adata: normalized anndata object (annotated data matrix)
    
    Returns: 
        Upper triangle of correlation matrix as array
    '''
    
    sampled = sc.pp.subsample(adata, n_obs = 1000, copy = True)
    sampled_count_matrix = pd.DataFrame.sparse.from_spmatrix(sampled.X) 
    full_cell_corr = sampled_count_matrix.transpose().corr()
    
    f_cell_corr_tri = full_cell_corr.where(np.triu(np.ones(full_cell_corr.shape)).astype(bool))
    f_cell_corr_arr = f_cell_corr_tri.values.flatten()
    f_cell_corr_arr = f_cell_corr_arr[~np.isnan(f_cell_corr_arr)]
    return f_cell_corr_arr
    
def get_cell_cell_correlation_hv(adata): 
    '''
    Calculates cell-to-cell correlation using Pearson correlation for 1000 cells over top 400 highly-variable genes.
    
    Args: 
        adata: normalized anndata object (annotated data matrix)
    
    Returns: 
        Upper triangle of correlation matrix as array
    '''
    
    sampled = sc.pp.subsample(adata, n_obs = 1000, copy = True)
    sampled_count_matrix = pd.DataFrame.sparse.from_spmatrix(sampled.X)
    highly_variable_index = utils.get_highly_variable_genes_index_number(adata)
    selected_cells = sampled_count_matrix.iloc[:, highly_variable_index]
    cell_corr = selected_cells.transpose().corr()
    
    cell_corr_tri = cell_corr.where(np.triu(np.ones(cell_corr.shape)).astype(bool))
    cell_corr_arr = cell_corr_tri.values.flatten()
    cell_corr_arr = cell_corr_arr[~np.isnan(cell_corr_arr)]
    return cell_corr_arr


def get_ks_cellmetrics(resultpath1, resultpath2):
    '''
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit, which compares the underlying continuous 
    distributions F(x) and G(x) of two independent samples, on all cell-wise metrics. 

    Args: 
        resultpath1: path to result folder of dataset1
        resultpath2: path to result folder of dataset2

    Returns: 
        A Dataframe containing KS test statistic and p-value for each of the following metrics:
            - 'log1p_total_counts_{all/malignant}': log library size of all cells or only of malignant cells respectively
            - 'zeroes_gene_{all/malignant}' : cell detection frequency, i.e. the number of genes expressed per cell
            divided by the number of all genes, of all cells or only malignant cells 
            - 'cell_corr_{hv/all}_{all/malignant}': cell-to-cell correlation over top highly variable genes or over 
            all genes, over all cells or over malignant cells only
    '''
    ks_values = []
    p_values = []

    df1 = pd.read_csv(os.path.join(resultpath1, 'cellmetrics_all.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cellmetrics_all.csv'))

    ks_log1p_total_counts_all = stats.ks_2samp(df1['log1p_total_counts'], df2['log1p_total_counts'])
    ks_values.append(ks_log1p_total_counts_all[0])
    p_values.append(ks_log1p_total_counts_all[1])
    
    ks_zeroes_gene_all = stats.ks_2samp(df1['zeroes_genes'], df2['zeroes_genes'])
    ks_values.append(ks_zeroes_gene_all[0])
    p_values.append(ks_zeroes_gene_all[1])
    
    df1 = pd.read_csv(os.path.join(resultpath1, 'cellmetrics_malignant.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cellmetrics_malignant.csv'))

    ks_log1p_total_counts_malignant = stats.ks_2samp(df1['log1p_total_counts'], df2['log1p_total_counts'])
    ks_values.append(ks_log1p_total_counts_malignant[0])
    p_values.append(ks_log1p_total_counts_malignant[1])
    
    ks_zeroes_gene_malignant = stats.ks_2samp(df1['zeroes_genes'], df2['zeroes_genes'])
    ks_values.append(ks_zeroes_gene_malignant[0])
    p_values.append(ks_zeroes_gene_malignant[1])
   
    df1 = pd.read_csv(os.path.join(resultpath1, 'cell_corr_all.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cell_corr_all.csv'))

    ks_cell_corr_all_all = stats.ks_2samp(df1['all'], df2['all'])
    ks_values.append(ks_cell_corr_all_all[0])
    p_values.append(ks_cell_corr_all_all[1])
    
    ks_cell_corr_hv_all = stats.ks_2samp(df1['hv'], df2['hv'])
    ks_values.append(ks_cell_corr_hv_all[0])
    p_values.append(ks_cell_corr_hv_all[1])

    df1 = pd.read_csv(os.path.join(resultpath1, 'cell_corr_malignant.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cell_corr_malignant.csv'))

    ks_cell_corr_all_malignant = stats.ks_2samp(df1['all'], df2['all'])
    ks_values.append(ks_cell_corr_all_malignant[0])
    p_values.append(ks_cell_corr_all_malignant[1])
    
    ks_cell_corr_hv_malignant = stats.ks_2samp(df1['hv'], df2['hv'])
    ks_values.append(ks_cell_corr_hv_malignant[0])
    p_values.append(ks_cell_corr_hv_malignant[1])
    

    metrics = ['log1p_total_counts_all', 'zeroes_gene_all', 'log1p_total_counts_malignant', 'zeroes_gene_malignant', 
    'cell_corr_all_all', 'cell_corr_hv_all', 'cell_corr_all_malignant', 'cell_corr_hv_malignant']
    
    data = {'metrics' : metrics,
        'KS_test_statistic':ks_values,
        'p_value': p_values}
    df = pd.DataFrame(data)
    return df