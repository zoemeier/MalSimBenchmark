#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
from scipy import stats
import pandas as pd
import numpy as np
import scipy.sparse  # pytype: disable=import-error
import benchmark.base.utils as utils


def set_gene_expression_metrics(adata): 
    '''
    Calculates and writes gene expression metrics in logCPM (log counts per million) to adata.var of anndata object. 
    Calculates mean gene expression level, variance of gene expression, standard deviation of genes and expression variability
    realitve its mean (coefficient of variation of gene expression).
    
    Args: 
        adata: normalized anndata object (annotated data matrix)
    '''
    
    adata.var['mean_logCPM'] = adata.X.mean(axis = 0).A1
    adata.var['var_logCPM'] = utils.vars(adata.X, axis = 0).A1
    adata.var['std_logCPM']= utils.stds(adata.X, axis = 0).A1
    adata.var['cv'] = adata.var['std_logCPM'] / adata.var['mean_logCPM']
    

def set_num_cells_by_counts(adata):
    '''
    Calculates number of cells this expression is measured in and writes result to adata.var in column 'n_cells_by_counts'.
    
    Args: 
        adata: anndata object (annotated data matrix) with raw counts
    '''
    
    arr_start = adata.X.tocsc().indptr[:-1]
    arr_end = adata.X.tocsc().indptr[1:]
    num_cells = arr_end - arr_start 
    adata.var['n_cells_by_counts'] = num_cells

def set_zeroes_cells(adata):
    '''
    Calculates gene detection frequency, i.e. the number of cells this gene is expressed in divided by the number of all cells 
    and writes result to adata.var. (= Inverse of proportion of zeroes per gene) 
    
    Args: 
        adata: anndata object (annotated data matrix) with raw counts
    '''
    
    total_cells = len(adata.obs)
    adata.var['zeroes_cells'] = adata.var['n_cells_by_counts'] / total_cells
    

def get_gene_gene_correlation_he(adata):
    '''
    Calculates gene-to-gene correlation using Pearson correlation for the top 400 most highly expressed genes. 
    
    Args: 
        adata: normalized anndata object (annotated data matrix)
    
    Returns: 
        Upper triangle of correlation matrix as array
    '''
    
    highly_expressed_index = utils.get_highly_expressed_genes_index_number(adata, 400)
    count_matrix = pd.DataFrame.sparse.from_spmatrix(adata.X)
    he_selected_genes = count_matrix.iloc[:,highly_expressed_index]
    he_gene_corr = he_selected_genes.corr()
    
    he_gene_corr_tri = he_gene_corr.where(np.triu(np.ones(he_gene_corr.shape)).astype(bool))
    he_gene_corr_arr = he_gene_corr_tri.values.flatten()
    he_gene_corr_arr = he_gene_corr_arr[~np.isnan(he_gene_corr_arr)]
    return he_gene_corr_arr
    
    
def get_gene_gene_correlation_hv(adata):
    '''
    Calculates gene-to-gene correlation using Pearson correlation for the top 400 most variable genes. 
    
    Args: 
        adata: normalized anndata object (annotated data matrix)
    
    Returns: 
        Upper triangle of correlation matrix as array
    '''
    
    highly_variable_index = utils.get_highly_variable_genes_index_number(adata)
    count_matrix = pd.DataFrame.sparse.from_spmatrix(adata.X)
    hv_selected_genes = count_matrix.iloc[:,highly_variable_index]
    hv_gene_corr = hv_selected_genes.corr()
    
    hv_gene_corr_tri = hv_gene_corr.where(np.triu(np.ones(hv_gene_corr.shape)).astype(bool))
    hv_gene_corr_arr = hv_gene_corr_tri.values.flatten()
    hv_gene_corr_arr = hv_gene_corr_arr[~np.isnan(hv_gene_corr_arr)]
    return hv_gene_corr_arr


def get_ks_genemetrics(resultpath1, resultpath2):
    '''
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit, which compares the underlying continuous distributions 
    F(x) and G(x) of two independent samples, on all gene-wise metrics. 

    Args: 
        resultpath1: path to result folder of dataset1
        resultpath2: path to result folder of dataset2

    Returns: 
        A Dataframe containing KS test statistic and p-values for each of the following metrics:
            - 'mean_logCPM_{all/malignant}': mean of gene expression over all cells or only over malignant cells respectively
            - 'var_logCPM_{all/malignant}' : variance of gene expression over all cells or only over 
            malignant cells respectively
            - 'cv_{all/malignant}' : expression variability relative its mean (coefficient of variation) over all 
            cells or only over malignant cells respectively
            - 'zeroes_cells_{all/malignant}' : gene detection frequency, i.e. the number of cells this gene is expressed in 
            divided by the number of all cells, calculated over all cells or only over malignant cells respectively
            - 'gene_corr_{he/hv}_{all/malignant}': gene-to-gene correlation over top highly expressed or highly variable genes, 
            over all cells or over malignant cells only

    '''
    ks_values = []
    p_values = []
    
    df1 = pd.read_csv(os.path.join(resultpath1, 'genemetrics_all.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'genemetrics_all.csv'))

    ks_mean_logCPM_all = stats.ks_2samp(df1['mean_logCPM'], df2['mean_logCPM'])
    ks_values.append(ks_mean_logCPM_all[0])
    p_values.append(ks_mean_logCPM_all[1])
    
    ks_var_logCPM_all = stats.ks_2samp(df1['var_logCPM'], df2['var_logCPM'])
    ks_values.append(ks_var_logCPM_all[0])
    p_values.append(ks_var_logCPM_all[1])
    
    ks_cv_all = stats.ks_2samp(df1['cv'], df2['cv'])
    ks_values.append(ks_cv_all[0])
    p_values.append(ks_cv_all[1])
    
    ks_zeroes_cells_all = stats.ks_2samp(df1['zeroes_cells'], df2['zeroes_cells'])
    ks_values.append(ks_zeroes_cells_all[0])
    p_values.append(ks_zeroes_cells_all[1])

    df1 = pd.read_csv(os.path.join(resultpath1, 'genemetrics_malignant.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'genemetrics_malignant.csv'))

    ks_mean_logCPM_malignant = stats.ks_2samp(df1['mean_logCPM'], df2['mean_logCPM'])
    ks_values.append(ks_mean_logCPM_malignant[0])
    p_values.append(ks_mean_logCPM_malignant[1])
    
    ks_var_logCPM_malignant = stats.ks_2samp(df1['var_logCPM'], df2['var_logCPM'])
    ks_values.append(ks_var_logCPM_malignant[0])
    p_values.append(ks_var_logCPM_malignant[1])
    
    ks_cv_malignant = stats.ks_2samp(df1['cv'], df2['cv'])
    ks_values.append(ks_cv_malignant[0])
    p_values.append(ks_cv_malignant[1])
    
    ks_zeroes_cells_malignant = stats.ks_2samp(df1['zeroes_cells'], df2['zeroes_cells'])
    ks_values.append(ks_zeroes_cells_malignant[0])
    p_values.append(ks_zeroes_cells_malignant[1])
    
    df1 = pd.read_csv(os.path.join(resultpath1, 'gene_corr_all.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'gene_corr_all.csv'))

    ks_gene_corr_he_all = stats.ks_2samp(df1['he'], df2['he'])
    ks_values.append(ks_gene_corr_he_all[0])
    p_values.append(ks_gene_corr_he_all[1])
    
    ks_gene_corr_hv_all = stats.ks_2samp(df1['hv'], df2['hv'])
    ks_values.append(ks_gene_corr_hv_all[0])
    p_values.append(ks_gene_corr_hv_all[1])

    df1 = pd.read_csv(os.path.join(resultpath1, 'gene_corr_malignant.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'gene_corr_malignant.csv'))

    ks_gene_corr_he_malignant = stats.ks_2samp(df1['he'], df2['he'])
    ks_values.append(ks_gene_corr_he_malignant[0])
    p_values.append(ks_gene_corr_he_malignant[1])
    
    ks_gene_corr_hv_malignant = stats.ks_2samp(df1['hv'], df2['hv'])
    ks_values.append(ks_gene_corr_hv_malignant[0])
    p_values.append(ks_gene_corr_hv_malignant[1])
    
    metrics = ['mean_logCPM_all', 'var_logCPM_all', 'cv_all', 'zeroes_cells_all', 
    'mean_logCPM_malignant', 'var_logCPM_malignant', 'cv_malignant', 'zeroes_cells_malignant',
    'gene_corr_he_all', 'gene_corr_hv_all', 'gene_corr_he_malignant', 'gene_corr_hv_malignant']
    
    data = {'metrics' : metrics,
        'KS_test_statistic':ks_values,
        'p_value': p_values}
    df = pd.DataFrame(data)
    
    return df
    