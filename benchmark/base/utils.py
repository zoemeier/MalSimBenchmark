#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import scanpy as sc
import numpy as np
import scipy.sparse  # pytype: disable=import-error

def vars(a, axis=None):
    '''
    Calculates variance of Scipy sparse matrix a (var = mean(a**2) - mean(a)**2)
    
    Args: 
        a: Scipy sparse matrix 
        axis: {0,1,None} specifies axis along which the variance is calculated
    '''
    
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def stds(a, axis=None):
    '''
    Calculates standard deviation of Scipy sparse matrix a (std = sqrt(var(a)))
    
    Args: 
        a: Scipy sparse matrix 
        axis: {0,1,None} specifies axis along which the standard deviation is calculated
    '''
    
    return np.sqrt(vars(a, axis))


def normalize_adata(adata, t_sum:None):
    '''
    Log1p normalization of data matrix. Normalize each cell by total counts over all gene and then logarithmize the datamatrix.
    If choosing t_sum=1e6, this is CPM normalization. If None, after normalization, each observation (cell) has a total count 
    equal to the median of total counts for observations (cells) before normalization.
    
    Args: 
        adata: anndata object (annotated data matrix)
        t_sum: target sum (default = None)
    '''
    
    sc.pp.normalize_total(adata, target_sum=t_sum)
    sc.pp.log1p(adata)

    
def get_malignant_index(adata): 
    '''
    Calculates integer index of malignant cells in anndata object
    
    Args: 
        adata: anndata object (annotated data matrix with malignancy annotation in 'malignant_key' column)
    
    Returns: 
        Array of integer indices of malignant cells
    '''
    
    copy = adata.obs.copy().reset_index()
    malignant_index = copy[copy['malignant_key'] == 'malignant']
    malignant_index = malignant_index.index.sort_values()
    return malignant_index


def get_highly_expressed_genes_index_number(adata, num): 
    '''
    Calculates integer index of highly expressed genes in anndata object
    
    Args: 
        adata: anndata object (annotated data matrix)
        num: number of highly-expressed genes to keep 
    
    Returns: 
        Array of integer indices of highly expressed genes
    '''
    
    bdata = adata.var.copy()
    bdata.reset_index(inplace = True)
    highly_expressed_index = bdata.sort_values(by='means', ascending = False).head(num).index
    highly_expressed_index = highly_expressed_index.sort_values()
    return highly_expressed_index
    
def get_highly_variable_genes_index_number(adata):
    '''
    Calculates integer index of highly variable genes in anndata object
    
    Args: 
        adata: anndata object (annotated data matrix with 'highly_variable' column (e.g. previously computed using scanpy))
      
    Returns: 
        Array of integer indices of highly variable genes
    '''
    
    bdata = adata.var.copy()
    bdata.reset_index(inplace = True)
    highly_variable_index = bdata[bdata['highly_variable']].index
    highly_variable_index = highly_variable_index.sort_values()
    return highly_variable_index


def get_highly_expressed_genes_index(adata, num): 
    '''
    Calculates gene name index of highly expressed genes in anndata object
    
    Args: 
        adata: anndata object (annotated data matrix)
        num: number of highly-expressed genes to keep 
    
    Returns: 
        Array of gene name indices of highly expressed genes
    '''
    
    highly_expressed_index = adata.var.sort_values(by='means', ascending = False).head(num).index
    highly_expressed_index = highly_expressed_index.sort_values()
    return highly_expressed_index
    
    
def get_highly_variable_genes_index(adata):
    '''
    Calculates gene name index of highly variable genes in anndata object
    
    Args: 
        adata: anndata object (annotated data matrix with 'highly_variable' column (e.g. previously computed using scanpy))
      
    Returns: 
        Array of gene name indices of highly variable genes
    '''
    
    highly_variable_index = adata.var[adata.var['highly_variable']].index
    highly_variable_index = highly_variable_index.sort_values()
    return highly_variable_index




def rename_subclonal(adata, subclonal_index, patient_index): 
    '''
    Renames subclonal column to combination of subclonal and patient, so patients can be merged into one anndata without
    being merged into one big subclonal group. Additionally renaming NA to non-malignant.
    
    Args: 
        adata: merged anndata object of several patients with overlapping names for subclonal groups
        subclonal_index: name of subclonal column  
        patient_index: name of patient column 
    '''
    def get_new_name(x):
        if x[subclonal_index] == 'NA':
            return 'non-malignant'
        return str(x[subclonal_index]) + '_' + str(x[patient_index])
    
    new_subclonal = adata.obs.apply(get_new_name, axis = 1)    
    adata.obs[subclonal_index] = new_subclonal
    
def sort_by_chromosome(df, chromosome_index):
    '''
    Returns dataframe sorted by chromosome and changes chromosome name from string to integer, so that the
    dataframe can be sorted (e.g. from 'chr22' to 22)
    
    Args: 
        df: dataframe, with chromosome column
        chromosome_index: name of chromosome column
    '''
    def replace_chr(x):
        return int(x[chromosome_index].replace('chr',''))
     
    df[chromosome_index] = df.apply(replace_chr,axis = 1)
    df.sort_values(by=[chromosome_index], inplace = True)
    

