#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import itertools
import re
import cansig 
import pandas as pd
from scipy import stats
from typing import Sequence, Tuple, Union

import numpy as np
import scipy.ndimage  # pytype: disable=import-error
import scipy.sparse  # pytype: disable=import-error
from anndata import AnnData  # pytype: disable=import-error
from infercnvpy._util import _ensure_array  # pytype: disable=import-error
from scanpy import logging  # pytype: disable=import-error
from tqdm.auto import tqdm  # pytype: disable=import-error
from tqdm.contrib.concurrent import process_map  # pytype: disable=import-error



def get_boundaries(cnv_mapping, chr_pos, threshold):

    """
    Finds boundaries for possible connected CNV regions 
    (i.e. within one chromosome, not further from each other than set threshold)
   
       Args:
            cnv_mapping: Dataframe describing the mapping of CNV windows of the CNV matrix to the genes used for the
            computation of the CNV call
            
            threshold: max distance between two CNV windows allowed within a consecutive CNV region
            
        Returns: A List of indexes for the boundaries of possible connected CNV regions

    """

    num_col = len(cnv_mapping.loc['chromosome'])
    split = []

    for i in range(num_col):
        if i > 0: 
            #if distance between end of window and beginning of next window is too big, we set boundary for possible region
            if cnv_mapping[i]['begin'] - cnv_mapping[i-1]['end'] > threshold: 
                split.append(i)
            
    split = np.asarray(split)

    #separate also by chromosomes
    splits = np.append(split, np.append(chr_pos, [num_col]))
    splits.sort()
    return splits

def get_region_index(segment, cnv_mapping, distance_threshold):
    
    """
    Finds consecutive regions of CNV within a segment
        Args:
            segment: subsection of consecutive columns of CNV gain/loss matrix, given by the boundaries
            of possible connected CNV regions
            cnv_mapping: .........
            
        Returns: A List of Tuples with start and end position of all the regions in this chromosome segment.
    """
    
    start_ends = []
    start = -1
    #the first element is special
    cols = segment.index
    begin = cols[0]
    end = cols[-1]
  
    last1_end_loc = 1000000000
    last1_index = -1
  
  
    if segment[begin] == 1:
        start = begin
        last1_end_loc = cnv_mapping[begin]['end']
        last1_index = begin
        
    index = begin + 1
    

    while index <= end:
        if segment[index] == 1:
            if (cnv_mapping[index]['begin'] - last1_end_loc <= distance_threshold):
                last1_end_loc = cnv_mapping[index]['end'] 
                if start == -1:
                    start = index
            else:
                start_ends.append((start, last1_index))
                start = index
                last1_end_loc = cnv_mapping[index]['end']          
            last1_index = index     
        index += 1

    #take care of the last one
    if start > -1:
        start_ends.append((start, last1_index))
        
    return start_ends



def get_start_chromosome(cnv_mapping, x):
    return cnv_mapping[x['start_index']]['begin']

def get_end_chromosome(cnv_mapping, x):
    return cnv_mapping[x['end_index']]['end']

def get_chromosome_number(cnv_mapping, x):
    return cnv_mapping[x['end_index']]['chromosome']

def get_num_genes_total(adata, x):
    chr_genes = adata.var[adata.var['chromosome'] == x['chromosome']]
    genes = chr_genes[chr_genes['start'] >= x['start_location']]
    genes = genes[genes['end'] <= x['end_location']]
    return len(genes)

def get_chr_length(chr_sizes, x):
    return chr_sizes.loc[x['chromosome']]['size']


def get_region_metrics(adata, cnv_mapping, chr_sizes, region):
     
    region['chromosome'] = region.apply(lambda x: get_chromosome_number(cnv_mapping,x), axis = 1)
    region['chromosome_length'] = region.apply(lambda x: get_chr_length(chr_sizes,x), axis = 1)
    region['start_location'] = region.apply(lambda x: get_start_chromosome(cnv_mapping,x), axis = 1)
    region['end_location'] = region.apply(lambda x: get_end_chromosome(cnv_mapping,x) , axis = 1)
    region['size_bp_total'] = region['end_location'] - region['start_location']+1
    region['middle_location'] = np.floor(region['start_location'] + (region['size_bp_total']/2))
    region['relative_location'] = region['middle_location'] / region['chromosome_length'] * 100    
    region['size_genes_total'] = region.apply(lambda x: get_num_genes_total(adata,x), axis = 1)
    
    return region 



def get_regions(subclonal_CNV, splits, cnv_mapping, distance_threshold):
    # Split CNV matrix into gains and loss matrix 
    cnv_gains = subclonal_CNV.replace([-1], 0)
    cnv_losses = abs(subclonal_CNV.replace([1], 0))
    
    # Split columns into possibly connected regions of CNV (i.e. split dataframe columns into by boundaries calculated prev. )
    list_gains = [cnv_gains.iloc[:,splits[i]:splits[i+1]] for i in range(len(splits)-1)]
    list_losses = [cnv_losses.iloc[:,splits[i]:splits[i+1]] for i in range(len(splits)-1)]
    
    # For each possible connected region: find consecutive gains or losses within threshold distance
    gain_regions_unique = pd.concat([l.apply(lambda x: get_region_index(x, cnv_mapping, distance_threshold), 
                                             axis = 1).to_frame('index').explode('index').dropna() for l in list_gains])
    loss_regions_unique = pd.concat([l.apply(lambda x: get_region_index(x, cnv_mapping, distance_threshold), 
                                             axis =1).to_frame('index').explode('index').dropna() for l in list_losses])
    loss_regions_unique[['start_index', 'end_index']] = pd.DataFrame(loss_regions_unique['index'].tolist(), 
                                                                      index=loss_regions_unique.index)
        
    gain_regions_unique[['start_index', 'end_index']] = pd.DataFrame(gain_regions_unique['index'].tolist(),
                                                                          index=gain_regions_unique.index)
    
    return gain_regions_unique, loss_regions_unique



def get_cnvmetrics_per_cell(adata, loss_regions, gain_regions):
    
    df = adata.obs[['Patient', 'subclonal','malignant_key']].copy()

    df.loc[:,'total_gains_genes'] = gain_regions[['cell_id','size_genes_total']].groupby('cell_id').sum()
    df.loc[:,'total_losses_genes'] = loss_regions[['cell_id','size_genes_total']].groupby('cell_id').sum()

    df.loc[:,'total_gains_bp'] = gain_regions[['cell_id','size_bp_total']].groupby('cell_id').sum()
    df.loc[:,'total_losses_bp'] = loss_regions[['cell_id','size_bp_total']].groupby('cell_id').sum() 

    df.loc[:,'num_gain_region'] = gain_regions.groupby('cell_id').size()
    df.loc[:,'min_gain_region'] = gain_regions[['cell_id','size_bp_total']].groupby('cell_id').min()
    df.loc[:,'mean_gain_region'] = gain_regions[['cell_id','size_bp_total']].groupby('cell_id').mean()
    df.loc[:,'max_gain_region'] = gain_regions[['cell_id','size_bp_total']].groupby('cell_id').max()
    df.loc[:,'median_gain_region'] = gain_regions[['cell_id','size_bp_total']].groupby('cell_id').median()

    df.loc[:,'num_loss_region'] = loss_regions.groupby('cell_id').size()
    df.loc[:,'min_loss_region'] = loss_regions[['cell_id','size_bp_total']].groupby('cell_id').min()
    df.loc[:,'mean_loss_region'] = loss_regions[['cell_id','size_bp_total']].groupby('cell_id').mean()
    df.loc[:,'max_loss_region'] = loss_regions[['cell_id','size_bp_total']].groupby('cell_id').max()
    df.loc[:,'median_loss_region'] = loss_regions[['cell_id','size_bp_total']].groupby('cell_id').median()
    
    metrics= ['total_gains_genes', 'total_losses_genes', 'total_gains_bp', 'total_losses_bp', 'num_gain_region',
              'min_gain_region', 'mean_gain_region', 'max_gain_region', 'median_gain_region', 'num_loss_region',
              'min_loss_region', 'mean_loss_region', 'max_loss_region', 'median_loss_region']
    
    df.loc[:, metrics] = df.loc[:, metrics].fillna(0)         


    return df

 
def get_unique_metrics(cnv_metrics_per_cell):
    
    unique_index = ['subclonal', 'total_gains_genes', 'total_losses_genes', 'total_gains_bp', 'total_losses_bp','num_gain_region',
                    'min_gain_region', 'mean_gain_region','max_gain_region', 'median_gain_region', 'num_loss_region',
                    'min_loss_region', 'mean_loss_region', 'max_loss_region', 'median_loss_region']
    new_index = ['subclonal', 'malignant_key', 'total_gains_genes', 'total_losses_genes', 'total_gains_bp', 'total_losses_bp',
                    'num_gain_region', 'min_gain_region', 'mean_gain_region', 'max_gain_region', 'median_gain_region',
                    'num_loss_region', 'min_loss_region', 'mean_loss_region', 'max_loss_region', 'median_loss_region']
    
    df_unique = cnv_metrics_per_cell.drop_duplicates(unique_index)
    df_unique = df_unique[new_index].set_index('subclonal')
    return df_unique
   
   
def get_ks_cnvmetrics(resultpath1, resultpath2, step, thresh):
    '''
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit, which compares the underlying 
    continuous distributions F(x) and G(x) of two independent samples, on all cnv metrics. 

    Args: 
        resultpath1: path to result folder of dataset1
        resultpath2: path to result folder of dataset2
        step: step size used in inferCNV, to select correct result datasets, if results for multiple step arguments available
        thresh: threshold used to group CNV regions, to select correct result datasets, 
        if results for multiple step arguments available

    Returns: 
        A Dataframe containing KS test statistic and p-value for each of the following metrics:
            - 'cnvregion_{gain/loss}_size_{bp/gene}': distribution of the size of consecutive regions over all cells
            of gain or losses respectively, meassured in bp or genes
            - 'cnvregion_size_{bp/gene}': distribution of the size of all consecutive regions over all cells, 
            measured in bp or genes.
            - 'total_{gains/losses}_per_{subclone/cell}_{bp/gene}': total size of sum of all CNV regions of type gain or loss, 
            per cell or subclone respectively, measured in bp or genes
            - 'num_{gain/loss}_region_per_{subclone/cell}': number of gain or loss cnv regions per subclone or cell. 
            - '{min/mean/median/max}_{gain/loss}_region_per_{subclone/cell}': Min (or mean, median, max) size of 
            consecutive CNV regions of type gain or loss per cell (or subclone), measured in bp

    '''
    paramstring = '_step' + str(step) + '_thresh' + str(thresh)
    ks_values = []
    p_values = []
    
    #overall
    df1 = pd.read_csv(os.path.join(resultpath1, 'cnv_regions' + paramstring + '.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cnv_regions' + paramstring + '.csv'))

    ks_size_bp_gain = stats.ks_2samp(df1[df1['type']=='gain']['size_bp_total'], df2[df2['type']=='gain']['size_bp_total'])
    ks_values.append(ks_size_bp_gain[0])
    p_values.append(ks_size_bp_gain[1])
    
    ks_size_gene_gain = stats.ks_2samp(df1[df1['type']=='gain']['size_genes_total'],
                                       df2[df2['type']=='gain']['size_genes_total'])
    ks_values.append(ks_size_gene_gain[0])
    p_values.append(ks_size_gene_gain[1])
    
    ks_size_bp_loss = stats.ks_2samp(df1[df1['type']=='loss']['size_bp_total'], df2[df2['type']=='loss']['size_bp_total'])
    ks_values.append(ks_size_bp_loss[0])
    p_values.append(ks_size_bp_loss[1])
    
    ks_size_gene_loss = stats.ks_2samp(df1[df1['type']=='loss']['size_genes_total'], 
                                       df2[df2['type']=='loss']['size_genes_total'])
    ks_values.append(ks_size_gene_loss[0])
    p_values.append(ks_size_gene_loss[1])
    
    ks_size_bp = stats.ks_2samp(df1['size_bp_total'], df2['size_bp_total'])
    ks_values.append(ks_size_bp[0])
    p_values.append(ks_size_bp[1])
    
    ks_size_gene = stats.ks_2samp(df1['size_genes_total'], df2['size_genes_total'])
    ks_values.append(ks_size_gene[0])
    p_values.append(ks_size_gene[1])

    # per subclonal
    df1 = pd.read_csv(os.path.join(resultpath1, 'cnvmetrics_per_subclonal' + paramstring + '.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cnvmetrics_per_subclonal' + paramstring + '.csv'))

    ks_total_gains_per_subclone_bp = stats.ks_2samp(df1['total_gains_bp'], df2['total_gains_bp'])
    ks_values.append(ks_total_gains_per_subclone_bp[0])
    p_values.append(ks_total_gains_per_subclone_bp[1])
    
    ks_total_losses_per_subclone_bp = stats.ks_2samp(df1['total_losses_bp'], df2['total_losses_bp'])
    ks_values.append(ks_total_losses_per_subclone_bp[0])
    p_values.append(ks_total_losses_per_subclone_bp[1])

    ks_total_gains_per_subclone_genes = stats.ks_2samp(df1['total_gains_genes'], df2['total_gains_genes'])
    ks_values.append(ks_total_gains_per_subclone_genes[0])
    p_values.append(ks_total_gains_per_subclone_genes[1])
    
    ks_total_losses_per_subclone_genes = stats.ks_2samp(df1['total_losses_genes'], df2['total_losses_genes'])
    ks_values.append(ks_total_losses_per_subclone_genes[0])
    p_values.append(ks_total_losses_per_subclone_genes[1])

    ks_num_gain_region_per_subclone = stats.ks_2samp(df1['num_gain_region'], df2['num_gain_region'])
    ks_values.append(ks_num_gain_region_per_subclone[0])
    p_values.append(ks_num_gain_region_per_subclone[1])
    
    ks_num_loss_region_per_subclone = stats.ks_2samp(df1['num_loss_region'], df2['num_loss_region'])
    ks_values.append(ks_num_loss_region_per_subclone[0])
    p_values.append(ks_num_loss_region_per_subclone[1])
    

    ks_min_gain_region_per_subclone = stats.ks_2samp(df1['min_gain_region'], df2['min_gain_region'])
    ks_values.append(ks_min_gain_region_per_subclone[0])
    p_values.append(ks_min_gain_region_per_subclone[1])
    
    ks_mean_gain_region_per_subclone = stats.ks_2samp(df1['mean_gain_region'], df2['mean_gain_region'])
    ks_values.append(ks_mean_gain_region_per_subclone[0])
    p_values.append(ks_mean_gain_region_per_subclone[1])
    
    ks_median_gain_region_per_subclone = stats.ks_2samp(df1['median_gain_region'], df2['median_gain_region'])
    ks_values.append(ks_median_gain_region_per_subclone[0])
    p_values.append(ks_median_gain_region_per_subclone[1])

    ks_max_gain_region_per_subclone = stats.ks_2samp(df1['max_gain_region'], df2['max_gain_region'])
    ks_values.append(ks_max_gain_region_per_subclone[0])
    p_values.append(ks_max_gain_region_per_subclone[1])

    ks_min_loss_region_per_subclone = stats.ks_2samp(df1['min_loss_region'], df2['min_loss_region'])
    ks_values.append(ks_min_loss_region_per_subclone[0])
    p_values.append(ks_min_loss_region_per_subclone[1])
    
    ks_mean_loss_region_per_subclone = stats.ks_2samp(df1['mean_loss_region'], df2['mean_loss_region'])
    ks_values.append(ks_mean_loss_region_per_subclone[0])
    p_values.append(ks_mean_loss_region_per_subclone[1])
    
    ks_median_loss_region_per_subclone = stats.ks_2samp(df1['median_loss_region'], df2['median_loss_region'])
    ks_values.append(ks_median_loss_region_per_subclone[0])
    p_values.append(ks_median_loss_region_per_subclone[1])
    
    ks_max_loss_region_per_subclone = stats.ks_2samp(df1['max_loss_region'], df2['max_loss_region'])
    ks_values.append(ks_max_loss_region_per_subclone[0])
    p_values.append(ks_max_loss_region_per_subclone[1])
    

    #per cell
    df1 = pd.read_csv(os.path.join(resultpath1, 'cnvmetrics_per_cell' + paramstring + '.csv'))
    df2 = pd.read_csv(os.path.join(resultpath2, 'cnvmetrics_per_cell' + paramstring + '.csv'))

    ks_total_gains_per_cell_bp = stats.ks_2samp(df1['total_gains_bp'], df2['total_gains_bp'])
    ks_values.append(ks_total_gains_per_cell_bp[0])
    p_values.append(ks_total_gains_per_cell_bp[1])
    
    ks_total_losses_per_cell_bp = stats.ks_2samp(df1['total_losses_bp'], df2['total_losses_bp'])
    ks_values.append(ks_total_losses_per_cell_bp[0])
    p_values.append(ks_total_losses_per_cell_bp[1])

    ks_total_gains_per_cell_genes = stats.ks_2samp(df1['total_gains_genes'], df2['total_gains_genes'])
    ks_values.append(ks_total_gains_per_cell_genes[0])
    p_values.append(ks_total_gains_per_cell_genes[1])

    ks_total_losses_per_cell_genes = stats.ks_2samp(df1['total_losses_genes'], df2['total_losses_genes'])
    ks_values.append(ks_total_losses_per_cell_genes[0])
    p_values.append(ks_total_losses_per_cell_genes[1])

    ks_num_gain_region_per_cell = stats.ks_2samp(df1['num_gain_region'], df2['num_gain_region'])
    ks_values.append(ks_num_gain_region_per_cell[0])
    p_values.append(ks_num_gain_region_per_cell[1])
    
    ks_num_loss_region_per_cell = stats.ks_2samp(df1['num_loss_region'], df2['num_loss_region'])
    ks_values.append(ks_num_loss_region_per_cell[0])
    p_values.append(ks_num_loss_region_per_cell[1])

    ks_min_gain_region_per_cell = stats.ks_2samp(df1['min_gain_region'], df2['min_gain_region'])
    ks_values.append(ks_min_gain_region_per_cell[0])
    p_values.append(ks_min_gain_region_per_cell[1])
    
    ks_mean_gain_region_per_cell = stats.ks_2samp(df1['mean_gain_region'], df2['mean_gain_region'])
    ks_values.append(ks_mean_gain_region_per_cell[0])
    p_values.append(ks_mean_gain_region_per_cell[1])
    
    ks_median_gain_region_per_cell = stats.ks_2samp(df1['median_gain_region'], df2['median_gain_region'])
    ks_values.append(ks_median_gain_region_per_cell[0])
    p_values.append(ks_median_gain_region_per_cell[1])
    
    ks_max_gain_region_per_cell = stats.ks_2samp(df1['max_gain_region'], df2['max_gain_region'])
    ks_values.append(ks_max_gain_region_per_cell[0])
    p_values.append(ks_max_gain_region_per_cell[1])

    ks_min_loss_region_per_cell = stats.ks_2samp(df1['min_loss_region'], df2['min_loss_region'])
    ks_values.append(ks_min_loss_region_per_cell[0])
    p_values.append(ks_min_loss_region_per_cell[1])
    
    ks_mean_loss_region_per_cell = stats.ks_2samp(df1['mean_loss_region'], df2['mean_loss_region'])
    ks_values.append(ks_mean_loss_region_per_cell[0])
    p_values.append(ks_mean_loss_region_per_cell[1])
    
    ks_median_loss_region_per_cell = stats.ks_2samp(df1['median_loss_region'], df2['median_loss_region'])
    ks_values.append(ks_median_loss_region_per_cell[0])
    p_values.append(ks_median_loss_region_per_cell[1])
    
    ks_max_loss_region_per_cell = stats.ks_2samp(df1['max_loss_region'], df2['max_loss_region'])
    ks_values.append(ks_max_loss_region_per_cell[0])
    p_values.append(ks_max_loss_region_per_cell[1])

    metrics = ['cnvregion_gain_size_bp','cnvregion_gain_size_gene','cnvregion_loss_size_bp','cnvregion_loss_size_gene',
              'cnvregion_size_bp','cnvregion_size_gene',
              'total_gains_per_subclone_bp', 'total_losses_per_subclone_bp','total_gains_per_subclone_genes',
              'total_losses_per_subclone_genes', 
              'num_gain_region_per_subclone','num_loss_region_per_subclone',
              'min_gain_region_per_subclone', 'mean_gain_region_per_subclone', 
              'median_gain_region_per_subclone', 'max_gain_region_per_subclone', 
              'min_loss_region_per_subclone', 'mean_loss_region_per_subclone',
              'median_loss_region_per_subclone', 'max_loss_region_per_subclone',
              'total_gains_per_cell_bp', 'total_losses_per_cell_bp','total_gains_per_cell_genes', 
              'total_losses_per_cell_genes', 
              'num_gain_region_per_cell','num_loss_region_per_cell',
              'min_gain_region_per_cell', 'mean_gain_region_per_cell', 
              'median_gain_region_per_cell', 'max_gain_region_per_cell', 
              'min_loss_region_per_cell', 'mean_loss_region_per_cell','median_loss_region_per_cell', 
              'max_loss_region_per_cell']

    data = {'metrics' : metrics,
        'KS_test_statistic':ks_values,
        'p_value': p_values}
    df = pd.DataFrame(data)
    return df
