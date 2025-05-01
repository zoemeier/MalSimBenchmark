import chardet
import datetime as dt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import benchmark.base.utils as utils


from bisect import bisect_left
from matplotlib import cm
from matplotlib import colors
from matplotlib.lines import Line2D
from benchmark.plot.plot_utils import *
from scipy import stats


def plot_cnv_bp(df1, df2, percentile, output_dir, output_name, 
                df1_name, df2_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots a, b, c are the histograms of length of
        CNVs for two different datasets, measured in base 
        pairs (bp): a) gain CNVs, b) loss CNVs, and c) all 
        CNVs (total = gain and loss). Sub-plot d shows the 
        comparison between cumulative distributions of total 
        between for the two datasets.         
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['type', 'size_bp_total']
    if not is_dataframe_valid(df1, col_names):
        return 
    if not is_dataframe_valid(df2, col_names):
        return 
        
    gain1 = df1[df1['type']=='gain']['size_bp_total']
    loss1 = df1[df1['type']=='loss']['size_bp_total']
    total1 = df1['size_bp_total']

    gain2 = df2[df2['type']=='gain']['size_bp_total']
    loss2 = df2[df2['type']=='loss']['size_bp_total']
    total2 = df2['size_bp_total']
        
    fig, axs = plt.subplots(2, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.07), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=None, wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim1 = get_hist_xlim(df1['size_bp_total'], percentile)
    xlim2 = get_hist_xlim(df2['size_bp_total'], percentile)

    xlim = max(xlim1, xlim2)
    
    props_list = [get_hist_props(total1, xlim, percentile, normalized=True),
                  get_hist_props(total2, xlim, percentile, normalized=True),
                  get_hist_props(gain1, xlim, percentile, normalized=True),
                  get_hist_props(gain2, xlim, percentile, normalized=True),
                  get_hist_props(loss1, xlim, percentile, normalized=True),
                  get_hist_props(loss2, xlim, percentile, normalized=True)]

    props = merge_props(props_list)

    plot_hist(axs[0,0], gain1, 'Copy Number Variation (Gain)', 
              'a', 'Length of CNV (bp)', props, color='#0000bb', alpha=0.3, density=True)
    plot_hist(axs[0,0], gain2, 'Copy Number Variation (Gain)', 
              'a', 'Length of CNV (bp)', props, color='#aa0000', alpha=0.3, density=True)
    ks = round(stats.ks_2samp(gain1, gain2)[0],4)
    axs[0,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,0].transAxes)
    
    plot_hist(axs[0,1], loss1, 'Copy Number Variation (Loss)', 
              'b', 'Length of CNV (bp)', props, color='#0000bb', alpha=0.3, density=True)
    plot_hist(axs[0,1], loss2, 'Copy Number Variation (Loss)', 
              'b', 'Length of CNV (bp)', props, color='#aa0000', alpha=0.3, density=True)
    
    ks = round(stats.ks_2samp(loss1, loss2)[0],4)
    axs[0,1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    plot_hist(axs[1,0], total1, 'Copy Number Variation', 
              'c', 'Length of CNV (bp)', props, color='#0000bb', alpha=0.3, density=True)
    plot_hist(axs[1,0], total2, 'Copy Number Variation', 
              'c', 'Length of CNV (bp)', props, color='#aa0000', alpha=0.3, density=True)
    ks = round(stats.ks_2samp(total1, total2)[0],4)
    axs[1,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    plot_cum_comp(axs[1,1], total1, total2, props, 'Length of CNV (bp)')

    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
        
        
    
def plot_cnv_gene(df1, df2, percentile, output_dir, output_name,
                  df1_name, df2_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots a, b, c are the histograms of length of
        CNVs for two different datasets, measured in genes
        a) gain CNVs, b) loss CNVs, and c) all CNVs 
        (total = gain and loss). Sub-plot d shows the 
        comparison between cumulative distributions of total 
        between for the two datasets.         
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """    
    col_names = ['type', 'size_genes_total']
    if not is_dataframe_valid(df1, col_names):
        return 
    if not is_dataframe_valid(df2, col_names):
        return 
    gain1 = df1[df1['type']=='gain']['size_genes_total']
    loss1 = df1[df1['type']=='loss']['size_genes_total']
    total1 = df1['size_genes_total']
    gain2 = df2[df2['type']=='gain']['size_genes_total']
    loss2 = df2[df2['type']=='loss']['size_genes_total']
    total2 = df2['size_genes_total']
    
    fig, axs = plt.subplots(2, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.07), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, 
                        right=None, top=None, wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim1 = get_hist_xlim(df1['size_genes_total'], percentile)
    xlim2 = get_hist_xlim(df2['size_genes_total'], percentile)
    xlim = max(xlim1, xlim2)
    
    props_list = [get_hist_props(total1, xlim, percentile, normalized=True),
                  get_hist_props(total2, xlim, percentile, normalized=True),
                  get_hist_props(gain1, xlim, percentile, normalized=True),
                  get_hist_props(gain2, xlim, percentile, normalized=True),
                  get_hist_props(loss1, xlim, percentile, normalized=True),
                  get_hist_props(loss2, xlim, percentile, normalized=True)]    
    
    props = merge_props(props_list)

    plot_hist(axs[0,0], gain1, 'Copy Number Variation (Gain)', 
              'a', 'Length of CNV (gene)', props, color='#0000bb', alpha=0.3, density=True)
    plot_hist(axs[0,0], gain2, 'Copy Number Variation (Gain)', 
              'a', 'Length of CNV (gene)', props, color='#aa0000', alpha=0.3, density=True)
    ks = round(stats.ks_2samp(gain1, gain2)[0],4)
    axs[0,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,0].transAxes)
    
    plot_hist(axs[0,1], loss1, 'Copy Number Variation (Loss)', 
              'b', 'Length of CNV (gene)', props, color='#0000bb', alpha=0.3, density=True)
    plot_hist(axs[0,1], loss2, 'Copy Number Variation (Loss)', 
              'b', 'Length of CNV (gene)', props, color='#aa0000', alpha=0.3, density=True)
    ks = round(stats.ks_2samp(loss1, loss2)[0],4)
    axs[0,1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    plot_hist(axs[1,0], total1, 'Copy Number Variation', 
              'c', 'Length of CNV (gene)', props, color='#0000bb', alpha=0.3, density=True)
    plot_hist(axs[1,0], total2, 'Copy Number Variation', 
              'c', 'Length of CNV (gene)', props, color='#aa0000', alpha=0.3, density=True)
    ks = round(stats.ks_2samp(total1, total2)[0],4)
    axs[1,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    plot_cum_comp(axs[1,1], total1, total2, props, 'Length of CNV (gene)')
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")    
    
    
 

def plot_stat_bp(df1, df2,  percentile, output_dir, output_name,
                 df1_name, df2_name):
    """
        Plots a figure with 8 sub-plots (2 rows, 4 columns) for two
        datasets. The sub-plots show histograms for the statistics 
        (max, mean, median, min)of the length of gain and 
        loss CNV regions per cell for the two datasets, 
        measured in base-pairs.
        
            Args:
                df1: dataframe that includes plot information for the
                            first dataset
                df2: dataframe that includes plot information for the
                            second dataset
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['malignant_key', 'max_gain_region', 'mean_gain_region',
                'median_gain_region', 'min_gain_region', 'max_loss_region',
                'mean_loss_region', 'median_loss_region', 'min_loss_region']
    if not is_dataframe_valid(df1, col_names):
        return 
    if not is_dataframe_valid(df2, col_names):
        return 
    
    max_gain1 = df1[df1['malignant_key']=='malignant']['max_gain_region']
    mean_gain1 = df1[df1['malignant_key']=='malignant']['mean_gain_region']
    median_gain1 = df1[df1['malignant_key']=='malignant']['median_gain_region']
    min_gain1 = df1[df1['malignant_key']=='malignant']['min_gain_region']
    max_loss1 = df1[df1['malignant_key']=='malignant']['max_loss_region']
    mean_loss1 =  df1[df1['malignant_key']=='malignant']['mean_loss_region']
    median_loss1 = df1[df1['malignant_key']=='malignant']['median_loss_region']
    min_loss1 = df1[df1['malignant_key']=='malignant']['min_loss_region']
    
    max_gain2 = df2[df2['malignant_key']=='malignant']['max_gain_region']
    mean_gain2 = df2[df2['malignant_key']=='malignant']['mean_gain_region']
    median_gain2 = df2[df2['malignant_key']=='malignant']['median_gain_region']
    min_gain2 = df2[df2['malignant_key']=='malignant']['min_gain_region']
    max_loss2 = df2[df2['malignant_key']=='malignant']['max_loss_region']
    mean_loss2 =  df2[df2['malignant_key']=='malignant']['mean_loss_region']
    median_loss2 = df2[df2['malignant_key']=='malignant']['median_loss_region']
    min_loss2 = df2[df2['malignant_key']=='malignant']['min_loss_region']
    
    fig, axs = plt.subplots(2, 4)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.07), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim1 = max(get_hist_xlim(max_gain1, percentile), 
               get_hist_xlim(max_loss1, percentile)) 
    xlim2 = max(get_hist_xlim(max_gain2, percentile), 
               get_hist_xlim(max_loss2, percentile)) 
    xlim = max(xlim1, xlim2)
    
    props_list = [get_hist_props(max_gain1, xlim, percentile, normalized=True),
                  get_hist_props(mean_gain1, xlim, percentile, normalized=True),
                  get_hist_props(median_gain1, xlim, percentile, normalized=True),
                  get_hist_props(min_gain1, xlim, percentile, normalized=True),
                  get_hist_props(max_loss1, xlim, percentile, normalized=True),
                  get_hist_props(mean_loss1, xlim, percentile, normalized=True),
                  get_hist_props(median_loss1, xlim, percentile, normalized=True),
                  get_hist_props(min_loss1, xlim, percentile, normalized=True),
                  get_hist_props(max_gain2, xlim, percentile, normalized=True),
                  get_hist_props(mean_gain2, xlim, percentile, normalized=True),
                  get_hist_props(median_gain2, xlim, percentile, normalized=True),
                  get_hist_props(min_gain2, xlim, percentile, normalized=True),
                  get_hist_props(max_loss2, xlim, percentile, normalized=True),
                  get_hist_props(mean_loss2, xlim, percentile, normalized=True),
                  get_hist_props(median_loss2, xlim, percentile, normalized=True),
                  get_hist_props(min_loss2, xlim, percentile, normalized=True)                 ]
    props = merge_props(props_list)

    plot_hist(axs[0, 0], min_gain1,
              'Min. Len. of CNV Region \n per Cell (Gain)', 
              'a', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[0, 0], min_gain2,
              'Min. Len. of CNV Region \n per Cell (Gain)', 
              'a', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(min_gain1, min_gain2)[0],4)
    axs[0,0].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,0].transAxes)
    
    plot_hist(axs[0, 1], mean_gain1, 
              'Mean Len. of CNV Region \n per Cell (Gain)', 
              'b', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[0, 1], mean_gain2, 
              'Mean Len. of CNV Region \n per Cell (Gain)', 
              'b', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(mean_gain1, mean_gain2)[0],4)
    axs[0,1].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    plot_hist(axs[0, 2], median_gain1,
              'Median Len. of CNV Region \n per Cell (Gain)', 
              'c', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[0, 2], median_gain2,
              'Median Len. of CNV Region \n per Cell (Gain)', 
              'c', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(median_gain1, median_gain2)[0],4)
    axs[0,2].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w',
                  transform=axs[0,2].transAxes)
    
    plot_hist(axs[0, 3], max_gain1, 
              'Max. Len. of CNV Region \n per Cell (Gain)', 
              'd', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[0, 3], max_gain2, 
              'Max. Len. of CNV Region \n per Cell (Gain)', 
              'd', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(max_gain1, max_gain2)[0],4)
    axs[0,3].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,3].transAxes)
    
    plot_hist(axs[1, 0], min_loss1,
              'Min. Len. of CNV Region \n per Cell (Loss)', 
              'e', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[1, 0], min_loss2,
              'Min. Len. of CNV Region \n per Cell (Loss)', 
              'e', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(min_loss1, min_loss2)[0],4)
    axs[1,0].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    plot_hist(axs[1, 1], mean_loss1,
              'Mean Len. of CNV Region \n per Cell (Loss)', 
              'f', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[1, 1], mean_loss2,
              'Mean Len. of CNV Region \n per Cell (Loss)', 
              'f', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(mean_loss1, mean_loss2)[0],4)
    axs[1,1].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,1].transAxes)
    
    
    plot_hist(axs[1, 2], median_loss1,
              'Median Len. of CNV Region \n per Cell (Loss)', 
              'g', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[1, 2], median_loss2,
              'Median Len. of CNV Region \n per Cell (Loss)', 
              'g', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(median_loss1, median_loss2)[0],4)
    axs[1,2].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,2].transAxes)
    
    plot_hist(axs[1, 3], max_loss1,
              'Max. Len. of CNV Region \n per Cell (Loss)', 
              'h', 'Length of CNV (bp)', props, '#0000bb', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    plot_hist(axs[1, 3], max_loss2,
              'Max. Len. of CNV Region \n per Cell (Loss)', 
              'h', 'Length of CNV (bp)', props, '#aa0000', 
              plot_shape=(2, 4), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(max_loss1, max_loss2)[0],4)
    axs[1,3].text(0.48, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,3].transAxes)
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    

def plot_nregions_per_cell(df1, df2, percentile, output_dir, output_name,
                           df1_name, df2_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots show the histograms of number of CNV regions
        per cell for gain and loss CNV types, for two datasets.
        
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['malignant_key', 'num_gain_region', 'num_loss_region']
    if not is_dataframe_valid(df1, col_names):
        return     
    if not is_dataframe_valid(df2, col_names):
        return     
    gain1 = df1[df1['malignant_key']=='malignant']['num_gain_region']
    loss1 = df1[df1['malignant_key']=='malignant']['num_loss_region']    
    gain2 = df2[df2['malignant_key']=='malignant']['num_gain_region']
    loss2 = df2[df2['malignant_key']=='malignant']['num_loss_region']    
    fig, axs = plt.subplots(1, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.2), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim1 = max(get_hist_xlim(gain1, percentile),
               get_hist_xlim(loss1, percentile))
    xlim2 = max(get_hist_xlim(gain2, percentile),
               get_hist_xlim(loss2, percentile))
    xlim = max(xlim1, xlim2)
    props_list = [get_hist_props(gain1, xlim, percentile, normalized=True), 
                  get_hist_props(loss1, xlim, percentile, normalized=True),
                  get_hist_props(gain2, xlim, percentile, normalized=True), 
                  get_hist_props(loss2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
    
    plot_hist(axs[0], gain1, 'CNV Regions per Cells (Gain)', 'a',
              'Number of CNV Regions', props, color='#0000bb',
              plot_shape=(1, 2), alpha=0.3, density=True)              
    plot_hist(axs[0], gain2, 'CNV Regions per Cells (Gain)', 'a',
              'Number of CNV Regions', props, color='#aa0000',
              plot_shape=(1, 2), alpha=0.3, density=True) 
    ks = round(stats.ks_2samp(gain1, gain2)[0],4)
    axs[0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                transform=axs[0].transAxes)
    
    plot_hist(axs[1], loss1, 'CNV Regions per Cells (Loss)', 'b', 
              'Number of CNV Regions', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)              
    plot_hist(axs[1], loss2, 'CNV Regions per Cells (Loss)', 'b', 
              'Number of CNV Regions', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(loss1, loss2)[0],4)
    axs[1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                transform=axs[1].transAxes)
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")



def plot_cnv_per_cell(df1, df2, percentile, output_dir, output_name,
                      df1_name, df2_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots show the histograms of length of gain and
        loss CNV regions as measured by base-pairs and genes for 
        two datasets.
        
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['malignant_key', 'total_gains_bp', 'total_losses_bp',
                'total_gains_genes', 'total_losses_genes']
    if not is_dataframe_valid(df1, col_names):
        return         
    if not is_dataframe_valid(df2, col_names):
        return         
    gain_bp1 = df1[df1['malignant_key']=='malignant']['total_gains_bp']
    loss_bp1 = df1[df1['malignant_key']=='malignant']['total_losses_bp']
    gain_gene1 = df1[df1['malignant_key']=='malignant']['total_gains_genes']
    loss_gene1 = df1[df1['malignant_key']=='malignant']['total_losses_genes']
    gain_bp2 = df2[df2['malignant_key']=='malignant']['total_gains_bp']
    loss_bp2 = df2[df2['malignant_key']=='malignant']['total_losses_bp']
    gain_gene2 = df2[df2['malignant_key']=='malignant']['total_gains_genes']
    loss_gene2 = df2[df2['malignant_key']=='malignant']['total_losses_genes']
    fig, axs = plt.subplots(2, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.07), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)        
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=None, wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram

    xlim_bp1 = max(get_hist_xlim(gain_bp1, percentile),
               get_hist_xlim(loss_bp1, percentile))
    xlim_bp2 = max(get_hist_xlim(gain_bp2, percentile),
               get_hist_xlim(loss_bp2, percentile))
    xlim_bp = max(xlim_bp1, xlim_bp2)
    props_list_bp = [get_hist_props(gain_bp1, xlim_bp, percentile, normalized=True), 
                     get_hist_props(loss_bp1, xlim_bp, percentile, normalized=True),
                     get_hist_props(gain_bp2, xlim_bp, percentile, normalized=True), 
                     get_hist_props(loss_bp2, xlim_bp, percentile, normalized=True)]
    props_bp = merge_props(props_list_bp)
    
    plot_hist(axs[0, 0], gain_bp1, 'CNV per Cell (Gain)', 
              'a', 'Length of CNV (bp)', props_bp, color='#0000bb',
              plot_shape=(2, 2), alpha=0.3, density=True)              
    plot_hist(axs[0, 0], gain_bp2, 'CNV per Cell (Gain)', 
              'a', 'Length of CNV (bp)', props_bp, color='#aa0000',
              plot_shape=(2, 2), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(gain_bp1, gain_bp2)[0],4)
    axs[0,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,0].transAxes)
    
    
    plot_hist(axs[0, 1], loss_bp1, 'CNV per Cell (Loss)', 
              'b', 'Length of CNV (bp)', props_bp, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)              
    plot_hist(axs[0, 1], loss_bp2, 'CNV per Cell (Loss)', 
              'b', 'Length of CNV (bp)', props_bp, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(loss_bp1, loss_bp2)[0],4)
    axs[0,1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    xlim_gene1 = max(get_hist_xlim(gain_gene1, percentile),
               get_hist_xlim(loss_gene1, percentile))
    xlim_gene2 = max(get_hist_xlim(gain_gene2, percentile),
               get_hist_xlim(loss_gene2, percentile))
    xlim_gene = max(xlim_gene1, xlim_gene2)
    props_list_gene = [get_hist_props(gain_gene1, xlim_gene, percentile, normalized=True), 
                       get_hist_props(loss_gene1, xlim_gene, percentile, normalized=True),
                       get_hist_props(gain_gene2, xlim_gene, percentile, normalized=True), 
                       get_hist_props(loss_gene2, xlim_gene, percentile, normalized=True)]
    props_gene = merge_props(props_list_gene)
    plot_hist(axs[1, 0], gain_gene1, 'CNV per Cell (Gain)', 
              'c', 'Length of CNV (gene)', props_gene, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)              
    plot_hist(axs[1, 0], gain_gene2, 'CNV per Cell (Gain)', 
              'c', 'Length of CNV (gene)', props_gene, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(gain_gene1, gain_gene2)[0],4)
    axs[1,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    plot_hist(axs[1, 1], loss_gene1, 'CNV per Cell (Loss)', 
              'd', 'Length of CNV (gene)', props_gene, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)      
    plot_hist(axs[1, 1], loss_gene2, 'CNV per Cell (Loss)', 
              'd', 'Length of CNV (gene)', props_gene, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True)     
    ks = round(stats.ks_2samp(loss_gene1, loss_gene2)[0],4)
    axs[1,1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,1].transAxes)
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    

def plot_expression_stats(df1, df2, percentile, output_dir, output_name,
                          df1_name, df2_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots are histograms of a) Mean of logCPM,
        b) Variance of logCPM, c) coefficient of variation,
        d) fraction of detected cells, for two datasets
        
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['mean_logCPM', 'var_logCPM', 'cv', 'zeroes_cells']
    if not is_dataframe_valid(df1, col_names):
        return     
    if not is_dataframe_valid(df2, col_names):
        return     
    mean1 = df1['mean_logCPM']
    variance1 = df1['var_logCPM']
    variability1 = df1['cv']
    fraction1 = df1['zeroes_cells']    
    mean2 = df2['mean_logCPM']
    variance2 = df2['var_logCPM']
    variability2 = df2['cv']
    fraction2 = df2['zeroes_cells']    
    fig, axs = plt.subplots(2, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.07), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = max(get_hist_xlim(mean1, percentile), 
               get_hist_xlim(mean2, percentile))
    props_list = [get_hist_props(mean1, xlim, percentile, normalized=True), 
                  get_hist_props(mean2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
        
    plot_hist(axs[0, 0], mean1, 'Mean of Gene Expression', 
              'a', 'Mean of logCPM', props, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)              
    plot_hist(axs[0, 0], mean2, 'Mean of Gene Expression', 
              'a', 'Mean of logCPM', props, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True)  
    ks = round(stats.ks_2samp(mean1, mean2)[0],4)
    axs[0,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,0].transAxes)

    xlim = max(get_hist_xlim(variance1, percentile), 
               get_hist_xlim(variance2, percentile))
    props_list = [get_hist_props(variance1, xlim, percentile, normalized=True), 
                  get_hist_props(variance2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[0, 1], variance1, 'Variance of Gene Expression', 
              'b', 'Variance of logCPM', props, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)     
    plot_hist(axs[0, 1], variance2, 'Variance of Gene Expression', 
              'b', 'Variance of logCPM', props, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True)   
    ks = round(stats.ks_2samp(variance1, variance2)[0],4)
    axs[0,1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    xlim = max(get_hist_xlim(variability1, percentile), 
               get_hist_xlim(variability2, percentile))
    props_list = [get_hist_props(variability1, xlim, percentile, normalized=True),
                  get_hist_props(variability2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)    
    plot_hist(axs[1, 0], variability1, 
              'Expression Variability\nRelative to Its Mean', 
              'c', 'Coefficient of Variation', props, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)     
    plot_hist(axs[1, 0], variability2, 
              'Expression Variability\nRelative to Its Mean', 
              'c', 'Coefficient of Variation', props, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(variability1, variability2)[0],4)
    axs[1,0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    xlim = max(get_hist_xlim(fraction1, percentile), 
               get_hist_xlim(fraction2, percentile))
    props_list = [get_hist_props(fraction1, xlim, percentile, normalized=True),
                  get_hist_props(fraction2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[1, 1], fraction1, 'Gene Detection Frequency', 
              'd', 'Fraction of Detected Cells', props, color='#0000bb', 
              plot_shape=(2, 2), alpha=0.3, density=True)     
    plot_hist(axs[1, 1], fraction2, 'Gene Detection Frequency', 
              'd', 'Fraction of Detected Cells', props, color='#aa0000', 
              plot_shape=(2, 2), alpha=0.3, density=True) 
    ks = round(stats.ks_2samp(fraction1, fraction2)[0],4)
    axs[1,1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,1].transAxes)
              
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")


def plot_gene2gene_corr(df1, df2, output_dir, output_name,
                        df1_name, df2_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots are histograms of a) Gene-to-Gene 
        correlation for the top-400 most variable genes, and b) 
        Gene-to-Gene correlation for the top 400 
        highly-expressed genes for two datasets.
        
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                output_dir: directory to save plots in
                output_name: file name of plot    
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['hv', 'he']
    if not is_dataframe_valid(df1, col_names):
        return     
    if not is_dataframe_valid(df2, col_names):
        return     
    most_var1 = df1['hv']
    most_expr1 = df1['he']
    most_var2 = df2['hv']
    most_expr2 = df2['he']
    fig, axs = plt.subplots(1, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.2), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = 1
    props_list = [get_hist_props(most_var1, xlim, normalized=True), 
                  get_hist_props(most_expr1, xlim, normalized=True),
                  get_hist_props(most_var2, xlim, normalized=True), 
                  get_hist_props(most_expr2, xlim, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[0], most_var1, 
              'Gene-to-Gene Correlation\n Top 400 Most Variable Genes', 
              'a', 'Pearson Correlation', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    plot_hist(axs[0], most_var2, 
              'Gene-to-Gene Correlation\n Top 400 Most Variable Genes', 
              'a', 'Pearson Correlation', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    ks = round(stats.ks_2samp(most_var1, most_var2)[0],4)
    axs[0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                transform=axs[0].transAxes)
    
    plot_hist(axs[1], most_expr1, 
              'Gene-to-Gene Correlation\n  Top 400 Highly-Expressed Genes', 
              'b', 'Pearson Correlation', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    plot_hist(axs[1], most_expr2, 
              'Gene-to-Gene Correlation\n  Top 400 Highly-Expressed Genes', 
              'b', 'Pearson Correlation', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)
    ks = round(stats.ks_2samp(most_expr1, most_expr2)[0],4)
    axs[1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                transform=axs[1].transAxes)
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")



def plot_cell_counts(df1, df2, output_dir, output_name,
                     df1_name, df2_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots are histograms of a) log-Library size
        , and b) fraction of detected genes, for two datasets.
        
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                output_dir: directory to save plots in
                output_name: file name of plot
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend                
    """
    col_names = ['log1p_total_counts', 'zeroes_genes']
    if not is_dataframe_valid(df1, col_names):
        return   
    if not is_dataframe_valid(df2, col_names):
        return   
    log_counts1 = df1['log1p_total_counts']
    detection_freq1 = df1['zeroes_genes']
    log_counts2 = df2['log1p_total_counts']
    detection_freq2 = df2['zeroes_genes']
    fig, axs = plt.subplots(1, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.2), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = max(get_hist_xlim(log_counts1), get_hist_xlim(log_counts1))
    props_list = [get_hist_props(log_counts1, xlim, normalized=True), 
                  get_hist_props(log_counts2, xlim, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[0], log_counts1, 'log1p-Transformed\n Total Counts', 
              'a', 'log-Library Size', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    plot_hist(axs[0], log_counts2, 'log1p-Transformed\n Total Counts', 
              'a', 'log-Library Size', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)   
    ks = round(stats.ks_2samp(log_counts1, log_counts2)[0],4)
    axs[0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                transform=axs[0].transAxes)
    
    props_list = [get_hist_props(detection_freq1, 1, normalized=True), 
                  get_hist_props(detection_freq2, 1, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[1], detection_freq1,'Cell Detection Frequency' , 
              'b', 'Fraction of Detected Genes', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    plot_hist(axs[1], detection_freq2,'Cell Detection Frequency' , 
              'b', 'Fraction of Detected Genes', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    ks = round(stats.ks_2samp(detection_freq1, detection_freq2)[0],4)
    axs[1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                transform=axs[1].transAxes)
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    
def plot_cell2cell_corr(df1, df2, output_dir, output_name,
                        df1_name, df2_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots are histograms of a) cell-to-cell correlation
        for the top-400 highly-variable genes, and b) cell-to-cell
        correlation for all genes, for two datasets
        
            Args:
                df1: dataframe that includes plot information for the 
                            first dataset
                df2: dataframe that includes plot information for the 
                            second dataset
                output_dir: directory to save plots in
                output_name: file name of plot     
                df1_name: Name of dataframe 1 to be displayed in legend
                df2_name: Name of dataframe 2 to be displayed in legend
    """
    col_names = ['hv', 'all']
    if not is_dataframe_valid(df1, col_names):
        return   
    if not is_dataframe_valid(df2, col_names):
        return   
    most_var_genes1 = df1['hv']
    all_genes1 = df1['all']
    most_var_genes2 = df2['hv']
    all_genes2 = df2['all']    
    fig, axs = plt.subplots(1, 2)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, -0.2), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.7, hspace=0.7)

    #set xlim and ylim according to the total histogram
    props_list = [get_hist_props(most_var_genes1, 1, normalized=True),
                  get_hist_props(most_var_genes2, 1, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[0], most_var_genes1, 
              'Cell-to-Cell Correlation \n Top-400 Highly-Variable Genes', 
              'a', 'Pearson Correlation', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)
    plot_hist(axs[0], most_var_genes2, 
              'Cell-to-Cell Correlation \n Top-400 Highly-Variable Genes', 
              'a', 'Pearson Correlation', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    ks = round(stats.ks_2samp(most_var_genes1, most_var_genes2)[0],4)
    axs[0].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w',
                transform=axs[0].transAxes)
    
    props_list = [get_hist_props(all_genes1, 1, normalized=True), 
                  get_hist_props(all_genes2, 1, normalized=True)]
    props = merge_props(props_list)
    plot_hist(axs[1], all_genes1,'Cell-to-Cell Correlation \n All Genes' , 
              'b', 'Pearson Correlation', props, color='#0000bb', 
              plot_shape=(1, 2), alpha=0.3, density=True)     
    plot_hist(axs[1], all_genes2,'Cell-to-Cell Correlation \n All Genes' , 
              'b', 'Pearson Correlation', props, color='#aa0000', 
              plot_shape=(1, 2), alpha=0.3, density=True)   
    ks = round(stats.ks_2samp(all_genes1, all_genes2)[0],4)
    axs[1].text(0.66, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w',
                transform=axs[1].transAxes)
    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    
def plot_summary_cnv(df_cnv1, df_cnv2, df_subclonal1, df_subclonal2,
                  percentile, output_dir, output_name, df1_name, df2_name):
    """
        Plots a figure with 12 sub-plots (4 rows, 3 columns) summarizing 
        the comparison of cancer-specific cnv metrics.
            Args:
                df_*_*1: dataframes of dataset 1 needed for plotting
                df_*_*2: dataframes of dataset 2 needed for plotting
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot     
                df1_name: Name of dataset 1 to be displayed in legend
                df2_name: Name of dataset 2 to be displayed in legend
    """
    #sanity check
    col_names_part1 = ['type', 'size_bp_total']
    col_names_part2 = ['malignant_key', 'max_gain_region', 'mean_gain_region',
                       'median_gain_region', 'min_gain_region', 'max_loss_region',
                       'mean_loss_region', 'median_loss_region', 'min_loss_region']
    col_names_part2 += ['malignant_key', 'num_gain_region', 'num_loss_region']
    col_names_part2 += ['malignant_key', 'total_gains_bp', 'total_losses_bp',
                       'total_gains_genes', 'total_losses_genes']
    if not is_dataframe_valid(df_cnv1, col_names_part1):
        return   
    if not is_dataframe_valid(df_cnv2, col_names_part1):
        return   
    if not is_dataframe_valid(df_subclonal1, col_names_part2):
        return   
    if not is_dataframe_valid(df_subclonal2, col_names_part2):
        return   
    #set legend
    fig, axs = plt.subplots(4, 3)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, 0.05), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.3, hspace=0.3)
    
    #start plotting
    gain1 = df_cnv1[df_cnv1['type']=='gain']['size_bp_total']
    gain2 = df_cnv2[df_cnv2['type']=='gain']['size_bp_total']    
    xlim_gain1 = get_hist_xlim(gain1, percentile)
    xlim_gain2 = get_hist_xlim(gain2, percentile)
    xlim_gain = max(xlim_gain1, xlim_gain2)
    loss1 = df_cnv1[df_cnv1['type']=='loss']['size_bp_total']
    loss2 = df_cnv2[df_cnv2['type']=='loss']['size_bp_total']        
    xlim_loss1 = get_hist_xlim(loss1, percentile)
    xlim_loss2 = get_hist_xlim(loss2, percentile)
    xlim_loss = max(xlim_loss1, xlim_loss2)
    xlim = max(xlim_loss, xlim_gain)
    props_list = [get_hist_props(gain1, xlim, percentile, normalized=True),
                  get_hist_props(gain2, xlim, percentile, normalized=True),
                  get_hist_props(loss1, xlim, percentile, normalized=True),
                  get_hist_props(loss2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)    
    plot_violin_comp(axs[0, 0], gain1, gain2, 
                     title='Copy Number Variation (Gain)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')
    
    ks = round(stats.ks_2samp(gain1, gain2)[0],4)
    axs[0,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w',
                  transform=axs[0,0].transAxes)
    
    
    plot_violin_comp(axs[1, 0], loss1, loss2, 
                     title='Copy Number Variation (Loss)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')
    stat = stats.ks_2samp(loss1, loss2)
    ks = round(stat[0], 4)
    axs[1,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    
    df1 = df_subclonal1
    df2 = df_subclonal2
    max_gain1 = df1[df1['malignant_key']=='malignant']['max_gain_region']
    mean_gain1 = df1[df1['malignant_key']=='malignant']['mean_gain_region']
    min_gain1 = df1[df1['malignant_key']=='malignant']['min_gain_region']
    max_loss1 = df1[df1['malignant_key']=='malignant']['max_loss_region']
    mean_loss1 =  df1[df1['malignant_key']=='malignant']['mean_loss_region']
    min_loss1 = df1[df1['malignant_key']=='malignant']['min_loss_region']
    max_gain2 = df2[df2['malignant_key']=='malignant']['max_gain_region']
    mean_gain2 = df2[df2['malignant_key']=='malignant']['mean_gain_region']
    min_gain2 = df2[df2['malignant_key']=='malignant']['min_gain_region']
    max_loss2 = df2[df2['malignant_key']=='malignant']['max_loss_region']
    mean_loss2 =  df2[df2['malignant_key']=='malignant']['mean_loss_region']
    min_loss2 = df2[df2['malignant_key']=='malignant']['min_loss_region']
    #set xlim and ylim according to the total histogram
    xlim1 = max(get_hist_xlim(max_gain1, percentile), 
               get_hist_xlim(max_loss1, percentile)) 
    xlim2 = max(get_hist_xlim(max_gain2, percentile), 
               get_hist_xlim(max_loss2, percentile)) 
    xlim = max(xlim1, xlim2)    
    props_list = [get_hist_props(max_gain1, xlim, percentile, normalized=True),
                  get_hist_props(mean_gain1, xlim, percentile, normalized=True),
                  get_hist_props(min_gain1, xlim, percentile, normalized=True),
                  get_hist_props(max_loss1, xlim, percentile, normalized=True),
                  get_hist_props(mean_loss1, xlim, percentile, normalized=True),
                  get_hist_props(min_loss1, xlim, percentile, normalized=True),
                  get_hist_props(max_gain2, xlim, percentile, normalized=True),
                  get_hist_props(mean_gain2, xlim, percentile, normalized=True),
                  get_hist_props(min_gain2, xlim, percentile, normalized=True),
                  get_hist_props(max_loss2, xlim, percentile, normalized=True),
                  get_hist_props(mean_loss2, xlim, percentile, normalized=True),
                  get_hist_props(min_loss2, xlim, percentile, normalized=True)                 ]
    props = merge_props(props_list)    
    plot_violin_comp(axs[2, 2], max_gain1, max_gain2, 
                     title='Max. Len. of CNV Region \n per Cell (Gain)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(max_gain1, max_gain2)[0],4)
    axs[2,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[2,2].transAxes)
    
    plot_violin_comp(axs[3, 2], max_loss1, max_loss2, 
                     title='Max. Len. of CNV Region \n per Cell (Loss)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(max_loss1, max_loss2)[0],4)
    axs[3,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[3,2].transAxes)
    
    plot_violin_comp(axs[2, 1], mean_gain1, mean_gain2, 
                     title='Mean Len. of CNV Region \n per Cell (Gain)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(mean_gain1, mean_gain2)[0],4)
    axs[2,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[2,1].transAxes)
    
    plot_violin_comp(axs[3, 1], mean_loss1, mean_loss2, 
                     title='Mean Len. of CNV Region \n per Cell (Loss)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(mean_loss1, mean_loss2)[0],4)
    axs[3,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[3,1].transAxes)
    
    plot_violin_comp(axs[2, 0], min_gain1, min_gain2, 
                     title='Min. Len. of CNV Region \n per Cell (Gain)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(min_gain1, min_gain2)[0],4)
    axs[2,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[2,0].transAxes)
    
    plot_violin_comp(axs[3, 0], min_loss1, min_loss2, 
                     title='Min. Len. of CNV Region \n per Cell (Loss)', 
                     label='Length of CNV (bp)', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(min_loss1, min_loss2)[0],4)
    axs[3,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[3,0].transAxes)
    
    gain1 = df1[df1['malignant_key']=='malignant']['num_gain_region']
    loss1 = df1[df1['malignant_key']=='malignant']['num_loss_region']    
    gain2 = df2[df2['malignant_key']=='malignant']['num_gain_region']
    loss2 = df2[df2['malignant_key']=='malignant']['num_loss_region']    
    xlim1 = max(get_hist_xlim(gain1, percentile),
               get_hist_xlim(loss1, percentile))
    xlim2 = max(get_hist_xlim(gain2, percentile),
               get_hist_xlim(loss2, percentile))
    xlim = max(xlim1, xlim2)
    props_list = [get_hist_props(gain1, xlim, percentile, normalized=True), 
                  get_hist_props(loss1, xlim, percentile, normalized=True),
                  get_hist_props(gain2, xlim, percentile, normalized=True), 
                  get_hist_props(loss2, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
    plot_violin_comp(axs[0, 1], gain1, gain2, 
                     title='CNV Regions per Cell (Gain)', 
                     label='Number of CNV Regions', props=props, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(gain1, gain2)[0],4)
    axs[0,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    plot_violin_comp(axs[1, 1], loss1, loss2, 
                     title='CNV Regions per Cell (Loss)', 
                     label='Number of CNV Regions', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(loss1, loss2)[0],4)
    axs[1,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,1].transAxes)
    
    gain_bp1 = df1[df1['malignant_key']=='malignant']['total_gains_bp']
    loss_bp1 = df1[df1['malignant_key']=='malignant']['total_losses_bp']
    gain_bp2 = df2[df2['malignant_key']=='malignant']['total_gains_bp']
    loss_bp2 = df2[df2['malignant_key']=='malignant']['total_losses_bp']
    xlim_bp1 = max(get_hist_xlim(gain_bp1, percentile),
               get_hist_xlim(loss_bp1, percentile))
    xlim_bp2 = max(get_hist_xlim(gain_bp2, percentile),
               get_hist_xlim(loss_bp2, percentile))
    xlim_bp = max(xlim_bp1, xlim_bp2)
    props_list_bp = [get_hist_props(gain_bp1, xlim_bp, percentile, normalized=True), 
                     get_hist_props(loss_bp1, xlim_bp, percentile, normalized=True),
                     get_hist_props(gain_bp2, xlim_bp, percentile, normalized=True), 
                     get_hist_props(loss_bp2, xlim_bp, percentile, normalized=True)]
    props_bp = merge_props(props_list_bp)
    
    plot_violin_comp(axs[0, 2], gain_bp1, gain_bp2, 
                     title='CNV per Cell (Gain)', 
                     label='Length of CNV (bp)', props=props_bp, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(gain_bp1, gain_bp2)[0],4)
    axs[0,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,2].transAxes)
    
    plot_violin_comp(axs[1, 2], loss_bp1, loss_bp2, 
                     title='CNV per Cell (Loss)', 
                     label='Length of CNV (bp)', props=props_bp, 
                     color1='#0000bb', color2='#aa0000')   
    
    ks = round(stats.ks_2samp(loss_bp1, loss_bp2)[0],4)
    axs[1,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,2].transAxes)
    
    

    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    

    
    
    
    
    
    
def plot_summary_normal(df_gene_all1, df_gene_all2, df_gene_mal1, df_gene_mal2,
                  df_counts_all1, df_counts_all2, df_counts_mal1, df_counts_mal2,
                  df_corr_all1, df_corr_all2, df_corr_mal1, df_corr_mal2,                      
                  percentile, output_dir, output_name, df1_name, df2_name):
    """
        Plots a figure with 12 sub-plots (4 rows, 3 columns) summarizing 
        the comparison of non-cancer-specific gene- and cell-wise metrics.
        The sub-plots are 2 violin plots for each of the dataset, displaying 
        mean and variance of gene expression, gene and cell detection frequency, 
        log1p total counts and cell-to-cell correlation. Once for all cells and 
        once for only malignant cells. 
            Args:
                df_*_*1: dataframes of dataset 1 needed for plotting
                df_*_*2: dataframes of dataset 2 needed for plotting
                percentile: only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot     
                df1_name: Name of dataset 1 to be displayed in legend
                df2_name: Name of dataset 2 to be displayed in legend
    """
    #sanity check
    col_names = ['mean_logCPM', 'var_logCPM', 'zeroes_cells']
    
    if not is_dataframe_valid(df_gene_all1, col_names):
        return     
    if not is_dataframe_valid(df_gene_all2, col_names):
        return     
    if not is_dataframe_valid(df_gene_mal1, col_names):
        return     
    if not is_dataframe_valid(df_gene_mal2, col_names):
        return  
    #set legend
    fig, axs = plt.subplots(4, 3)
    legend_lines = [Line2D([0], [0], color='#0000bb4c', lw=8),
                    Line2D([0], [0], color='#aa00004c', lw=8)]
    legend = fig.legend(legend_lines, [df1_name, df2_name], 
               fontsize=18, loc='lower center', 
               bbox_to_anchor=(0.5, 0.05), ncol=2)
    legend.get_frame().set_edgecolor('#000000')
    legend.get_frame().set_boxstyle('square', pad=0.2)    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.3, hspace=0.3)
    mean1 = df_gene_all1['mean_logCPM']
    variance1 = df_gene_all1['var_logCPM']
    fraction1 = df_gene_all1['zeroes_cells']    
    mean2 = df_gene_all2['mean_logCPM']
    variance2 = df_gene_all2['var_logCPM']
    fraction2 = df_gene_all2['zeroes_cells']    
    
    mean1m = df_gene_mal1['mean_logCPM']
    variance1m = df_gene_mal1['var_logCPM']
    fraction1m = df_gene_mal1['zeroes_cells']    
    mean2m = df_gene_mal2['mean_logCPM']
    variance2m = df_gene_mal2['var_logCPM']
    fraction2m = df_gene_mal2['zeroes_cells']  
    
    #set xlim and ylim according to the total histogram
    xlim = max(max(get_hist_xlim(mean1, percentile), 
               get_hist_xlim(mean2, percentile)), 
               max(get_hist_xlim(mean1m, percentile), 
               get_hist_xlim(mean2m, percentile)))
    props_list = [get_hist_props(mean1, xlim, percentile, normalized=True), 
                  get_hist_props(mean2, xlim, percentile, normalized=True),
                  get_hist_props(mean1m, xlim, percentile, normalized=True), 
                  get_hist_props(mean2m, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
    
    plot_violin_comp(axs[0, 0], mean1, mean2, 
                     title='Mean of Gene Expression', 
                     label='Mean of logCPM', props=props, 
                     color1='#0000bb', color2='#aa0000')  
    ks = round(stats.ks_2samp(mean1, mean2)[0],4)
    axs[0,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,0].transAxes)
    
    plot_violin_comp(axs[1, 0], mean1m, mean2m, 
                     title='Mean of Gene Expression', 
                     label='Mean of logCPM', props=props, 
                     color1='#0000bb', color2='#aa0000')  
    ks = round(stats.ks_2samp(mean1m, mean2m)[0],4)
    axs[1,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,0].transAxes)
    
    xlim = max(max(get_hist_xlim(variance1, percentile), 
               get_hist_xlim(variance2, percentile)),
               max(get_hist_xlim(variance1m, percentile), 
               get_hist_xlim(variance2m, percentile)))
    props_list = [get_hist_props(variance1, xlim, percentile, normalized=True), 
                  get_hist_props(variance2, xlim, percentile, normalized=True),
                  get_hist_props(variance1m, xlim, percentile, normalized=True), 
                  get_hist_props(variance2m, xlim, percentile, normalized=True)]
    props = merge_props(props_list)
    
    plot_violin_comp(axs[0, 1], variance1, variance2, 
                     title='Variance of Gene Expression', 
                     label='Variance of logCPM', props=props, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(variance1, variance2)[0],4)
    axs[0,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,1].transAxes)
    
    plot_violin_comp(axs[1, 1], variance1m, variance2m, 
                     title='Variance of Gene Expression', 
                     label='Variance of logCPM', props=props, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(variance1m, variance2m)[0],4)
    axs[1,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,1].transAxes)
    
    xlim = max(get_hist_xlim(fraction1, percentile), 
               get_hist_xlim(fraction2, percentile),
               get_hist_xlim(fraction1m, percentile), 
               get_hist_xlim(fraction2m, percentile))
    props_list = [get_hist_props(fraction1, xlim, percentile, normalized=True),
                  get_hist_props(fraction2, xlim, percentile, normalized=True),
                  get_hist_props(fraction1m, xlim, percentile, normalized=True),
                  get_hist_props(fraction2m, xlim, percentile, normalized=True),]
    props = merge_props(props_list)     
    
    plot_violin_comp(axs[0, 2], fraction1, fraction2, 
                     title='Gene Detection Frequency', 
                     label='Fraction of Detected Cells', props=props, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(fraction1, fraction2)[0],4)
    axs[0,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[0,2].transAxes)
    
    plot_violin_comp(axs[1, 2], fraction1m, fraction2m, 
                     title='Gene Detection Frequency', 
                     label='Fraction of Detected Cells', props=props, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(fraction1m, fraction2m)[0],4)
    axs[1,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[1,2].transAxes)
    
    
    col_names = ['log1p_total_counts', 'zeroes_genes']
    if not is_dataframe_valid(df_counts_all1, col_names):
        return   
    if not is_dataframe_valid(df_counts_all2, col_names):
        return   
    if not is_dataframe_valid(df_counts_mal1, col_names):
        return   
    if not is_dataframe_valid(df_counts_mal2, col_names):
        return   
    log_counts_all1 = df_counts_all1['log1p_total_counts']
    detection_freq_all1 = df_counts_all1['zeroes_genes']
    log_counts_all2 = df_counts_all2['log1p_total_counts']
    detection_freq_all2 = df_counts_all2['zeroes_genes']
    log_counts_mal1 = df_counts_mal1['log1p_total_counts']
    detection_freq_mal1 = df_counts_mal1['zeroes_genes']
    log_counts_mal2 = df_counts_mal2['log1p_total_counts']
    detection_freq_mal2 = df_counts_mal2['zeroes_genes']
    #set xlim and ylim according to the total histogram
    xlim = max(max(get_hist_xlim(log_counts_all1), get_hist_xlim(log_counts_all2)),
               max(get_hist_xlim(log_counts_mal1), get_hist_xlim(log_counts_mal2)))
    props_list = [get_hist_props(log_counts_all1, xlim, normalized=True), 
                  get_hist_props(log_counts_all2, xlim, normalized=True),
                  get_hist_props(log_counts_mal1, xlim, normalized=True), 
                  get_hist_props(log_counts_mal2, xlim, normalized=True)]
    props = merge_props(props_list)    
    
    plot_violin_comp(axs[2, 0], log_counts_all1, log_counts_all2, 
                     title='log1p-Transformed\n Total Counts', 
                     label='log-Library Size', props=props, 
                     color1='#0000bb', color2='#aa0000')  
    ks = round(stats.ks_2samp(log_counts_all1, log_counts_all2)[0],4)
    axs[2,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[2,0].transAxes)
    
    plot_violin_comp(axs[3, 0], log_counts_mal1, log_counts_mal2, 
                     title='log1p-Transformed\n Total Counts', 
                     label='log-Library Size', props=props, 
                     color1='#0000bb', color2='#aa0000') 
    ks = round(stats.ks_2samp(log_counts_mal1, log_counts_mal2)[0],4)
    axs[3,0].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[3,0].transAxes)
    
    props_list = [get_hist_props(detection_freq_all1, 1, normalized=True), 
                  get_hist_props(detection_freq_all2, 1, normalized=True),
                  get_hist_props(detection_freq_mal1, 1, normalized=True), 
                  get_hist_props(detection_freq_mal2, 1, normalized=True)]
    props = merge_props(props_list)  
    
    plot_violin_comp(axs[2, 1], detection_freq_all1, detection_freq_all2, 
                     title='Cell Detection Frequency', 
                     label='Fraction of Detected Genes', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(detection_freq_all1, detection_freq_all2)[0],4)
    axs[2,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[2,1].transAxes)
    
    plot_violin_comp(axs[3, 1], detection_freq_mal1, detection_freq_mal2, 
                     title='Cell Detection Frequency', 
                     label='Fraction of Detected Genes', props=props, 
                     color1='#0000bb', color2='#aa0000')
    ks = round(stats.ks_2samp(detection_freq_mal1, detection_freq_mal2)[0],4)
    axs[3,1].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[3,1].transAxes)
    
    
    col_names = ['all']
    if not is_dataframe_valid(df_corr_all1, col_names):
        return   
    if not is_dataframe_valid(df_corr_all2, col_names):
        return   
    if not is_dataframe_valid(df_corr_mal1, col_names):
        return   
    if not is_dataframe_valid(df_corr_mal2, col_names):
        return   
    genes_all1 = df_corr_all1['all']
    genes_all2 = df_corr_all2['all']    
    genes_mal1 = df_corr_mal1['all']
    genes_mal2 = df_corr_mal2['all']    
    props_list = [get_hist_props(genes_all1, 1, normalized=True), 
                  get_hist_props(genes_all2, 1, normalized=True),
                  get_hist_props(genes_mal1, 1, normalized=True), 
                  get_hist_props(genes_mal2, 1, normalized=True)]
    props = merge_props(props_list)
    plot_violin_comp(axs[2, 2], genes_all1, genes_all2, 
                     title='Cell-to-Cell Correlation \n All Genes', 
                     label='Pearson Correlation', props=props, 
                     color1='#0000bb', color2='#aa0000')  
    ks = round(stats.ks_2samp(genes_all1, genes_all2)[0],4)
    axs[2,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[2,2].transAxes)
    
    plot_violin_comp(axs[3, 2], genes_mal1, genes_mal2, 
                     title='Cell-to-Cell Correlation \n All Genes', 
                     label='Pearson Correlation', props=props, 
                     color1='#0000bb', color2='#aa0000')    
    ks = round(stats.ks_2samp(genes_mal1, genes_mal2)[0],4)
    axs[3,2].text(0.63, 0.88, 'KS = ' + str(ks), {'fontsize': 16}, backgroundcolor = 'w', 
                  transform=axs[3,2].transAxes)

    #start plotting
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
        
    
    
    
    
    
    
    
    
    
    
    
    