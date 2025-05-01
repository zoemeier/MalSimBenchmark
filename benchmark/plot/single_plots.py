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
from benchmark.plot.plot_utils import *



def plot_cnv_bp(df, percentile, output_dir, output_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots a, b, c are the histograms of length of
        CNVs, measured in base pairs (bp): a) gain CNVs in 
        blue, b) loss CNVs in red, and all CNVs 
        (gain and loss) in black. Sub-plot d shows the 
        cumulative distribution of gain, loss, total in one
        plot.
        
            Args:
                df: Dataframe that includes plot information.
                percentile: Only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    col_names = ['type', 'size_bp_total']
    if not is_dataframe_valid(df, col_names):
        return 
    gain = df[df['type']=='gain']['size_bp_total']
    loss = df[df['type']=='loss']['size_bp_total']
    total = df['size_bp_total']
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=None, wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = get_hist_xlim(df['size_bp_total'], percentile)
    props = get_hist_props(df['size_bp_total'], xlim, percentile)
    plot_hist(axs[0,0], gain, 'Copy Number Variation (Gain)', 
              'a', 'Length of CNV (bp)', props, color='#0504aa', 
              report_max=True)
    plot_hist(axs[0,1], loss, 'Copy Number Variation (Loss)', 
              'b', 'Length of CNV (bp)', props, color='#aa0000',
              report_max=True)
    plot_hist(axs[1,0], total, 'Copy Number Variation', 
              'c', 'Length of CNV (bp)', props, color='#000000',
              report_max=True)
    plot_cum(axs[1,1], gain, loss, total, props, 'Length of CNV (bp)')
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    
def plot_cnv_gene(df, percentile, output_dir, output_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots a, b, c are the histograms of length of
        CNVs, measured in genes: a) gain CNVs in 
        blue, b) loss CNVs in red, and all CNVs 
        (gain and loss) in black. Sub-plot d shows the 
        cumulative distribution of gain, loss, total in one
        plot.
        
            Args:
                df: Dataframe that includes plot information.
                percentile: Only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
    """    
    col_names = ['type', 'size_genes_total']
    if not is_dataframe_valid(df, col_names):
        return 
    gain = df[df['type']=='gain']['size_genes_total']
    loss = df[df['type']=='loss']['size_genes_total']
    total = df['size_genes_total']
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(left=None, bottom=None, 
                        right=None, top=None, wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = get_hist_xlim(df['size_genes_total'], percentile)
    props = get_hist_props(df['size_genes_total'], xlim, percentile)
    plot_hist(axs[0,0], gain, 'Copy Number Variation (Gain)', 
              'a', 'Length of CNV (gene)', props, color='#0504aa',
              report_max=True)
    plot_hist(axs[0,1], loss, 'Copy Number Variation (Loss)',
              'b', 'Length of CNV (gene)', props, color='#aa0000',
              report_max=True)
    plot_hist(axs[1,0], total, 'Copy Number Variation',
              'c', 'Length of CNV (gene)', props, color='#000000',
              report_max=True)
    plot_cum(axs[1,1], gain, loss, total, props, 'Length of CNV (gene)')
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")  
    

def plot_stat_bp(df,  percentile, output_dir, output_name):
    """
        Plots a figure with 8 sub-plots (2 rows, 4 columns)
        The sub-plots show histograms for the statistics 
        (max, mean, median, min)of the length of gain and 
        loss CNV regions per cell, measured in base-pairs. 
        
            Args:
                df: Dataframe that includes plot information.
                percentile: Only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    col_names = ['malignant_key', 'max_gain_region', 'mean_gain_region',
                'median_gain_region', 'min_gain_region', 'max_loss_region',
                'mean_loss_region', 'median_loss_region', 'min_loss_region']
    if not is_dataframe_valid(df, col_names):
        return 
    max_gain = df[df['malignant_key']=='malignant']['max_gain_region']
    mean_gain = df[df['malignant_key']=='malignant']['mean_gain_region']
    median_gain = df[df['malignant_key']=='malignant']['median_gain_region']
    min_gain = df[df['malignant_key']=='malignant']['min_gain_region']
    max_loss = df[df['malignant_key']=='malignant']['max_loss_region']
    mean_loss =  df[df['malignant_key']=='malignant']['mean_loss_region']
    median_loss = df[df['malignant_key']=='malignant']['median_loss_region']
    min_loss = df[df['malignant_key']=='malignant']['min_loss_region']
    fig, axs = plt.subplots(2, 4)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = max(get_hist_xlim(max_gain, percentile), 
               get_hist_xlim(max_loss, percentile)) 
    props_list = [get_hist_props(max_gain, xlim, percentile),
                  get_hist_props(mean_gain, xlim, percentile),
                  get_hist_props(median_gain, xlim, percentile),
                  get_hist_props(min_gain, xlim, percentile),
                  get_hist_props(max_loss, xlim, percentile),
                  get_hist_props(mean_loss, xlim, percentile),
                  get_hist_props(median_loss, xlim, percentile),
                  get_hist_props(min_loss, xlim, percentile)]
    props = merge_props(props_list)
    plot_hist(axs[0, 0], min_gain,
              'Min. Len. of CNV Region \n per Cell (Gain)', 
              'a', 'Length of CNV (bp)', props, '#0504aa', plot_shape=(2, 4))
    plot_hist(axs[0, 1], mean_gain, 
              'Mean Len. of CNV Region \n per Cell (Gain)', 
              'b', 'Length of CNV (bp)', props, '#0504aa', plot_shape=(2, 4))
    plot_hist(axs[0, 2], median_gain,
              'Median Len. of CNV Region \n per Cell (Gain)', 
              'c', 'Length of CNV (bp)', props, '#0504aa', plot_shape=(2, 4))
    plot_hist(axs[0, 3], max_gain, 
              'Max. Len. of CNV Region \n per Cell (Gain)', 
              'd', 'Length of CNV (bp)', props, '#0504aa', plot_shape=(2, 4))
    plot_hist(axs[1, 0], min_loss,
              'Min. Len. of CNV Region \n per Cell (Loss)', 
              'e', 'Length of CNV (bp)', props, '#aa0000', plot_shape=(2, 4))
    plot_hist(axs[1, 1], mean_loss,
              'Mean Len. of CNV Region \n per Cell (Loss)', 
              'f', 'Length of CNV (bp)', props, '#aa0000', plot_shape=(2, 4))
    plot_hist(axs[1, 2], median_loss,
              'Median Len. of CNV Region \n per Cell (Loss)', 
              'g', 'Length of CNV (bp)', props, '#aa0000', plot_shape=(2, 4))
    plot_hist(axs[1, 3],  max_loss,
              'Max. Len. of CNV Region \n per Cell (Loss)', 
              'h', 'Length of CNV (bp)', props, '#aa0000', plot_shape=(2, 4))
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    

def plot_nregions_per_cell(df, percentile, output_dir, output_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots show the histograms of number of CNV regions
        per cell for gain and loss CNV types.
        
            Args:
                df: Dataframe that includes plot information.
                percentile: Only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    col_names = ['malignant_key', 'num_gain_region', 'num_loss_region']
    if not is_dataframe_valid(df, col_names):
        return     
    gain = df[df['malignant_key']=='malignant']['num_gain_region']
    loss = df[df['malignant_key']=='malignant']['num_loss_region']    
    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = max(get_hist_xlim(gain, percentile),
               get_hist_xlim(loss, percentile))
    props_list = [get_hist_props(gain, xlim, percentile), 
                  get_hist_props(loss, xlim, percentile)]
    props = merge_props(props_list)
    plot_hist(axs[0], gain, 'CNV Regions per Cells (Gain)', 'a',
              'Number of CNV Regions', props, color='#0504aa', 
              plot_shape=(1, 2))
    plot_hist(axs[1], loss, 'CNV Regions per Cells (Loss)', 'b', 
              'Number of CNV Regions', props, color='#aa0000', 
              plot_shape=(1, 2))
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")



def plot_cnv_per_cell(df, percentile, output_dir, output_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots show the histograms of length of gain and
        loss CNV regions as measured by base-pairs and genes.
        
            Args:
                df: Dataframe that includes plot information.
                percentile: Only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot         
    """
    col_names = ['malignant_key', 'total_gains_bp', 'total_losses_bp',
                'total_gains_genes', 'total_losses_genes']
    if not is_dataframe_valid(df, col_names):
        return         
    gain_bp = df[df['malignant_key']=='malignant']['total_gains_bp']
    loss_bp = df[df['malignant_key']=='malignant']['total_losses_bp']
    gain_gene = df[df['malignant_key']=='malignant']['total_gains_genes']
    loss_gene = df[df['malignant_key']=='malignant']['total_losses_genes']
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=None, wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim_bp = max(get_hist_xlim(gain_bp, percentile),
               get_hist_xlim(loss_bp, percentile))
    props_list_bp = [get_hist_props(gain_bp, xlim_bp, percentile), 
                     get_hist_props(loss_bp, xlim_bp, percentile)]
    props_bp = merge_props(props_list_bp)
    plot_hist(axs[0, 0], gain_bp, 'CNV per Cell (Gain)', 
              'a', 'Length of CNV (bp)', props_bp, color='#0504aa',
              plot_shape=(2, 2))    
    plot_hist(axs[0, 1], loss_bp, 'CNV per Cell (Loss)', 
              'b', 'Length of CNV (bp)', props_bp, color='#aa0000', 
              plot_shape=(2, 2))    
    xlim_gene = max(get_hist_xlim(gain_gene, percentile),
               get_hist_xlim(loss_gene, percentile))
    props_list_gene = [get_hist_props(gain_gene, xlim_gene, percentile), 
                       get_hist_props(loss_gene, xlim_gene, percentile)]
    props_gene = merge_props(props_list_gene)
    plot_hist(axs[1, 0], gain_gene, 'CNV per Cell (Gain)', 
              'c', 'Length of CNV (gene)', props_gene, color='#0504aa', 
              plot_shape=(2, 2))    
    plot_hist(axs[1, 1], loss_gene, 'CNV per Cell (Loss)', 
              'd', 'Length of CNV (gene)', props_gene, color='#aa0000', 
              plot_shape=(2, 2))    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    

def plot_expression_stats(df, percentile, output_dir, output_name):
    """
        Plots a figure with 4 sub-plots (2 rows, 2 columns)
        The sub-plots are histograms of a) Mean of logCPM,
        b) Variance of logCPM, c) coefficient of variation,
        d) fraction of detected cells.
        
            Args:
                df: Dataframe that includes plot information.
                percentile: Only bottom percentile of x values are taken 
                            into account
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    col_names = ['mean_logCPM', 'var_logCPM', 'cv', 'zeroes_cells']
    if not is_dataframe_valid(df, col_names):
        return     
    mean = df['mean_logCPM']
    variance = df['var_logCPM']
    variability = df['cv']
    fraction = df['zeroes_cells']    
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = get_hist_xlim(mean, percentile)
    props = get_hist_props(mean, xlim, percentile)
    plot_hist(axs[0, 0], mean, 'Mean of Gene Expression', 
              'a', 'Mean of logCPM', props, color='#004400', 
              plot_shape=(2, 2))    
    xlim = get_hist_xlim(variance, percentile)
    props = get_hist_props(variance, xlim, percentile)
    plot_hist(axs[0, 1], variance, 'Variance of Gene Expression', 
              'b', 'Variance of logCPM', props, color='#004400', 
              plot_shape=(2, 2))  
    xlim = get_hist_xlim(variability, percentile)
    props = get_hist_props(variability, xlim, percentile)
    plot_hist(axs[1, 0], variability, 
              'Expression Variability\nRelative to Its Mean', 
              'c', 'Coefficient of Variation', props, color='#004400', 
              plot_shape=(2, 2))  
    xlim = get_hist_xlim(fraction, percentile)
    props = get_hist_props(fraction, xlim, percentile)
    plot_hist(axs[1, 1], fraction, 'Gene Detection Frequency', 
              'd', 'Fraction of Detected Cells', props, color='#004400', 
              plot_shape=(2, 2))    
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")


def plot_gene2gene_corr(df, output_dir, output_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots are histograms of a) Gene-to-Gene 
        correlation for the top-400 most variable genes, and b) 
        Gene-to-Gene correlation for the top 400 
        highly-expressed genes.
        
            Args:
                df: Dataframe that includes plot information
                output_dir: directory to save plots in
                output_name: file name of plot
                
    """
    col_names = ['hv', 'he']
    if not is_dataframe_valid(df, col_names):
        return     
    most_var = df['hv']
    most_expr = df['he']
    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = 1
    props_list = [get_hist_props(most_var, xlim), 
                  get_hist_props(most_expr, xlim)]
    props = merge_props(props_list)
    plot_hist(axs[0], most_var, 
              'Gene-to-Gene Correlation\n Top 400 Most Variable Genes', 
              'a', 'Pearson Correlation', props, color='#004400', 
              plot_shape=(1, 2))
    plot_hist(axs[1], most_expr, 
              'Gene-to-Gene Correlation\n  Top 400 Highly-Expressed Genes', 
              'b', 'Pearson Correlation', props, color='#004400', 
              plot_shape=(1, 2))
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")



def plot_cell_counts(df, output_dir, output_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots are histograms of a) log-Library size
        , and b) fraction of detected genes.
        
            Args:
                df: Dataframe that includes plot information
                output_dir: directory to save plots in
                output_name: file name of plot
                
    """
    col_names = ['log1p_total_counts', 'zeroes_genes']
    if not is_dataframe_valid(df, col_names):
        return   
    log_counts = df['log1p_total_counts']
    detection_freq = df['zeroes_genes']
    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    xlim = get_hist_xlim(log_counts)
    props = get_hist_props(log_counts, xlim)
    plot_hist(axs[0], log_counts, 'log1p-Transformed\n Total Counts', 
              'a', 'log-Library Size', props, color='#4B0082', 
              plot_shape=(1, 2))
    props = get_hist_props(detection_freq, 1)
    plot_hist(axs[1], detection_freq,'Cell Detection Frequency' , 
              'b', 'Fraction of Detected Genes', props, color='#4B0082', 
              plot_shape=(1, 2))
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    
def plot_cell2cell_corr(df, output_dir, output_name):
    """
        Plots a figure with 2 sub-plots (1 row, 2 columns)
        The sub-plots are histograms of a) cell-to-cell correlation
        for the top-400 highly-variable genes, and b) cell-to-cell
        correlation for all genes.
        
            Args:
                df: Dataframe that includes plot information
                output_dir: directory to save plots in
                output_name: file name of plot
               
    """
    col_names = ['hv', 'all']
    if not is_dataframe_valid(df, col_names):
        return   
    most_var_genes = df['hv']
    all_genes = df['all']
    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.7, hspace=0.7)
    #set xlim and ylim according to the total histogram
    props = get_hist_props(most_var_genes, 1)
    plot_hist(axs[0], most_var_genes, 
              'Cell-to-Cell Correlation \n Top-400 Highly-Variable Genes', 
              'a', 'Pearson Correlation', props, color='#4B0082', 
              plot_shape=(1, 2))
    props = get_hist_props(all_genes, 1)
    plot_hist(axs[1], all_genes,'Cell-to-Cell Correlation \n All Genes' , 
              'b', 'Pearson Correlation', props, color='#4B0082', 
              plot_shape=(1, 2))
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")


def plot_chr_locations(df, output_dir, output_name):
    """
        Plots a heat-map marking relative locations of CNV regions
        on the 22 chromosomes. The relative location is divided into
        20 bins. Color scale is not normalized per chromosome. 
        
            Args:
                df: Dataframe that includes plot information
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    plt.figure(figsize=(6, 4))
    num_bins = 20
    bin_size = 5
    bins=[i*bin_size for i in range(num_bins)]
    bins_chr = np.zeros((22,num_bins), dtype=np.float32)
    bins_chr_n = np.zeros((22,num_bins), dtype=np.float32)
    for i in range(0, 22):
        x = df[df['chromosome']=='chr'+str(i+1)]['relative_location'].tolist()
        sum_ = 0
        for pos in x:
            bins_chr[i, int(pos/bin_size)] += 1
            bins_chr_n[i, int(pos/bin_size)] += 1
            sum_ += 1
        bins_chr_n[i, :] /= float(sum_)
    bins_ = bins_chr
    scale_x = 200
    scale_y = 300
    bins_scaled = np.zeros((scale_x*22, scale_y*num_bins), dtype=np.float32)
    for i in range(0, 22*scale_x):
        for j in range(0, num_bins*scale_y):
            bins_scaled[i, j] = bins_[int(i/scale_x), int(j/scale_y)]
    plt.gca().set_xticks([4*scale_y*val for val in range(6)],[4*scale_y*val for val in range(6)], fontsize=16)
    plt.gca().set_xlim([0, 20*scale_y])
    plt.gca().set_xticklabels([str(20*val) for val in range(6)], fontsize=16)
    plt.gca().set_yticks([scale_x*val+(scale_x/2) for val in range(22)],[scale_x*val+(scale_x/2) for val in range(22)], fontsize=4)
    plt.gca().set_ylim([0, 22*scale_x])
    plt.gca().set_yticklabels(['chr'+str(i+1) for i in range(22)], fontsize=8)
    plt.gca().set_xlabel('Relative Location on Chromosome (%)', fontsize=16)
    plt.gca().set_title('Location of CNV Regions\n', fontsize=18)
    cbar_lim = get_hist_xlim(bins_.flatten())
    props = get_hist_props(bins_.flatten(), cbar_lim)
    cbar_vals = [i*props['xtick_size'] for i in range(int(cbar_lim/props['xtick_size']+1))]
    norm = colors.Normalize(vmin=0, vmax=cbar_lim)
    im = plt.imshow(bins_scaled, interpolation='none')
    im.set_norm(norm)
    cbar = plt.colorbar(im, fraction=0.02, ticks=cbar_vals)
    cbar.ax.set_yticklabels([str(val) for val in cbar_vals], fontsize=16)  # horizontal colorbar
    full_path = os.path.join(output_dir, output_name+'.pdf')
    plt.savefig(full_path, bbox_inches="tight")
    
    
def plot_chr_location_and_size_all(df, output_dir, output_name):
    """
        Plots a relational plot marking relative locations of CNV regions
        on the 22 chromosomes and their respecitve sizes.
        
            Args:
                df: Dataframe that includes plot information
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    
    if not np.issubdtype(df['chromosome'].dtype, np.integer):
        utils.sort_by_chromosome(df, 'chromosome')
    
    sns.set_theme()
    p = sns.relplot(data=df, x="chromosome", y="relative_location", hue = 'size_bp_total', height = 6, aspect = 1.5)
    p.set(xticks=np.arange(1,23,1), xlabel='Chromosome',
          ylabel='Relative Location on Chromosome (%)')
    plt.title('Location of CNV regions per chromosome', fontsize=20)
    full_path = os.path.join(output_dir, output_name+'.pdf')
    p.savefig(full_path) 
    
    
def plot_chr_location_and_size_gl(df, output_dir, output_name):
    """
        Plots a relational plot marking relative locations of CNV regions
        on the 22 chromosomes and their respecitve sizes, separated into gains and losses.
        
            Args:
                df: Dataframe that includes plot information
                output_dir: directory to save plots in
                output_name: file name of plot
    """
    
    if not np.issubdtype(df['chromosome'].dtype, np.integer):
        utils.sort_by_chromosome(df, 'chromosome')
    
    sns.set_theme()
    p = sns.relplot(data=df, x="chromosome", y="relative_location", hue = 'size_bp_total', height = 6, aspect = 1.5, col = 'type')
    p.set(xticks=np.arange(1,23,1), xlabel='Chromosome',
          ylabel='Relative Location on Chromosome (%)')
    plt.suptitle('Location of CNV regions per chromosome', fontsize=20)
    p.fig.subplots_adjust(top=0.88)
    full_path = os.path.join(output_dir, output_name+'.pdf')
    p.savefig(full_path) 
    
    
   

