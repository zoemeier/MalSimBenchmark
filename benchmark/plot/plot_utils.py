import chardet
import datetime as dt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect_left


def read_data(file_path, skiprows=None):
    with open(file_path, 'rb') as f:
        enc = chardet.detect(f.read())
    data = pd.read_csv(
        file_path,
        encoding = enc['encoding'],
        sep=',',
        skiprows=skiprows, 
    )
    return data
    
    
def print_seperator():
    """Prints a seperator line between sections in standard output"""
    print('')
    print('------------------------------------------------')
    print('')
    
    
def analyze_data(df: pd.DataFrame):
    """
        Prints general informations about dataframe in standard output,
        including: number of rows, shape, column names, first few rows
       
            Args:
                df: dataframe to be described.
    """
    #general info
    print(f'Number of rows: {len(df)}')
    print_seperator()
    print(f'Dataframe shape: {df.shape}') 
    print_seperator()
    print('Column names:')
    for col in df.columns:
        print(col)
    print_seperator()
    print('Describing data:')
    print(df.describe())
    print_seperator()
    print('First few rows:')
    print(df.head())
    

def is_dataframe_valid(df, col_names):
    """
        Checks if a given dataframe has all the given column names.
        
            Args:
                df: input dataframe to be checked
                col_names: a list of column names
            
            Returns: 
                True if df has all column names specified in col_names, 
                otherwise False.
    """    
    for col_name in col_names:
        if not(col_name in df):
            print('Dataframe does not have ', col_name)
            return False
    return True
    

def scientific_form(value): 
    """
        Converts a value to the form (a x 10^exponent), where (1 < a <= 10)
        
            Args:
                value: value to be converted.
            
            Returns: 
                Tuple [a, exponent]
    """
    exponent = 0
    a = float(value)
    while a < 1:
        a *= 10
        exponent -= 1
    while a > 10:
        a /= 10
        exponent += 1
    return [a, exponent]



def fancy_format(value):
    """
        If a value has 0 floating point, it converts it to an integer,
        otherwise it returns the same value. For example the function
        returns 3 for input value 3.0, and returns 3.5 for input 3.5.
        
            Args:
                value: value to be converted.
            
            Returns: 
                Converted value of the input.
    """    
    if value - int(value) == 0.0:
        return int(value)
    else:
        return value

    
    
def is_integer(value):
    """
        Checks if the input value is of type int
            Args:
                value: value that will be checked
            
            Returns: 
                True if the value is integer otherwise False
    """   
    return isinstance(value, int)



def get_bottom_percentile(x, percentile):
    """
        For an input list of values, it select the bottom percentile. 
        For example from 1000 values if we select 90 percent the lowest 
        900 values will be selected. In the case that the highest value 
        in the percentile is not unique, we make sure that we select 
        all of its instances, resulting in a bigger selection than the 
        specified percentile.
        
            Args:
                x: list of input values. The list doesn't have to be 
                   sorted.
                percentile: the percentile (float) to be selected from 
                   0 to 100 
            
            Returns: 
                The selected list of values.
    """      
    sorted_x = np.asarray(sorted(x))
    num_vals = sorted_x.shape[0]
    num_selected_vals = int((num_vals*percentile)/100)
    #include equal values to the last selected values
    pivot = sorted_x[num_selected_vals-1]
    index = num_selected_vals-1 
    while (index < num_vals) and (sorted_x[index] == pivot):
        index += 1
    selected_x = sorted_x[:index]
    return selected_x



def get_hist_xlim(x, percentile=100):
    """
        It calculated the limit of x-axis for a vertical histogram. Note that 
        x-axis limit is not necessarily the maximum value, but the closest nice
        number to the maximum. For example if the maximum value is 31, 40 would
        be chosen as the limit. If the maximum value is 72, 75 is chosen.
        
            Args:
                x: list of values in the histogram
                percentile: only bottom percentile of x values are taken 
                            into account
            
            Returns: 
                The limit of x-axis for a vertical histogram based on input 
                values and the specified percentile.
    """      
    selected_x = get_bottom_percentile(x, percentile)
    max_x = np.max(selected_x)
    a_x, exp_x = scientific_form(max_x)
    #calculate limit of x axis (xlim)
    #choose x and y limits among nice numbers 
    candidate_lims =       [1.2, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 
                            7.5, 8.0, 9.0, 10.0]
    candidate_tick_sizes = [0.4, 0.5, 0.4, 0.5, 1.0, 1.0, 1.0, 2.0, 
                            2.5, 2.0, 3.0, 2.5]
    candidate_bin_sizes =  [x/20 for x in candidate_lims]
    xlim_index = bisect_left(candidate_lims, a_x)
    a_xlim = candidate_lims[xlim_index] 
    xlim = a_xlim * (10**exp_x)
    return xlim
    

    
def get_hist_props(x, xlim, percentile=100, normalized=False):
    """
        Calculates properties of a histogram plot based on x values. It 
        makes sure to choose nice values for ranges and properties. 
        The properties include: x and y axes limits, x and y axes tick 
        sizes, and bin size.
        
        
            Args:
                x: list of values in the histogram
                xlim: the limit of x-axis.
                percentile: only bottom percentile of x values are taken
                            into account
            
            Returns: 
                A dictionary with the following properties:
                    'xlim': limit of x-axis
                    'ylim': limit of y-axis
                    'xtick_size': tick size of x-axis
                    'ytick_size': tick size of y-axis
                    'bin_size': histogram bin size
        
    """ 
    selected_x = get_bottom_percentile(x, percentile)
    max_x = np.max(selected_x)
    a_x, exp_x = scientific_form(xlim)
    #calculate limit of x axis (xlim)
    #choose x and y limits among nice numbers 
    candidate_lims =       [1.2, 1.5, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 
                            7.5, 8.0, 9.0, 10.0]
    candidate_tick_sizes = [0.4, 0.5, 0.4, 0.5, 1.0, 1.0, 1.0, 2.0, 
                            2.5, 2.0, 3.0, 2.5]
    candidate_bin_sizes =  [x/20 for x in candidate_lims]
    xlim_index = bisect_left(candidate_lims, a_x)
    a_xlim = candidate_lims[xlim_index] 
    xlim = a_xlim * (10**exp_x)
    xtick_size = candidate_tick_sizes[xlim_index] * (10**exp_x)
    xtick_size = fancy_format(xtick_size)
    #calculate bin sizes and histogram
    if (np.max(x)<=1) and (np.min(x)>=0):
        bin_size = 0.05
        bin_vals = [0 for i in range(20)]
        for val in selected_x:
            index = int(val/bin_size)
            bin_vals[min(index, 19)] += 1    
    elif (np.max(x)<=1) and (np.min(x)>=-1):
        bin_size = 0.1
        bin_vals = [0 for i in range(20)]
        for val in selected_x:
            index = int((val+1)/bin_size)
            bin_vals[min(index, 19)] += 1    
    else:
        bin_size = candidate_bin_sizes[xlim_index] * (10**exp_x)
        num_bins = int(xlim/bin_size)
        bins = [(i*bin_size) for i in range(num_bins)]
        bin_vals = [0 for i in range(num_bins)]
        for val in selected_x:
            index = int(val/bin_size)
            bin_vals[min(index, 19)] += 1
    
    if normalized:
        sum_bins = np.sum(np.asarray(bin_vals)) * bin_size
        bin_vals = np.asarray(bin_vals) / sum_bins
    
    max_height = np.max(np.asarray(bin_vals))    
    #calculate ylim and y_tick_sizes
    a_y, exp_y = scientific_form(max_height)    
    ylim_index = bisect_left(candidate_lims, a_y)
    a_ylim = candidate_lims[ylim_index] 
    ylim = a_ylim * (10**exp_y)
    ytick_size = fancy_format(candidate_tick_sizes[ylim_index] * (10**exp_y))
    #set y ticks and autoscale it in case it needs an exponent
    props = dict()
    props['xlim'] = xlim
    props['ylim'] = ylim
    props['xtick_size'] = xtick_size
    props['ytick_size'] = ytick_size
    props['bin_size'] = bin_size
    return props



def merge_props(props_list):
    """
        Merges the properties of several histogram plots. When a
        figure has several parts, this function is used to make sure 
        that the x and y of all plots have the same limits.
        
        
            Args:
                props_list: list of histogram properties(see get_hist_props)
            
            Returns: 
                A dictionary with the following properties:
                    'xlim': limit of x-axis
                    'ylim': limit of y-axis
                    'xtick_size': tick size of x-axis
                    'ytick_size': tick size of y-axis
                    'bin_size': histogram bin size    
    """     
    xlim = ylim = xtick_size = ytick_size = bin_size = -1
    for props in props_list:
        xlim_ = props['xlim']
        xtick_size_ = props['xtick_size']
        bin_size_ = props['bin_size']
        if xlim_ > xlim:
            xlim = xlim_
            xtick_size = xtick_size_
            bin_size = bin_size_
        ylim_ = props['ylim']
        ytick_size_ = props['ytick_size']
        if ylim_ > ylim:
            ylim = ylim_
            ytick_size = ytick_size_
    result_props = dict()
    result_props['xlim'] = xlim
    result_props['ylim'] = ylim
    result_props['xtick_size'] = xtick_size
    result_props['ytick_size'] = ytick_size
    result_props['bin_size'] = bin_size
    return result_props



def plot_hist(ax, x, title, subpart, xlabel, 
              props, color='#000000', alpha=1.0, density=False,
              plot_shape=(2,2), report_max=False):
    """
        Plots a histogram with well-defined limits, bin size, and plot size.
        
            Args:
                ax: matplotlib.pyplot axes.
                x: list of values of the histogram
                title: title printed above the plot
                subpart: title of plot subpart, e.g. 'a'
                xlabel: x-axis label
                props: histogram properties (see get_hist_props)
                color: color of histogram bars
                alpha: transparency of the histogram bars
                density: If True, normalizes the histogram
                plot_shape: the general plot shape (rows, cols), 
                            that includes the current histogram
                            as a part of it.
                report_max: If True, maximum value of variable will be 
                            written under the plot. 
  
    """ 
    if plot_shape == (1, 2):
        ax.figure.set_size_inches(17, 5)
    if plot_shape == (2, 2):
        ax.figure.set_size_inches(17, 12)
    if plot_shape == (2, 4):
        ax.figure.set_size_inches(23, 10)
    xlim = props['xlim']
    ylim = props['ylim']
    xtick_size = props['xtick_size']
    ytick_size = props['ytick_size']
    bin_size = props['bin_size']    
    bins = [(i*bin_size) for i in range(int(xlim/bin_size))]
    if (np.max(x) <= 1.0) and (np.min(x) >= 0):
            bin_size = 0.05
            bins = [(i*bin_size) for i in range(int(xlim/bin_size))]
    elif (np.max(x) <= 1.0) and (np.min(x) >= -1):
            bin_size = 0.1
            bins = [(i*bin_size)-1 for i in range(20)]
    
    if density:
        n, bins, patches = ax.hist(x, bins=bins, color=color, 
                                   alpha=alpha, density=True, rwidth=0.85, histtype='stepfilled')        
    else:
        n, bins, patches = ax.hist(x, bins=bins, color=color, alpha=alpha, rwidth=0.85)
        
    #calculate bin heights
    bin_vals = [0 for i in range(int(bin_size))]   
    
    if density:
        #ax.set_ylabel('Density', fontsize=18)
        yvalues = [i*ytick_size for i in range( int(float("{:.1f}".format(ylim/ytick_size))) +1)]
        a_ytick, exp_ytick = scientific_form(ytick_size)        

        if exp_ytick < -1:
            ax.set_ylabel('Density ' + 
                              r'$\times 10^{0}$'.format(exp_ytick), fontsize=18)
            ax.set_yticks(yvalues, 
                              ["{:.1f}".format((val/(10**(exp_ytick)))) for val in yvalues], 
                              fontsize=16)
        else:
            ax.set_ylabel('Density', fontsize=18)
            ax.set_yticks(yvalues, 
                              ["{:.1f}".format(val) for val in yvalues], 
                              fontsize=16)

    else:
        #set y ticks and autoscale it in case it needs an exponent
        yvalues = [i*ytick_size for i in range(int(ylim/ytick_size)+1)]
        a_ytick, exp_ytick = scientific_form(ytick_size)        
        if int(a_ytick) == 10:
            exp_ytick += 1
        if exp_ytick >= 4:
            ax.set_ylabel('Frequency ' + 
                          r'$\times 10^{0}$'.format((exp_ytick-1)), fontsize=18)
            ax.set_yticks(yvalues, 
                          ['%d' % int(val/(10**(exp_ytick-1))) for val in yvalues], 
                          fontsize=16)
        else:
            ax.set_ylabel('Frequency', fontsize=18)
            ax.set_yticks(yvalues, ['%d' % int(val) for val in yvalues], 
                          fontsize=16)
    
        
    #set y ticks and autoscale it in case it needs an exponent
    xvalues = [i*xtick_size for i in range(int(xlim/xtick_size)+1)]
    a_xtick, exp_xtick = scientific_form(xtick_size)        
    if int(a_xtick) % 10 == 0:
        exp_xtick += 1
    if exp_xtick >= 4:
        ax.set_xlabel(xlabel +
                      r' $\times 10^{0}$'.format((exp_xtick-exp_xtick%3)), 
                      fontsize=18)
        ax.set_xticks(xvalues, 
                      [fancy_format(val/(10**(exp_xtick-exp_xtick%3))) 
                       for val in xvalues], fontsize=16)
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        if is_integer(xtick_size):
            ax.set_xticks(xvalues, [fancy_format(val) for val in xvalues], 
                          fontsize=16)
        else:
            ax.set_xticks(xvalues, [float(val) for val in xvalues], 
                          fontsize=16)            
    #set axes limits
    margin = 0.02
    ax.set_xlim([-margin*xlim, (xlim+margin*xlim)])
    
    ax.set_ylim([0, 1.02*ylim])
    
    ax.tick_params('both', length=10, width=2, which='major')        
    if (np.max(x) <= 1.0) and (np.min(x) >= 0):
        ax.set_xlabel(xlabel, fontsize=18)
        xvalues = [i * 0.2 for i in range(6)]
        ax.set_xticks(xvalues, ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], 
                      fontsize=16)
        ax.set_xlim([-0.02, 1.02])
    elif (np.max(x) <= 1.0) and (np.min(x) >= -1.0):
        ax.set_xlabel(xlabel, fontsize=18)
        xvalues = [(i-2) * 0.5 for i in range(5)]
        ax.set_xticks(xvalues, ['-1.0', '-0.5', '0.0', '0.5', '1.0'], 
                      fontsize=16)        
        ax.set_xlim([-1.02, 1.02])

    
    if plot_shape != (2, 4):
        ax.text(-0.38, 0.9, subpart, fontweight='bold', fontsize=60, transform=ax.transAxes)
    else:
        ax.text(-0.45, 0.9, subpart, fontweight='bold', fontsize=60, transform=ax.transAxes)
        
    
    if not density:
        if report_max:
            ax.text(0.3*xlim, -0.3*ylim, 
                    'Maximum value = ' + str(fancy_format(max(x))), fontsize=10)
        
    ax.set_title(title, fontsize=22)

    
def plot_cum_comp(ax, total1, total2, props, xlabel):
    """
        Compares two cumulative distributions from two datasets
        on top of each other. 
        
            Args:
                ax: matplotlib.pyplot axes.
                tota1l: list of values for first dataset
                total2: list of values for second dataset
                props: histogram properties (see get_hist_props)
                xlabel: x-axis label
    """ 
    xlim = props['xlim']
    xtick_size = props['xtick_size']
    bin_size = props['bin_size']
    bins = [(i*bin_size) for i in range(int(xlim/bin_size))]
    total1_ = total1.tolist()
    total2_ = total2.tolist()
    total1_.append(max(total1_+ total2_)+10*xlim)
    total2_.append(max(total1_+ total2_)+10*xlim)
    ax.hist(total1_, 2000, density=True, histtype='step',
                           cumulative=True, color='#0000bb', alpha=0.6)
    ax.hist(total2_, 2000, density=True, histtype='step',
                           cumulative=True, color='#aa0000', alpha=0.6)
    xvalues = [i*xtick_size for i in range(int(xlim/xtick_size)+1)]
    a_xtick, exp_xtick = scientific_form(xtick_size)        
    if int(a_xtick) == 10:
        exp_xtick += 1
    if exp_xtick >= 4:
        ax.set_xlabel(xlabel +
                      r' $\times 10^{0}$'.format((exp_xtick-1)), 
                      fontsize=18)
        ax.set_xticks(xvalues, 
                      ['%d' % fancy_format(val/(10**(exp_xtick-1))) 
                       for val in xvalues], fontsize=16)
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_xticks(xvalues, ['%d' % int(val) for val in xvalues], 
                      fontsize=16)    
    
    yvalues = [0.2 * i for i in range(6)]
    ax.set_yticks(yvalues, ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], 
                  fontsize=16)
    ax.set_ylabel(r'$Pr\{L \leq l\}$', fontsize=20)
    
    ax.tick_params('both', length=10, width=2, which='major')
    margin = 0.02
    ax.set_xlim([-margin*xlim, (1+margin)*xlim])
    ax.set_ylim([-.1, 1.1])
        
    ax.text(-0.38, 0.9, 'd', fontweight='bold', fontsize=60, transform=ax.transAxes)
    
    ax.set_title('Cumulative Distribution', fontsize=22)
    
    

def plot_cum(ax, gain, loss, total, props, xlabel):
    """
        Plots 3 cumulative distribution (gain, loss, total)
        on top of each other. 
        
            Args:
                ax: matplotlib.pyplot axes.
                gain: list of values (blue color)
                loss: list of values (red color)
                total: list of values (black color)
                props: histogram properties (see get_hist_props)
                xlabel: x-axis label
    """ 
    xlim = props['xlim']
    xtick_size = props['xtick_size']
    bin_size = props['bin_size']
    bins = [(i*bin_size) for i in range(int(xlim/bin_size))]
    total_ = total.tolist()
    gain_ = gain.tolist()
    loss_ = loss.tolist()
    total_.append(max(total)+bin_size+10*xlim)
    gain_.append(max(total)+bin_size+10*xlim)
    loss_.append(max(total)+bin_size+10*xlim)    
    ax.hist(total_, 1000, density=True, histtype='step',
                           cumulative=True, color='#000000')
    ax.hist(gain_, 1000, density=True, histtype='step',
                           cumulative=True, color='#0504aa')
    ax.hist(loss_, 1000, density=True, histtype='step',
                           cumulative=True, color='#bb0000')
    xvalues = [i*xtick_size for i in range(int(xlim/xtick_size)+1)]
    a_xtick, exp_xtick = scientific_form(xtick_size)        
    if int(a_xtick) == 10:
        exp_xtick += 1
    if exp_xtick >= 4:
        ax.set_xlabel(xlabel +
                      r' $\times 10^{0}$'.format((exp_xtick-1)), 
                      fontsize=18)
        ax.set_xticks(xvalues, 
                      ['%d' % fancy_format(val/(10**(exp_xtick-1))) 
                       for val in xvalues], fontsize=16)
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_xticks(xvalues, ['%d' % int(val) for val in xvalues], 
                      fontsize=16)    
    
    yvalues = [0.2 * i for i in range(6)]
    ax.set_yticks(yvalues, ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], 
                  fontsize=16)
    ax.set_ylabel(r'$Pr\{L \leq l\}$', fontsize=20)
    
    ax.tick_params('both', length=10, width=2, which='major')
    margin = 0.02
    ax.set_xlim([-margin*xlim, (1+margin)*xlim])
    ax.set_ylim([-.1, 1.1])
        
    ax.text(-0.38, 0.9, 'd', fontweight='bold', fontsize=60, transform=ax.transAxes)

    ax.set_title('Cumulative Distribution', fontsize=22)

    
def plot_violin_comp(ax, x1, x2, title, label, props, color1, color2):
    """
        Plots a histogram with well-defined limits, bin size, and plot size.
        
            Args:
                ax: matplotlib.pyplot axes.
                x1: list of values of first distribution
                x2: list of values of second distribution
                title: title printed above the plot
                label: label of y-axis
                props: histogram properties (see get_hist_props)
                color1: color of first dataset x1
                color2: color of second dataset x2  
    """ 
    ax.figure.set_size_inches(20, 20)
    
    xlim = props['xlim']  
    xtick_size = props['xtick_size']
    
    parts = ax.violinplot([x1, x2], positions = [1, 2], 
                          showmeans=False, showmedians=False, showextrema=False)    
    
    for i, xs in enumerate([x1, x2]):
        x1s = [sorted(xs)]
        qbottom, quartile1, medians, quartile3, qtop = np.percentile(x1s, [10, 25, 50, 75, 90], axis=1)
        inds = [(i+1)]
        whiskers_min, whiskers_max = qbottom, qtop
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=14)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=3)
        
    colors = [color1, color2]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('#000000')
        pc.set_linewidths(2)
        pc.set_alpha(0.3)
    ax.set_xticks([1, 2], ['', ''])
    ax.set_xlim([0.5, 2.5])
    ax.set_ylabel(label, fontsize=16)
    #set axes limits
    xvalues = [i*xtick_size for i in range(int(xlim/xtick_size)+1)]
    a_xtick, exp_xtick = scientific_form(xtick_size)        
    if int(a_xtick) % 10 == 0:
        exp_xtick += 1
    if exp_xtick >= 4:
        ax.set_ylabel(label +
                      r' $\times 10^{0}$'.format((exp_xtick-exp_xtick%3)), 
                      fontsize=16)
        ax.set_yticks(xvalues, 
                      [fancy_format(val/(10**(exp_xtick-exp_xtick%3))) 
                       for val in xvalues], fontsize=16)
    else:
        ax.set_ylabel(label, fontsize=18)
        if is_integer(xtick_size):
            ax.set_yticks(xvalues, [fancy_format(val) for val in xvalues], 
                          fontsize=16)
        else:
            ax.set_yticks(xvalues, [float(val) for val in xvalues], 
                          fontsize=16)            
    ax.set_ylim([0, 1.02*xlim])
    
    if  ((np.max(x1) <= 1.0) and (np.min(x1) >= 0) 
        and (np.max(x2) <= 1.0) and (np.min(x2) >= 0)):
        ax.set_ylabel(label, fontsize=16)
        xvalues = [i * 0.2 for i in range(6)]
        ax.set_yticks(xvalues, ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], 
                      fontsize=16)
        ax.set_ylim([-0.02, 1.02])
    elif ((np.max(x1) <= 1.0) and (np.min(x1) >= -1.0) 
         and (np.max(x2) <= 1.0) and (np.min(x2) >= -1.0)):
        ax.set_ylabel(label, fontsize=18)
        xvalues = [(i-2) * 0.5 for i in range(5)]
        ax.set_yticks(xvalues, ['-1.0', '-0.5', '0.0', '0.5', '1.0'], 
                      fontsize=16)        
        ax.set_ylim([-1.02, 1.02])
        
    ax.set_title(title, fontsize=18)
    
    
    


    
