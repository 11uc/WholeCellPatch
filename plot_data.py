#!/usr/local/python3

import os
import numpy as np
import pandas as pd
import scipy.stats as stat
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from cycler import cycler
import util
import process
import plot
import ap
import ahp

def bar_graph(m, ylabel = None, title = None, se = None, points = None, link = False, \
    xticklabels = [], legendlabels = [], colors = [], stars = [], bar_width = 0.8, \
    fname = 'tmp.png'):
    '''
    bar_graph(m, ylabel = None, title = None, se = None, points = None, link = False, \
        xticklabels = [], legendlabels = [], colors = [], stars = [], bar_width = 0.8, \
        fname = 'tmp.png'):
        Plot a bar graph.
    parameters:
        m (array_like) - mean values
        ylabel (string) - Y-axis label, may be omitted
        title (string) - graph title, may be omitted
        se (array_like) - standard error for error bar, same length as m, may be omitted
        points (array_like) - elements are arrays of sample values for each group, may
            be omitted
        link (boolean) - whether to link points with the same indices
        xticklabels (array_like) - elements are strings, xtick labels, may be omitted
        legendlabels (array_like) - elements are strings, legend labels, may be omitted
        colors (array_like) - elements are matplotlib accepted color variables, 
            colors of barr. If it's empty, use grey scale or black color.
        stars (array_like) - significance stars, elements are 3-elemtes arrays, first 2 
            are indices of the two sample groups and the 3rd is the number of stars.
        bar_width (float) - relative bar witdh
        fname (string) - output file direcotry
    '''


    left = np.arange(len(m)) + 1 - bar_width / 2
    mid = np.arange(len(m)) + 1

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)

    bars = ax.bar(left, m, width = bar_width, ec = 'k', fc = 'w', align = 'edge')
    if se is not None:
        ax.errorbar(mid, m, yerr = se, fmt = 'none', ecolor = [0, 0, 0, 1], elinewidth = 3)
    if points is not None:
        if link:
            for i in range(points.shape[1]):
                ax.plot(mid, points[:, i], color = [0.6, 0.6, 0.6])
        else:
            for i in range(len(points)):
                ax.scatter([mid[i]] * len(points[i]), points[i], marker = 'o', \
                    edgecolors = [0.6, 0.6, 0.6], facecolors = 'none', zorder = 2, \
                    linewidth = 3)

    sign = 1  # sign of the significant star position
    if all([d > 0 for d in m]):
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylim(bottom = 0)
    elif all([d < 0 for d in m]):
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.set_ylim(top = 0)
        sign = -1
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.plot([mid[0] - 0.5, mid[-1] + 0.5], [0, 0], color = [0, 0, 0])
    ax.xaxis.set_tick_params(width=0)

    # plot xtick labels
    if len(xticklabels):
        ax.xaxis.set_ticks(mid)
        ax.xaxis.set_ticklabels(xticklabels)
        if len(colors):
            for i, cs in zip(ax.xaxis.get_ticklabels(), colors):
                i.set(size = 26, color = cs)
        else:
            for i in ax.xaxis.get_ticklabels():
                i.set(size = 26)
    else:
        ax.xaxis.set_ticklabels([''] * len(m))
    # plot legend in the figure
    if len(legendlabels):
        if not len(colors):
            colors = [[i / len(m)] * 3 for i in range(len(m))]  # face colors
        for i in range(len(m)):
            bars[i].set_fc(colors[i])
            bars[i].set_ec(colors[i])
        mpl.rcParams['font.size'] = 26
        patches = [mpatches.Patch(color = colors[i], label = legendlabels[i]) \
            for i in range(len(m))]
        f.legend(handles = patches , bbox_to_anchor = (1.5, 0.8), \
            loc = 2, borderaxespad = 0, frameon = False)
    for i in ax.yaxis.get_ticklabels():
        i.set(size = 26)

    if len(stars) and (se != None or points is not None):
        if se != None:
            star_level = sign * max([sign * m[i] + se[i] for i in range(len(m))]) * 1.05
        else:
            star_level = sign * max([max([(sign * p) for p in ps]) for ps in points]) * 1.05
        sw = 0.1 # asteroid width
        step = star_level / 25
        for star in stars:
            pair = star[0:2]
            sig = star[-1]
            ax.plot(mid[pair], [star_level] * 2, color = [0, 0, 0])
            sx = (np.arange(sig) - (sig - 1) / 2) * sw + np.mean(mid[pair]) # asteroid x value
            print(sx)
            if sig:
                ax.plot(sx, [star_level + step] * sig, \
                    ls = None, marker = '*', markerfacecolor = 'k', \
                    markeredgecolor = 'k')
            star_level  = star_level + 2 * step
        if sign > 0:
            ax.set_ylim(top = star_level)
        else:
            ax.set_ylim(bottom = star_level)

    ax.spines['right'].set_visible(False)
    ax.locator_params(nbins = 5, axis = 'y')
    ax.yaxis.set_ticks_position('left')
    if ylabel is not None:
        ax.yaxis.set_label_text(ylabel, fontsize = 26)
    if title is not None:
        ax.set_title(title, fontsize = 26)
    ax.set_xlim([mid[0] - bar_width, mid[-1] + bar_width]) 

    mpl.rcParams.update({'font.size': 26})  # set all font size to 26
    f.set_size_inches(1 + bar_width * len(m), 7)
    #f.set_size_inches(9.5, 4.5)
    f.savefig(fname, dpi = 96, bbox_inches = 'tight', transparent = True)

    plt.close(f)
    del(f)
    return 0

def corr_graph(data_file, group, x_name, y_name, xlabel, ylabel, cl = [], fname = 'tmp.png'):

    data = pd.read_csv(data_file)
    x = data[x_name]
    y = data[y_name]
    p = np.poly1d(np.polyfit(x, y, 1))
    xx = np.linspace(min(x) - 0.05 * (max(x) - min(x)), max(x) + 0.05 * (max(x) - min(x)), \
        len(x) * 5)
    yy = p(xx)

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    types = np.unique(data[group])
    if not len(cl):
        ncolors = len(np.unique(types))
        cm = plt.get_cmap('gist_rainbow')
        cl = [cm(1 * i / ncolors) for i in range(ncolors)]
    plots = []
    for t, c in zip(types, cl):
        t_ind = np.nonzero(data[group] == t)[0]
        plots.append(ax.scatter(x[t_ind], y[t_ind], c = 'none', edgecolors = c, label = t))
    plot = ax.plot(xx, yy, 'k')
    # plt.legend(handles = plots)
    ax.spines['right'].set_visible(False)
    ax.locator_params(nbins = 5, axis = 'y')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_text(ylabel, fontsize = 26)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_text(xlabel, fontsize = 26)

    mpl.rcParams.update({'font.size': 26})  # set all font size to 26
    #f.set_size_inches(1 + bar_width * len(m), 7)
    #f.set_size_inches(9.5, 4.5)
    f.savefig(fname, dpi = 96, bbox_inches = 'tight', transparent = True)

    plt.close(f)
    del(f)
    return 0

def sample_trace(folder, cell_num, trial_num, win, shift = [0], scale = [1, 1], \
    units = None, scalebar = [0, 0], stim = [], lowpass = 0, medfilt = 0, ave = False, \
    interp = 0, ylim = [-50e-12, 50e-12], lw = 1, cl = 'k', fname = 'tmp.png'):
    '''
    sample_trace(folder, cell_num, trial_num,  win, shift = [0], scale = [1, 1], \
        units = None, scalebar = [0, 0], stim = [], lowpass = 0, medfilt = 0, ave = False, \
        ylim = [-50e-12, 50e-12])
        plot a scaled fragment of ephys recording trace as sample
    inputs:
        file_dir (string) - directory to igor binary data file.
        cell_num (list) - list of indices of cells whose traces will be plotted.
        trial_num (list) - list of trial numbers to be plotted, a pair with the same indices
            in cell_num and trial_num will define one trace.
        win (list) - 2 scalar element [w1, w2], fragment time window without scaling. 
        shift (list) - shift on the value axis of each trace, default is not shift.
        scale (list) -  2 scalar element [sx, sy], scaling factor for time and value axis
            sx and sy respectively.
        units (list) - 2 string elements list [ux, uy], units after scaling.
        scalebar (list) - 2 scalar elements list [sx, sy] - scale bar lengh of the two axis.
        stim (list) - time points of stimulation
        lowpass (float) - band with of lowpass filter applied to the trace, default: no 
            filter applied
        medfilt (float) - median filter threshold, default: no median filter
        ave (boolean) - whether to plot the average trace of all the traces
        ylim (list) - 2 scalar element [y1, y2], limits of the y axis
        interp (float) - time of number of interpolated points to plot over current number of
            points
    returns:
        f - matplotlib figure
    TODO:
        1. more convenient parameters
    '''

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.axis('off')

    if len(shift) == 1 and shift[0] == 0 and len(cell_num) > 1:
        shift = [0] * len(cell_num)
    low = 0
    if folder[-1] != os.sep:
        folder += os.sep
    for i in range(len(cell_num)):
        file_dir = folder + 'Cell_{0:04d}_{1:04d}.ibw'.format(cell_num[i], trial_num[i])
        print(file_dir)
        trace, sr, st = util.load_wave(file_dir)
        trace = np.array(trace).copy()
        if trace is 0:
            return 0
        if medfilt:
            trace = process.thmedfilt(trace, 7, 20e-12)
        if lowpass:
            trace = process.smooth(trace, sr, lowpass)
        if shift[i]:
            trace += shift[i]
        else:
            trace += 0 - np.mean(trace[int(0.5 * sr):int(win[0] * sr)])  # move bl to 0
        t = np.arange(len(trace)) / sr
        win_ind1 = int(win[0] * sr)
        win_ind2 = int(win[1] * sr)
        xwin = t[win_ind1:win_ind2]
        ywin = trace[win_ind1:win_ind2]
        trace_color = mpl.colors.colorConverter.to_rgb(cl)
        if ave:
            if i == 0:
                ave_trace = np.zeros(win_ind2 - win_ind1)
            ave_trace = ave_trace + ywin
            ave_color = trace_color
            trace_color = [(1 + d) / 2 for d in color]
        # interpolate the data to make it look smoother
        if interp:
            interf = interpolate.interp1d(xwin, ywin, 'cubic')
            xwin = np.linspace(xwin[0], xwin[-1], len(xwin) * interp)
            ywin = interf(xwin)
        ax.plot(xwin, ywin, color = trace_color, linewidth = lw)
        low = min(min(ywin), low)
    if len(stim):
        level = 40e-12
        ax.plot(stim, [level] * len(stim), 'b|', ms = 10, mew = 2, linewidth = lw)
    if ave:
        ave_trace /= len(cell_num)
        ax.plot(xwin, ave_trace, color = ave_color, linewidth = lw)

    if units is not None:
        xscale_text = '{0:d} '.format(scalebar[0]) + units[0]
        yscale_text = '{0:d} '.format(scalebar[1]) + units[1]
    else:
        yscale_text, xscale_text = '', ''
    xscalebar = scalebar[0] / scale[0]
    yscalebar = scalebar[1] / scale[1]
    #sc = [t[win_ind2] - xscalebar * 0.5, np.mean(ylim)]
    sc = [t[win_ind2] - xscalebar * 0.5, low]
        # coordinate of top right corner of scale bar
    if yscalebar != 0:
        yscale = ax.plot([sc[0] - xscalebar] * 2, [sc[1] - yscalebar, sc[1]], \
            color = [0, 0, 0, 1], linewidth = 2.0)
        t2 = ax.text(sc[0] - xscalebar * 1.01, sc[1] - yscalebar * 0.8, \
            yscale_text, ha = 'right', va = 'bottom', rotation = 'vertical', size = 20)
    if xscalebar != 0:
        xscale = ax.plot([sc[0] - xscalebar, sc[0]], [sc[1] - yscalebar] * 2, \
            color = [0, 0, 0, 1], linewidth = 2.0)
        t1 = ax.text(sc[0] - xscalebar * 0.8, sc[1] - yscalebar * 1.05, \
            xscale_text, va = 'top', size = 20)
    ax.set_xlim(*win)
    ax.set_ylim(*ylim)
    #f.set_size_inches(3, 5)
    f.savefig(fname, dpi = 200, transparent = True)
    plt.close(f)
    del(f)
    return 0

def current(win, step_win, steps, unit, ylim, sr = 1000, lw = 1):
    '''
    current(win, step_win, steps, unit, ylim, sr = 1000)
        Plot current steps stimulation
    parameters:
        win (array_like) - two scalars range of the trace
        step_win (array_like) - two scalars range of the stimulation
        unit (string) - unit of current value
        ylim (array_like) - two scalars range of the y axis
        sr (float) - sampling rate for generating the steps
    '''
        
    t= np.arange(*win, 1 / sr)
    f = plt.figure()
    ax = f.add_subplot(111)

    for st in steps:
        trace = np.zeros(t.shape)
        trace[int((step_win[0] - win[0])* sr):int((step_win[1] - win[0])* sr)] = st
        text = str(st) + ' ' + unit
        text_pos = (0.9 * step_win[1] + 0.1 * win[1], st)
        ax.plot(t, trace, color = [0, 0, 0, 1], linewidth = lw)
        ax.text(*text_pos, text, va = 'top', size = 20)

    ax.set_xlim(*win)
    ax.set_ylim(*ylim)
    ax.axis('off')
    f.savefig('tmp.png', dpi = 96)
    plt.close(f)
    del(f)

def FI_curve(data_file, type_file = None, ave = False, stims = [], cl = [], \
    cells = [], sigtp = 0, out = 'tmp.png'):
    '''
    FI_curve(data_file, type_file = None, ave = False, stims = [], cl = [], out = 'tmp.png')
        Plot FI curves, for each cell, average replicated traces with the same current
        stimulation.
    parameters:
        data_file (string) - directory of data file with firing rate data, refer to 
            ap.firing_rate
        type_file (string) - directory of cell type csv files with cell indices in column No
            type value (0, 1, 2, ...) in column group. If not provided, won't 
            differentiate cell types.
        ave (boolean) - whether to average cells of the same type
        stim (array_like) - stimulation current steps. If not provided, use the steps of the 
            first cell from the data file, assuming all the cells have the same steps as the 
            first one.
        cl (array_like) - color of the different types. If type is specified but color is not 
            provided, generate color from gist_rainbow color map.
        cells (array_like) - ids of cells to be analyze, default is an empty list, this will
            calculate all the cells in the type_file.
        sigtp (float) - significance test p-value, default is 0, means no test
        out (string) - directory of output figure file
    '''

    data = util.read_dict(data_file, 'int')
    type_data = pd.read_csv(type_file)
    if not len(stims):
        stims = list(data.values())[0][0][0]
    else:
        stims = list(stims)
    ind = 0
    if len(cells):
        keys = cells
    elif type_file != None:
        keys = type_data['No']
    else:
        keys = data.keys()
    crates = np.empty((len(keys), len(stims)))
    crates[:] = np.nan
    _cells = []
    for key in keys:
        values = data[key]
        _cells.append(key)
        '''
        _stim = values[0]
        _target = np.array([stims]).T * np.ones((1, len(_stim)))
        ind = np.nonzero(_target == _stim)[1]
        print(key)
        print(np.array(values[1])[:, ind].mean(0).reshape((1, -1)).shape)
        if 'crates' in locals():
            crates = np.vstack((crates, \
                np.array(values[1])[:, ind].mean(0).reshape((1, -1))))
        else:
            crates = np.array(values[1])[:, ind].mean(0).reshape((1, -1))
        '''
        _stim = np.array(values[0][0])
        for s in stims:
            s_ind = np.nonzero(abs(_stim - s) < 1e-14)[0]
            if len(s_ind):
                crates[ind][stims.index(s)] = values[0][1][s_ind[0]]
        ind = ind + 1

    if len(cells):
        cells = np.array(cells)
        crates = crates[[_cells.index(d) for d in cells], :]
    else:
        cells = np.array(_cells)
    stims = np.array(stims) * 1e12

    if type_file != None:
        '''
        type_data = util.read_csv(type_file)
        types = type_data[np.nonzero(type_data[:, [0]] == \
            np.ones((type_data.shape[0], 1)) * cells)[1], -1]
        '''
        types = type_data.loc[np.nonzero(np.array(type_data['No']) == 
            cells.reshape(-1, 1) * np.ones((1, len(type_data.index))))[1], 'group']
        if sigtp != 0 and len(np.unique(types)) == 2:
            ps = [];
            for i in range(len(stims)):
                p = util.permutationTest(*[crates[types == d, i].flatten() \
                        for d in np.unique(types)])
                ps.append(p)
            print(ps)
            ps = np.array(ps) < sigtp

    print('type', types)
    print('cells', cells)
    f = plt.figure()
    ax = f.add_subplot(111)
    if ave:
        if type_file == None:
            mrates = np.nanmean(crates, 0)
            se = np.nanstd(crates, 0) / crates.shape[0]
            ax.errorbar(stims, mrates, se, ecolor = 'k', label = 'Average')
        else:
            if not len(cl):
                ncolors = len(np.unique(types))
                cm = plt.get_cmap('gist_rainbow')
                cl = [cm(1 * i / ncolors) for i in range(ncolors)]
            for t, color in zip(np.unique(types), cl):
                print('t', t)
                print('color', color)
                _crates = crates[types == t, :]
                print('cells', cells[types == t])
                mrates = np.nanmean(_crates, 0)
                se = np.nanstd(_crates, 0) / _crates.shape[0]
                ax.errorbar(stims, mrates, se, color = color, \
                    label = t, lw = 2)
    else:
        if type_file == None:
            if not len(cl):
                ncolors = len(np.unique(types))
                cm = plt.get_cmap('gist_rainbow')
                cl = [cm(1 * i / ncolors) for i in range(ncolors)]
            for crate, color, c in zip(ctates, cl, cells):
                ax.plot(stims, crate, ecolor = color, label = str(c))
        else:
            if not len(cl):
                ncolors = len(np.unique(types))
                cm = plt.get_cmap('gist_rainbow')
                cl = [cm(1 * i / ncolors) for i in range(ncolors)]
            for i, t in enumerate(np.unique(types)):
                _crates = crates[types == t, :]
                for crate in _crates:
                    ax.plot(stims, crate, c = cl[i])#, label = str(t))
    ax.legend(loc = 2)
    ax.set_xlabel('Current (pA)')
    ax.set_ylabel('Firing rate (Hz)')
    mpl.rcParams['font.size'] = 30

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim([stims[0], stims[-1] + 5])
    f.savefig(out, dpi = 96, bbox_inches = 'tight', transparent = True)
    plt.close(f)
    del(f)
    # return f
    return 0

def sample_aps(folder, cell_num, trial_num, train_win = [], spike_win = [-0.5e-3, 3e-3], \
    freq_range = [], ap_ind = [], type_file = None, ave = False, cl = [], lw = 1, \
    units = None, scale = [1, 1], scalebar = [0, 0], interp = 0, fname = 'tmp.png'):
    '''
    sample_aps(folder, cell_num, trial_num, train_win, spike_win = [-0.5e-3, 3e-3], \
        freq_range = [], ap_ind = [], type_file = None, ave = False, cl = [], \
        fname = 'tmp.png'):
        plot action potentials of each cells with all the qualified action potentials 
        averaged and depending on the input action potentials of cells in the same
        groups averages. In the latter case, plot shade standard of error.
    parameters:
        folder (String) - directory of the folder with the data
        cell_num (array_o) - indices of cells to plot
        trial_num (list) - indices of trails in each cell
        train_win (list) - of two scalar values for the time window where there are spikes
        spike_win (list) - of two scalar values for the time windoe for the spikes
        freq_range (list) - of two scalar, range of frequencies to use
        ap_ind (list) - index of action potentials in each trial to use
        type_file (String) - directory of the csv file recording the type of cells, the 
            first column has the cell indices and the last column has the cells' types
        ave (boolean) - whether to average across cells with same type
        cl (list) - color traces in each type, use default color sequences if not provided
        interp (float) - time of number of interpolated points to plot over current 
            number of points
        scale (list) -  2 scalar element [sx, sy], scaling factor for time and value axis
            sx and sy respectively.
        units (list) - 2 string elements list [ux, uy], units after scaling.
        scalebar (list) - 2 scalar elements list [sx, sy] - scale bar lengh of the two axis.
        fname ('String') - directory of the file to save the image
    '''

    if folder[-1] != '/':
        folder += '/'
    spike_params = ap.get_params('param_spike_detect')
    spikes = []
    for cell in cell_num:
        print('cell: ', cell)
        _spikes = 0
        count = 0
        for trial in trial_num:
            _trace, _sr, _stim_i = util.load_wave(folder + util.gen_name(cell, trial))
            if len(_trace) == 0:
                continue
            trace, sr, stim_i = _trace, _sr, _stim_i
            if len(train_win):
                spike_start = ap.spike_detect(trace, sr, spike_params, train_win[0], \
                    train_win[1])
                freq = len(spike_start) / (train_win[1] - train_win[0])
            else:
                spike_start = ap.trace_detect(trace, sr, spike_params)
                freq = len(spike_start) / len(trace) * sr
            if not (len(freq_range) and (freq < freq_range[0] or freq_range[1] < freq)):
                if len(ap_ind):
                    for i in ap_ind:
                        pre_ind = max(spike_start[i] + int(spike_win[0] * sr), 0)
                        post_ind = min(spike_start[i] + int(spike_win[1] * sr), len(trace))
                        single = trace[pre_ind:post_ind] - trace[spike_start[i]]
                else:
                    for t in spike_start:
                        pre_ind = max(int((t + spike_win[0]) * sr), 0)
                        post_ind = min(int((t + spike_win[1]) * sr), len(trace))
                        single = trace[pre_ind:post_ind] - trace[t]
                if interp:
                    interf = interpolate.interp1d(np.arange(pre_ind, post_ind), \
                        single, 'cubic')
                    single = interf(np.arange(interp * pre_ind, interp * (post_ind - 1)) \
                        / interp)
                _spikes = _spikes + single 
                count += 1
        print('spike count: ', count)
        spikes.append(_spikes / count)
    spikes = np.array(spikes)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.axis('off')
    if interp:
        xt = np.arange(spikes.shape[1]) / sr / interp
    else:
        xt = np.arange(spikes.shape[1]) / sr
    if type_file != None:
        type_data = util.read_csv(type_file)
        type_data = type_data[np.nonzero(type_data[:, [0]] == \
            np.ones((type_data.shape[0], 1)) * np.array(cell_num))[0], :]
        types = np.unique(type_data[:, -1])
        if len(cl) != len(types):
            ncolors = len(types)
            cm = plt.get_cmap('gist_rainbow')
            cl = [cm(1 * i / ncolors) for i in range(ncolors)]
        if ave:
            for i, t in enumerate(types):
                group_spikes = spikes[np.nonzero(type_data[:, -1] == t)[0], :]
                m = np.mean(group_spikes, 0)
                se = np.std(group_spikes, 0) / np.sqrt(group_spikes.shape[0])
                # sd = np.std(group_spikes, 0)
                ax.plot(xt, m, color = cl[i], lw = lw)
                ax.fill_between(xt, m - se, m + se, facecolor = cl[i], edgecolor = 'none', \
                    alpha = 0.3)
        else:
            for i, cspikes in enumerate(spikes):
                ax.plot(xt, cspikes, color = cl[np.nonzero(types == type_data[i, -1])[0][0]], \
                    lw = lw)
    else:
        ax.plot(xt, spikes.T, lw = lw)

    if units is not None:
        xscale_text = '{0:d} '.format(scalebar[0]) + units[0]
        yscale_text = '{0:d} '.format(scalebar[1]) + units[1]
    else:
        yscale_text, xscale_text = '', ''
    xscalebar = scalebar[0] / scale[0]
    yscalebar = scalebar[1] / scale[1]
    sc = [xt[-1] - xscalebar * 0.5, ax.get_ylim()[0] - yscalebar * 0.5]
        # coordinate of top right corner of scale bar
    if yscalebar != 0:
        yscale = ax.plot([sc[0] - xscalebar] * 2, [sc[1] - yscalebar, sc[1]], \
            color = [0, 0, 0, 1], linewidth = 2.0)
        t2 = ax.text(sc[0] - xscalebar * 1.01, sc[1] - yscalebar * 0.8, \
            yscale_text, ha = 'right', va = 'bottom', rotation = 'vertical', size = 20)
    if xscalebar != 0:
        xscale = ax.plot([sc[0] - xscalebar, sc[0]], [sc[1] - yscalebar] * 2, \
            color = [0, 0, 0, 1], linewidth = 2.0)
        t1 = ax.text(sc[0] - xscalebar * 0.8, sc[1] - yscalebar * 1.05, \
            xscale_text, va = 'top', size = 20)

    f.set_size_inches(2, 7)
    f.savefig(fname, dpi = 200, transparent = True)
    plt.close(f)
    del(f)
    return 0

def latency_scatter(folder, group_file, group_code, pre = 'resp', \
    resp_i = 0, cl = [], out = 'tmp.png', annotate = 0):
    '''
    latency_scatter(folder, group_file, group_code, resp_i = 0, cl = [])
        Scatter plot of optogenetic response latancy and standard deviation of latency.
    parameters:
        folder (String) - directory of folder for the analyzed response data files.
        group_file (String) - directory of csv file with cell number, group information, 
            and intensities, responses under which will be plotted.
            The file has cell numbers in the first column and group number in the 
            second column. The group numbers are integer sequence starting from 0.
            The following columns are intensities conditions.
        group_code (array_like) - String elements storing names of cell groups in the 
            group_file.
        resp_i (int) - index of response to analyze in a train of responses.
        cl (list) - of color value recognized by matplotlib for different groups.
        out (String) - output image file directory.
        annotate (int) - whether and how to annotate the points with cell numbers
            0 - don't annotate, 1 - annotate with cell number,
            2 - annotate with cell number and intensity
    '''

    data = util.read_csv(group_file)
    cells = np.array([int(d) for d in data[:, 0]])
    groups = data[:, 1]
    mlat = np.empty((data.shape[0], data.shape[1] - 2))  # mean latency
    jitter = np.empty((data.shape[0], data.shape[1] - 2))  # latency jitter, std of latencies
    # allow different cells to have different number of intensity conditions
    # put NaN in position without a data sample
    mlat[:] = np.nan
    jitter[:] = np.nan
    for i, cell_num in enumerate(cells):
        for j, intensity in enumerate(np.extract(data[i, 2:], data[i, 2:])):
            if intensity is np.nan:
                break
            data_dir = folder + os.sep +  pre + '_cell_{:04d}_'.format(cell_num) + \
                str(int(round(intensity * 100))) + '.dat'
            _lat = util.read_arrays(data_dir, 'lat')
            latencies = _lat[np.nonzero(0 < _lat[:, resp_i])[0], resp_i]
            if len(latencies):
                mlat[i, j] = np.mean(latencies)
                jitter[i, j] = np.std(latencies)
    # Change unit to ms
    mlat = mlat * 1e3
    jitter = jitter * 1e3

    f = plt.figure()
    ax = f.add_subplot(111)
    if not len(cl):
        ncolors = len(group_code)
        cm = plt.get_cmap('gist_rainbow')
        cl = [mpl.colors.rgb2hex(cm(1 * i / ncolors)) for i in range(ncolors)]
    scatter = []
    print(cl)
    print('mlat', mlat)
    print('jitter', jitter)
    for group in range(len(group_code)):
        print(cl[group])
        ax.scatter(mlat[np.nonzero(groups == group)[0], :].flatten(), \
            jitter[np.nonzero(groups == group)[0], :].flatten(), \
            c = cl[group], edgecolors = 'face', label = group_code[group])
    if 0 < annotate:
        for i, cell_num in enumerate(cells):
            for j, intensity in enumerate(data[i, 2:]):
                if intensity is np.nan:
                    break
                if annotate < 2:
                    ax.annotate(str(cell_num), (mlat[i, j], jitter[i, j]), size = 10)
                else:
                    ax.annotate(str(cell_num) + ', ' + str(intensity), \
                        (mlat[i, j], jitter[i, j]), size = 10)

    ax.set_xlabel('Mean latency (ms)')
    ax.set_ylabel('Latency standard deivation (ms)')
    #ax.legend(scatter, group_code, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #mpl.rcParams['font.size'] = 26
    f.savefig(out, dpi = 96, bbox_inches = 'tight', transparent = True)
    plt.close(f)
    del(f)
    return 0

def tmpplot(yaxis, data, scale):
    y = data[:, range(1, 8, 2)]
    n = data[:, range(0, 7, 2)]
    ym = [np.mean(d[np.nonzero(d != -1)]) * scale for d in y.T]
    nm = [np.mean(d[np.nonzero(d != -1)]) * scale for d in n.T]
    points = np.hstack((np.array(ym).reshape(4, 1), np.array(nm).reshape(4, 1)))
    bar_graph([np.mean(ym), np.mean(nm)], yaxis, points = points, legendlabels = ['YFPH+', 'YFPH-'])
    
def resp_graph(folder, cond_file, group_code, props = ['amp', 'lat', 'slp', 'area'], \
    ylabels = ['Amplitude', 'Latency', 'Slope', 'Charge'], \
    scales = [1e12, 1e3, 1e9, 1e12], units = ['pA', 'ms', 'pA/ms', 'pC'], \
    pre = 'resp', resp_i = 0, cl = []):
    '''
    resp_graph(folder, group_file, group_code, props = ['amp', 'lat', 'slp', 'area'], \
        ylabels = ['Amplitude', 'Latency', 'Slope', 'Charge'], \
        scales = [1e12, 1e3, 1e9, 1e12], units = ['pA', 'ms', 'pA/ms', 'pC'], \
        pre = 'resp', resp_i = 0, cl = []):
        Plot bar graphs of optogenetic response properties.
    parameters:
        folder (string) - directory of folder with response property data and for image output
        cond_file (string) - directory of file defining groups and stimulation conditions, 
            1st column of the file has cell indices and the 2nd column has the group 
            values (0, 1, 2, ...) and the 3rd column has conditions.
        group_code (array_like) - string elements, name of groups corresponding to values
        props (array_like) - string elements, properties to plot, the same as in data files
        ylabels (array_like) - names shown in the Y-axis label for each properties
        scales (array_like) - scale values for each properties
        units (array_like) - units of each properties
        pre (string) - prefix of the data file name
        resp_i (int) - index of the response to look at in a train of reponses
        cl (array_like) - color of groups
    '''

    cond_data = util.read_csv(cond_file)
    cells = np.array(list(map(int, cond_data[:, 0])))
    groups = np.array(list(map(int, cond_data[:, 1])))
    if not len(cl):
        ncolors = len(group_code)
        cm = plt.get_cmap('gist_rainbow')
        cl = [cm(1 * i / ncolors) for i in range(ncolors)]
    for prop, ylabel, scale, unit in zip(props, ylabels, scales, units):
        mval = np.empty(len(cells))
        print('prop', prop)
        for i, cell_num in enumerate(cells):
            intensity = cond_data[i][2]  # use the first intensity in the intensity list
            data_dir = folder + os.sep +  pre + '_cell_{:04d}_'.format(cell_num) + \
                str(int(round(intensity * 100))) + '.dat'
            _val = util.read_arrays(data_dir, prop) 
            val = _val[np.nonzero((0 != _val[:, resp_i]) * (-1 != _val[:, resp_i]))[0], \
                resp_i]
            mval[i] = np.mean(val)
        mval = mval * scale # change unit to pA
        
        m = [np.mean(mval[np.nonzero(groups == group)]) \
            for group in range(len(group_code))]
        points = [mval[np.nonzero(groups == group)] for group in range(len(group_code))]
        bar_graph(m, ylabel + ' (' + unit + ')', points = points, colors = cl, \
            legendlabels = group_code, fname = folder + prop + '.png')

def IV_curve(folder, cells, data_file, type_file = '', out = 'IV_curve.png'):
    f = plt.figure()
    ax = f.add_subplot(111)
    data = util.read_dict(folder + data_file, 'int')
    if len(type_file):
        type_data = pd.read_csv(folder + type_file)
        groups = type_data['group']
        ncolors = len(np.unique(groups))
        cm = plt.get_cmap('gist_rainbow')
        cl = np.array([cm(1 * i / ncolors) for i in range(ncolors)])
    for cell in cells:
        if len(type_file):
            color = cl[np.nonzero(np.unique(groups) == \
                groups[np.nonzero(type_data['No'] == cell)[0][0]])[0][0]]
        else:
            color = 'k'
        ax.plot(data[cell][0], data[cell][1], color = color)
    f.savefig(folder + out, dpi = 200, transparent = True)
    return 0

def Fslope(folder, data_file, firing_rate_file, out, rates, cl, out_file):
    f = plt.figure()
    ax = f.add_subplot(111)
    aps = ap.Ap_prop(folder, data_file, firing_rate_file)
    for r in rates:
        aps.choose(10, [r, r+1])
        aps.get_spike_time()
        slopes = aps.calc_slope([0, 1], out + 'ap_slope.txt')
        for i, c in enumerate(slopes):
            color = cl[i]
            for t in c:
                if t[0] != 0:
                    ax.scatter(r, t[0], color = color)
    f.savefig(out + out_file, dpi = 200, transparent = True)
        
def overlap_trace(folder, cells, trials, win, basewin, legends = []):
    '''
    plot averaged traces of each cell in different colors, 
    '''
    if folder[-1] != os.sep:
        folder += os.sep
    ave_traces = [];
    for i in range(len(cells)):
        traces = [];
        for t in trials[i]:
            file_dir = folder + 'Cell_{0:04d}_{1:04d}.ibw'.format(\
                    cells[i], t)
            print(file_dir)
            trace, sr, st = util.load_wave(file_dir)
            trace = np.array(trace).copy()
            traces.append(trace)
        ave_traces.append(ahp.average(traces, sr, basewin = basewin))
        if not len(legends):
            lg.append('{0:d}. cell {1:d}'.format(i, cells[i]))
    if len(legends):
        lg = legends
    f = plot.overlap(ave_traces, sr, legends = lg, basewin = basewin, win = win)
    return f

def scatterPlot(dataFile, x, y, colorby, colors = None, output = "tmp.png"):
	"""
	Plot data in file in 2d coodinates of two attributes, x and y, and color by
	some categorical attributes.

	Arguments
	---------
	dataFile: string
		csv file with data to be plotted.
	x: string
		Name of the x-axis attribute.
	y: string
		Name of the y-axis attribute.
	coloby: string
		Name of the categorical attributes used to group the points.
	colors: list of colors acceptable to Matplotlib or None
		Colors use to plot the different groups. Default None, using colormap.
	output: string
		Directory of the output file.
	"""
	fig, ax = plt.subplots()
	data = pd.read_csv(dataFile)
	total = len(np.unique(data[colorby]))
	cm = plt.get_cmap("gist_rainbow")
	if colors == None:
		colors = [cm(1 * i / total) for i in range(total)]
	for i, g in enumerate(data.groupby(colorby)):
		ax.scatter(g[1][x], g[1][y], color = colors[i], label = g[0])
	ax.legend()
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	fig.savefig(output, dpi = 200, transparent = True)
