#!/usr/local/python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pandas as pd
import util
import plot
import process
import ap

def aveplot(intensity_dir, group_size, folder, cells, intensities, trial, start, freq, num, \
    win = None,  medfilt = True, smooth = False, base_win = 0.3, col = 3, row = 3):
    '''
    f = aveplot(intensity_dir, group_size, folder, cells, intensities, trial, start, freq, \
        num, medfilt = True, smooth = False, base_win = 0.3):
        Plot the averaged responses across several trials with stimulus of certain 
        intensity for different cells at different intensity levels.
    parameters:
        intensity_dir (string) - intensity data file
        folder (string) - dirctory to the data folder
        cells (array_like) - indices of cells
        intensities (array_like) - intensities of the trials to be plotted for each cell
        trial (array_like) - number of trials in each intensity group to be plotted
        start (float) - time of stimulation start
        freq (float) - frequency of stimulation
        num (int) - number of stimuli
        medfilt (float) - median filter threshold
        smooth (boolean) - whether to smooth the traces with default parameters
        base_win (float) - size of baseline window, right before the first stimulation used
            for baseline alignment
        col (int) - number of columns of subplots
        row (int) - number of rows of subplots
    return:
        f (GraphicsWindow) - window object of pyqtgraph package
    '''
    f = []
    count = 0
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    for i, cell in enumerate(cells):
        for intensity in intensities[i]:
            file_dirs = util.intensity2file(folder, [cell], [intensity], trial, \
                intensity_dir, group_size)
            traces = []
            sr = 0
            if len(file_dirs) == 0:
                continue
            for fd in file_dirs:
                t, sr, stim = util.load_wave(fd)
                if base_win:
                    baseline = np.mean(t[int(start - base_win):int(start)])
                    t = t - baseline
                if smooth:
                    t = process.smooth(t, sr, 600)
                if medfilt:
                    t = process.thmedfilt(t, 5, 30e-12)
                traces.append(t)
            ave_trace = np.mean(traces, 0)
            #print(np.mean(ave_trace))
            sub_num = count - row * col * (len(f) - 1)
            if row * col <= sub_num:
                f.append(pg.GraphicsWindow(title = \
                    'Group {0:d}'.format(len(f) + 1)))
                sub_num -= row * col
            axis = f[-1].addPlot(int(sub_num / col), np.mod(sub_num, col))
            axis.setTitle('Cell {0:d}, Int {1:.2f}'.format(cell, intensity))
            plot.plot_trace_v(ave_trace, sr, ax = axis, win = win, \
                stim = [start + d / freq for d in range(num)])
            count += 1
    return f

def multiplot(folder, cells, trials, start, freq, num, intensity_dir = '', group_size = 0, \
    intensity = [0], window = 0, medfilt = True, smooth = False, average = False, \
    base_win = 0.3, lat = []):
    '''
    f = multiplot(folder, cell, trials, start, freq, num, window = 0, medfilt = 0, \
        smooth = False, average = False):
        Overlap segments of one or more traces after stimulus or overlap several traces, with
        the stimulation time point marked.
    parameters:
        folder (string) - dirctory to the data folder
        cell (array_like) - indices of cells
        trial (array_like) - number of trials in each intensity group to be plotted
        start (float) - time of stimulation start
        freq (float) - frequency of stimulation
        num (int) - number of stimulations
        intensity_dir (string) - intensity data file, not used when it's empty
        group_size (int) - number of trials in each group of the same intensity, not used
            when it's 0
        intensity (array_like) - intensities of the trials to be plotted
        window (float) - window size of each segment after the stimulation to be plotted
            when it's 0, plot the entire trace and mark the stimulus time ponit instead
        medfilt (float) - median filter threshold
        smooth (boolean) - whether to smooth the traces with default parameters
        average (boolean) - whether to plot the average trace
        base_win (float) - size of baseline window, right before the first stimulation used
            for baseline alignment
        lat (array_like) - latencies of the responses to plot
    return:
        f (GraphicsWindow) - window object of pyqtgraph package
    '''
    if folder[-1] != os.sep:
        folder += os.sep
    if group_size:
        file_dirs = util.intensity2file(folder, cells, intensity, trials, intensity_dir, \
            group_size)
    else:
        file_dirs = []
        for cell_num in cells:
            for trial in trials:
                file_dirs.append(folder + 'Cell_{0:04d}_{1:04d}.ibw'.format(cell_num, trial))
    traces = []
    lat_data = []
    sr = 0
    for fd in file_dirs:
        t, sr, stim = util.load_wave(fd)
        if base_win:
            baseline = np.mean(t[int(start - base_win):int(start)])
            t = t - baseline
        if smooth:
            t = process.smooth(t, sr, 600)
        if medfilt:
            t = process.thmedfilt(t, 5, 30e-12)
        traces.append(t)
        if(len(lat)):
            #lat_data.append(lat[int(fd[-8:-4]) - 1])
            lat_data.append(lat[file_dirs.index(fd)])

    if window:
        segs = plot.gen_seg(freq, num, start, window)
        seg_traces = []
        rise_points = []
        if(len(lat)):
            for t, trial_lat in zip(traces, lat_data):
                for s, l in zip(segs, trial_lat):
                    seg_traces.append(t[int(s[0] * sr):int(s[1] * sr)])
                    if s[0] + l < s[1] and 0 < l:
                        rise_points.append([l])
                    else:
                        rise_points.append([])
        else:
            for t in traces:
                for s in segs:
                    seg_traces.append(t[int(s[0] * sr):int(s[1] * sr)])

        legends = []
        for c in cells:
            for i in intensity:
                for t in trials:
                    for s in range(len(segs)):
                        if group_size:
                            legends.append('cell{0:d}stim{1:.1f}t{2:d}s{3:d}'.format(c, \
                                i, t, s + 1))
                        else:
                            legends.append('cell{0:d}t{1:d}s{2:d}'.format(c, \
                                t, s + 1))
        if average:
            s = np.zeros(len(seg_traces[0]))
            for st in seg_traces:
                s = s + st
            ave = s / len(seg_traces)
            seg_traces.append(ave)
            if(len(lat)):
                rise_points.append([])
            legends.append('Average')
        f = plot.overlap(seg_traces, sr, legends = legends, rise = rise_points)
    else:
        legends = []
        for c in cells:
            for i in intensity:
                for t in trials:
                    if group_size:
                        legends.append('cell{0:d}stim{1:.1f}t{2:d}'.format(c, i, t))
                    else:
                        legends.append('cell{0:d}t{1:d}'.format(c, t))
        if average:
            s = np.zeros(len(traces[0]))
            for t in traces:
                s = s + t
            ave = s / len(traces)
            traces.append(ave)
            legends.append('Average')
        if(len(lat)):
            rise_points = []
            stim = [start + d / freq for d in range(num)]
            for i, trial_lat in enumerate(lat_data):
                rise_points.append([])
                for j, s in enumerate(stim):
                    rise_points[i].append(s + trial_lat[j])
            f = plot.overlap(traces, sr, stim = stim, legends = legends, rise = lat_data)
        else:
            f = plot.overlap(traces, sr, stim = stim, legends = legends)
    return f

def browse(folder, cell_num, trial_num = None, row = 3, col = 3, medfilt = True): 
    '''
    f = browse(folder, cell_num, trial_num = None, row = 3, col = 3, medfilt = True): 
        Plot data traces of a cell in figures with multiple subplots to quicky review 
        the traces
    parameters:
        folder (string) - directory to the folder with the data files
        cell_num (int) - cell index
        trial_num (int or list) - trial index or indices, include all the trials if it's None
        row (int) - number of rows of subplots
        col (int) - number of cols of subplots
        medfilt (boolean) - whether to apply median filter with a default threshold
    return:
        f (list) - list of window objects of pyqtgraph package
    '''
    if(folder[-1] is not os.sep):
        folder = folder + os.sep
    if type(trial_num) is int:
        file_dir = folder + 'Cell_{0:04d}_{1:04d}.ibw'.format(cell_num, trial_num)
        print(file_dir)
        trace, sr, stim = util.load_wave(file_dir)
        if not len(trace):
            return
        if medfilt:
            trace = process.thmedfilt(trace, 5, 40e-12)
        f = plot.plot_trace_v(trace, sr)
        if stim[2] != 0:
            f.getItem(0, 0).setTitle('Trial {0:d}, I = {1:.2e}'.format(trial_num, stim[2]))
        #f.save_fig('tmp.png', dpi = 96)
    elif type(trial_num) is list:
        f = []
        for i in range(len(trial_num)):
            file_dir = folder + 'Cell_{0:04d}_{1:04d}.ibw'.format(cell_num, trial_num[i])
            print(file_dir)
            trace, sr, stim = util.load_wave(file_dir)
            if not len(trace):
                continue
            if medfilt:
                trace = process.thmedfilt(trace, 5, 40e-12)
            sub_num = i - row * col * (len(f) - 1)
            if row * col <= sub_num:
                f.append(pg.GraphicsWindow(title = \
                    'Cell {0:d}, Group {1:d}'.format(cell_num, len(f) + 1)))
                sub_num -= row * col
            #axis = f[-1].addPlot(int(sub_num / row), np.mod(sub_num, col), \
            axis = f[-1].addPlot(int(sub_num / col), np.mod(sub_num, col))
            if stim[2] != 0:
                axis.setTitle('Trial {0:d}, I = {1:.2e}'.format(trial_num[i], stim[2]))
            else:
                axis.setTitle('Trial {0:d}'.format(trial_num[i]))
            plot.plot_trace_v(trace, sr, ax = axis)
    else:
        f = []
        file_pre = 'Cell_{0:04d}_'.format(cell_num)
        print(file_pre)
        data_files = os.listdir(folder)
        count = 0
        for data_file in data_files:
            matched = re.match(file_pre + '0*([1-9][0-9]*).ibw', data_file)
            if matched:
                trial_num = int(matched.group(1))
                sub_num = count - row * col * (len(f) - 1)
                if row * col <= sub_num:
                    f.append(pg.GraphicsWindow(title = 
                        'Cell {0:d}, Group {1:d}'.format(cell_num, len(f) + 1)))
                    sub_num -= row * col
                axis = f[-1].addPlot(int(sub_num / col), np.mod(sub_num, col))
                file_dir = folder + data_file
                trace, sr, stim = util.load_wave(file_dir)
                print(file_dir + ' {:f}'.format(len(trace) / sr))
                if stim[2] != 0:
                    axis.setTitle('Trial {0:d}, I = {1:.2e}'.format(trial_num, stim[2]))
                else:
                    axis.setTitle('Trial {0:d}'.format(trial_num))
                if medfilt:
                    trace = process.thmedfilt(trace, 5, 40e-12)
                plot.plot_trace_v(trace, sr, ax = axis)
                count += 1
    return f

class Fisample:
    ''' 
    To plot samples from firing data from current clamp recording. Has methods for choosing
    samples based on their firing rate and methods for plotting the samples in different ways.
    '''
    def __init__(self, folder, data_file, fi_file):
        '''
        Get basic information about the data.
        parameters:
            folder (String) - directory to folder of the raw data files
            data_file (String) - directory to the data file with the cell type info
            fi_file (String) - directory to the data file with the firing rate data
        '''
        self.folder = folder  # raw data folder directory
        self.data = pd.read_csv(data_file)  # cell type info data
        self.fi_data = util.read_dict(fi_file, 'int')  # firing rate and stim current data
        self.trial_data = []  # chosen data to plot

    def choose(self, trial_num = 1, target_range = [], target_rate = 0):
        '''
        Choose samples based on the firing rates.
        Parameters:
            trial_num (int) - number of trials to plot from each cell
            target_range (array_like) - begin and end of the range of firing rate, choose
                trials with rates within the range. Omit end as inf. Not considered in empty.
                It's possible to end up with less than trial_num of trials if there's not 
                enough trial within the trails_range.
            target_rate (float) - the target rate, choose the trial_num of traces with 
                firing rate closest to the target rate. If range is also defined, apply both.
        '''
        cell_num = len(self.data.index)
        self.trial_data = np.zeros((cell_num, trial_num, 3))  # chosne trials 
            # including the indices, the firing rates and stim currents 
            # of those trials
        for i in self.data.index:
            # expand the data in the firing rate data dictionary
            _data = self.fi_data[self.data['No'][i]][1]
            rates = np.array(_data[2])
            trials = np.array(_data[0])
            stim = np.array(_data[1])
            rel_rates = rates - target_rate  # rates relative to target
            # sort the absolute values in elative rates to get the order based on distance 
            # to the target rate
            sind = np.argsort(np.abs(rel_rates))
            counter = 0
            for j in sind:
                # start from the nearest to the target rate
                if not ((len(target_range) == 1 and rates[j] < target_range[0]) or \
                    (len(target_range) == 2 and (rates[j] < target_range[0] or \
                    target_range[1] <= rates[j]))):
                    # if the rate is within the target range, record the trace
                    self.trial_data[i, counter] = trials[j], rates[j], stim[j]
                    counter += 1
                    if counter >= trial_num:
                        break
        
    def browse(self, row = 3, col = 3):
        '''
        Plot the raw traces grouped close to each other based on the groups of cells
        parameters:
            row (int) - number of rows of subplots
            col (int) - number of columns of subplots
        return:
            f (list) - list of image windows
        '''
        f = []
        for t in np.unique(self.data['group']):
            cell_ind = self.data.index[np.nonzero(self.data['group'] == t)]
            ft = []
            counter = 0
            for c in cell_ind:
                for trial in self.trial_data[c, :, :]:
                    trace, sr, stim = util.load_wave(self.folder + \
                        util.gen_name(self.data['No'][c], trial[0]))
                    if not len(trace):
                        continue
                    sub_num = counter - row * col * (len(ft) - 1)
                    if row * col <= sub_num:
                        ft.append(pg.GraphicsWindow(title = \
                            'Image {:d}'.format(len(ft) + 1) + ' in ' + t))
                        sub_num -= row * col
                    axis = ft[-1].addPlot(int(sub_num / col), np.mod(sub_num, col))
                    axis.setTitle('Cell {0:d}, rate = {1:.0f} Hz, I = {2:.0f} pA'.format(\
                        self.data['No'][c], trial[1], trial[2] * 1e12))
                    plot.plot_trace_v(trace, sr, ax = axis)
                    counter += 1
            f.extend(ft)
        return f

    def align_ahp(self, spikes = [0], max_trace = 30, param_file = 'param_spike_detect'):
        '''
        Align and plot specified action potentials from chosen trials in the same graphs.
        parameters:
            spikes (list) - indices of spikes to plot, 0 is the first
            max_trace (int) - maximum number of traces in each graph, not to be too crowded
            param_file (String) - directory of spike detection parameter file
        return:
            f (list) - list of image windows
        '''
        # Traverse all the trials, find the spikes and store the trace, time range 
        # of the spikes and the type of the cell which the trace belong to in 
        # separated lists sequentially
        traces = [[] for d in range(len(spikes))]
        limits = [[] for d in range(len(spikes))]
        cell_types = [[] for d in range(len(spikes))]
        spike_params = ap.get_params(param_file)
        for c in self.data.index:
            for trial in self.trial_data[c, :, :]:
                trace, sr, stim = util.load_wave(self.folder + \
                    util.gen_name(self.data['No'][c], trial[0]))
                if sr > 0:
                    starts = ap.spike_detect(trace, sr, spike_params)  # start of all spikes
                    for i, spike in enumerate(spikes):
                        if spike < len(starts) - 1:
                            traces[i].append(trace)
                            cell_types[i].append(self.data['group'][c])
                            limits[i].append([starts[spike], starts[spike + 1]])
        
        f = []
        for i, spike in enumerate(spikes):
            types = np.sort(np.unique(cell_types[i]))
            image_num = int(np.ceil(len(cell_types[i]) / max_trace))  # number of images
            fs = [pg.GraphicsWindow(title = 'Image {0:d} in spike {1:d}'.format(d, spike)) \
                for d in range(image_num)]
            ax = [d.addPlot(0, 0) for d in fs]
            cl = [pg.intColor(i, hues = len(types)) for i in range(len(types))]  # colors
            # Add legend to the plotItems
            lgit = [pg.PlotDataItem(pen = cl[d]) for d in range(len(cl))]
            for ax_ in ax:
                lg = ax_.addLegend()
                for t, l in zip(types, lgit):
                    lg.addItem(l, t)
            group_max = []  # maximum number of traces in one image for each groups
            for t in types:
                group_max.append(np.ceil(np.count_nonzero(np.array(cell_types[i]) == t) \
                    / image_num))
            group_trial = [0 for d in range(len(types))]  # keep track of trials 
                # plotted in each groups
            for j in range(len(cell_types[i])):
                _group = np.nonzero(types == cell_types[i][j])[0][0]
                plot.plot_trace_v(traces[i][j], 1, win = limits[i][j], ax = \
                    ax[int(np.floor(group_trial[_group] / group_max[_group]))], \
                    shift = [-limits[i][j][0], -traces[i][j][limits[i][j][0]]], \
                    cl = cl[_group])
                group_trial[_group] += 1
            f.extend(fs)
        return f

    def align(self, baseline_win = 0.2, max_trace = 30):
        '''
        Plot the chosen trials in the same graphs, aligned to baseline and colored based 
        on groups 
        parameters:
            baseline_win (float) - baseline window size before the start of the stimulation
            max_trace (int) - maximum number of traces in each graph
        retrun:
            f (list) - list of image windows
        '''
        types = np.sort(np.unique(self.data['group']))
        trial_num = np.count_nonzero(self.trial_data[:, :, 0])
        # trial_num = self.trial_data.shape[0] * self.trial_data.shape[1]
        image_num = int(np.ceil(trial_num / max_trace))  # number of images
        f = [pg.GraphicsWindow(title = 'Image {0:d}'.format(d)) for d in range(image_num)]
        ax = [d.addPlot(0, 0) for d in f]
        cl = [pg.intColor(i, hues = len(types)) for i in range(len(types))]  # colors
        # Add legend to the plotItems
        lgit = [pg.PlotDataItem(pen = cl[d]) for d in range(len(cl))]
        for ax_ in ax:
            lg = ax_.addLegend()
            for t, l in zip(types, lgit):
                lg.addItem(l, t)
        counter = 0
        for c in self.data.index:
            for t in self.trial_data[c]:
                if t[0] == 0:
                    break
                trace, sr, stim = util.load_wave(self.folder + \
                    util.gen_name(self.data['No'][c], t[0]))
                if sr > 0:
                    _group = np.nonzero(types == self.data['group'][c])[0][0]
                    plot.plot_trace_v(trace, sr, ax = \
                        ax[int(np.floor(counter / max_trace))], shift = \
                        [0, -np.mean(trace[int(stim[0] - baseline_win):int(stim[0])])], \
                        cl = cl[_group])
                    counter += 1
        return f

def FIcompare(folder, cells, currents = [], freqs = [],\
    firing_rate_data = 'firing_rate_data.txt'):
    '''
    f = FIcompare(folder, cells, currents = [], freqs = [],\
        firing_rate_data = 'firing_rate_data.txt'):
        Plot current clamp firing traces with certain currents input and with firing
        frequencies in a certain range.
    parameters:
        folder (string) - directory to the folder with raw data
        cells (array_like) - indices of neurons to plot
        currents (array_like) - list of input currents
        freqs (list) - of two scalars, range of the firing rates to be included
        firing_rate_data (string) - firing rate data file directory
    return:
        f (list) - list of figure windows
    '''
    data = util.read_dict(firing_rate_data, 'int')
    f = []
    for cell in cells:
        for trial, stim, fr in zip(*data[cell][1]):
            if (len(currents) == 0 or stim in currents) and \
                (len(freqs) == 0 or (freqs[0] <= fr and fr < freqs[1])):
                trace, sr, st = util.load_wave(folder + util.gen_name(cell, trial))
                f.append(plot.plot_trace_v(trace, sr))
                f[-1].setWindowTitle('Cell {0:d}, Trial {1:d}, I = {2:.2e}'.\
                    format(cell, trial, st[2]))
    return f
