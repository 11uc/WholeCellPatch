#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

def plot_seg(segs, trace, sr, subx = 2, suby = 2):
    ''' 
    plot_seg(segs, trace, sr, subx = 2, suby = 2):
        Plot segments of one trace seperately in subplots in one or more figures if needed.
        Save the figure in 
    parameters:
        segs (list) - start and end pair of each time segments.
        trace (array_like) - signal trace.
        sr (float) - sampling rate of the trace.
        subx (int) - number of columns of subplots.
        suby (int) - number of rows of subplots.
    returns:
        0
    '''

    figs = []
    axes = []
    lines = []
    fig_num = 0
    for plot_num in range(len(segs)):
        subplot_num = plot_num % (subx * suby)
        if(not subplot_num):
            figs.append(plt.figure(fig_num))
            fig_num += 1
        axes.append(figs[-1].add_subplot(subx, suby, subplot_num + 1))
        ind1 = int(segs[plot_num][0] * sr)
        ind2 = int(segs[plot_num][1] * sr)
        time = np.arange(ind1, ind2) / sr
        lines.append(axes[-1].plot(time, trace[ind1:ind2]))
        axes[-1].set_xlim(time[0], time[-1] + 1 / sr)
        for label in axes[-1].get_xticklabels() + axes[-1].get_yticklabels():
            label.set_fontsize(5)

    for i in range(fig_num):
        plt.figure(i)
        plt.savefig('trace_name_fig{0:02d}'.format(i), dpi = 200)

def gen_seg(stim_rate, stim_num, start_time, window):
    '''
    segs = gen_seg(stim_rate, stim_num, start_time, window):
        Generate time segments of optogenetic stimulation for plotting.
    parameters:
        stim_rate (float) - rate of stimulation train
        stim_num (int) - number of stimulation pulses
        start_time (float) - stimulation start time
        window (float) - duration of each stimulation pulse
    return:
        segs (array_like) - array of pair of start and end time of each pulse
    '''
    return \
        [(start_time + i / stim_rate, start_time + i / stim_rate + window) \
        for i in range(stim_num)]

def plot_trace(trace, sr, smooth_trace = None , points = None, win = None, ax = None): 
    '''
    f = plot_trace(trace, sr, smooth_trace = None , points = None, win = None, ax = None): 
        Plot recorded traces with Matplotlib.
    parameters:
        trace (array_like) - time sequence of recorded signal
        sr (float) - sampling rate
        smooth_trace (array_like) - smoothed trace, plot in the same axis if it's not None
        points (array_like) - time points to hightlight on the trace if it's not None
        win (array_like) - a pair of two scalars, time window of the trace to plot. If 
            it's None, plot the entire trace.
        ax (class 'matplotlib.axes._subplots.AxesSubplot') - matplotlib axis for plotting,
            if None, make a new figure object for plotting
    return:
        f (matplotlib.figure.Figure) - figure with the plots, if ax is not None, return 0
    '''
    t = np.arange(len(trace)) / sr
    if win is None:
        win = [0, len(trace)]
    else:
        win = [int(d * sr) for d in win]
    f = 0
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(t[win[0]:win[1]], trace[win[0]:win[1]])
    if smooth_trace is not None:
        ax.plot(t[win[0]:win[1]], smooth_trace[win[0]:win[1]])
    if points is not None and len(points):
        points = np.array(points)
        points_in = points[(win[0] < points) * (points < win[1])]
        ax.plot(t[points_in], smooth_trace[points_in], 'o')
    return f

def plot_trace_v(trace, sr, smooth_trace = None , points = None, stim = None, \
        shift = [], cl = 'w', pcl = 'b', win = None, ax = None): 
    '''
    plot_trace_v(trace, sr, smooth_trace = None , points = None, stim = None, \
        win = None, ax = None): 
        Plot recorded traces with PyQtGraph.
    parameters:
        same as plot_trace, except for
        stim (array_like) - stimulation time points
    return:
        same as plot_trace
    '''

    t = np.arange(len(trace)) / sr
    if len(shift):
        t = t + shift[0]
        trace = trace + shift[1]
    if win is None:
        win = [0, len(trace)]
    else:
        win = [int(d * sr) for d in win]
    f = 0
    if ax is None:
        f = pg.GraphicsWindow()
        ax = f.addPlot()
    ax.plot(t[win[0]:win[1]], trace[win[0]:win[1]], pen = cl)
    # ax.plot(t[win[0]:win[1]], trace[win[0]:win[1]])
    if smooth_trace is not None:
        ax.plot(t[win[0]:win[1]], smooth_trace[win[0]:win[1]], pen = 'g')
    if points is not None and len(points):
        points = (np.array(points) * sr).astype(int)
        points_in = points[(win[0] < points) * (points < win[1])]
        ax.plot(t[points_in], trace[points_in], pen = None, \
                symbol = 'o', symbolBrush = pcl)
    if stim is not None:
        yr = max(trace) - min(trace)
        for st in stim:
            y1 = trace[int(st / sr)] - 0.1 * yr
            y2 = trace[int(st / sr)] + 0.1 * yr
            ax.plot([st, st], [y1, y2], pen = pg.mkPen('k')) 
    return f

def overlap_v(traces, sr, stim = [], legends = [], rise = [], basewin = [], win = []):
    '''
    Overlap multiple traces or segments of traces.
    parameters:
        traces (list) - list of traces to plot
        sr (float) - sampling rate
        stim (array_like) - stimulus time points
        legends (array_like) - legend strings for all the traces
        rise (array_like) - spikes' rise time points
        basewin (array_like) - 2 scalars, begin and end of baseline window
                               to align different traces
        win (array_like) - 2 scalars, begin and end of the segment of the trace to plot
    return:
        fig (GraphicsWindow) - window object from the pyqtgraph with the plots
    '''
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    f = pg.GraphicsWindow()
    ax = f.addPlot()
    total = len(traces)
    if len(legends):
        ax.addLegend()
    for idx, trace in enumerate(traces):
        if len(basewin):
            trace -= trace[int(basewin[0] * sr):int(basewin[1] * sr)].mean();
        if len(win):
            trace = trace[int(win[0] * sr):int(win[1] * sr)]
        time = np.arange(len(trace)) / sr
        if len(legends):
            ax.plot(time, trace, pen = pg.mkPen(idx, total, width = 2), \
                    name = legends[idx])
        else:
            ax.plot(time, trace)
    if len(stim):
        yr = max([max(d) for d in traces]) - min([min(d) for d in traces])
        for st in stim:
            y1 = traces[0][int(st * sr)] - 0.1 * yr
            y2 = traces[0][int(st * sr)] + 0.1 * yr
            ax.plot([st, st], [y1, y2], pen = pg.mkPen('k')) 
    if len(rise):
        for t, trial_rise in enumerate(rise):
            y = [traces[t][int(d * sr)] for d in trial_rise]
            ax.plot(trial_rise, y, pen = None, symbol = 'o', symbolPen = 'k',\
                symbolBrush = None) 

    return f

def overlap(traces, sr, stim = [], legends = [], rise = [], basewin = [], win = []):
    '''
    Overlap multiple traces or segments of traces.
    parameters:
        traces (list) - list of traces to plot
        sr (float) - sampling rate
        stim (array_like) - stimulus time points
        legends (array_like) - legend strings for all the traces
        rise (array_like) - spikes' rise time points
        basewin (array_like) - 2 scalars, begin and end of baseline window
                               to align different traces
        win (array_like) - 2 scalars, begin and end of the segment of the trace to plot
    return:
        fig (GraphicsWindow) - window object from the pyqtgraph with the plots
    '''
    f = plt.figure()
    ax = f.add_subplot(111)
    total = len(traces)
    cm = plt.get_cmap('gist_rainbow')
    cl = [cm(1 * i / total) for i in range(total)]
    for idx, trace in enumerate(traces):
        if len(basewin):
            trace -= trace[int(basewin[0] * sr):int(basewin[1] * sr)].mean();
        if len(win):
            trace = trace[int(win[0] * sr):int(win[1] * sr)]
        time = np.arange(len(trace)) / sr
        if len(legends):
            ax.plot(time, trace, color = cl[idx], linewidth = 1, \
                    label = legends[idx])
        else:
            ax.plot(time, trace)
    if len(stim):
        yr = max([max(d) for d in traces]) - min([min(d) for d in traces])
        for st in stim:
            y1 = traces[0][int(st * sr)] - 0.1 * yr
            y2 = traces[0][int(st * sr)] + 0.1 * yr
    if len(rise):
        for t, trial_rise in enumerate(rise):
            y = [traces[t][int(d * sr)] for d in trial_rise]

    f.legend()
    return f
