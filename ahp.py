#!/usr/local/python3

import numpy as np

'''
Analyze AHP and IAHP traces
'''

def average(traces, sr, basewin = []):
    '''
    ave_trace = average(traces, sr, basewin):
        Average raw traces of neurons after normalizing them to baseline 
    parameters:
        traces (array_like) - raw voltage or current traces
        sr (float) - sampling rate
        basewin (array_like) - two scalars, baseline time window, don't normalize if 
            it is empty
    return:
        ave_trace (array_like) - averaged trace
    '''

    ave_trace = np.zeros(len(traces[0]))
    for t in traces:
        if len(basewin):
            ave_trace += t - np.mean(t[int(basewin[0] * sr):int(basewin[1] * sr)])
        else:
            ave_trace += t
    ave_trace = ave_trace / len(traces)
    return ave_trace

def diff(trace1, trace2, sr, basewin = []):
    '''
    diff_trace = diff(trace1, trace2, sr, basewin = []):
        Calculate the difference of two traces, trace1 - trace2, after 
        normalizing them to baseline or not
    parameters:
        trace1 (array_like) - first trace
        trace2 (array_like) - second trace
        sr (float) - sampling rate
        basewin (array_like) - two scalars, baseline time window, don't normalize if 
            it is empty
    '''

    if len(basewin):
        diff_trace = (trace1 - np.mean(trace1[int(basewin[0] * sr):int(basewin[1] * sr)])) - \
            (trace2 - np.mean(trace2[int(basewin[0] * sr):int(basewin[1] * sr)]))
    else:
        diff_trace = trace1 - trace2
    return diff_trace

def amp(trace, sr, basewin, tp):
    '''
    calculate the amplitude of signal *trace* with sampling ratio *sr* at
    time point *tp* relative to baseline in *basewin*.
    Arguments:
        trace (array_like) - signal trace
        sr (double) - sampling rate
        basewin (array) - of two scalars, begin and end of baseline window
        tp (double) - time point at which to calculate the amplitude
    return:
        r (double) - relative amplitude
    '''

    r = trace[int(tp * sr)] - \
            np.mean(trace[int(basewin[0] * sr):trace[int(basewin[1]) * sr]])
    return r
