#!/usr/local/python3

import statistics
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time
import plot
import util

class SignalProc:
    '''
    utility functions for signal processing in slice physiology data
    '''
    def __init__(self):
        pass

    def thmedfilt(self, x, wsize, thresh):
        '''
        median filter with threshold, only sample points with value larger 
        than a threshold are filtered.
        paramters:
            x (array_like) - input signal
            wsize (int) - median filter window size
            thresh (float) - threshold for filter
        return:
            z (array) - filterd signal
        '''

        y = signal.medfilt(x, wsize)
        z = np.where(thresh < abs(y - x), y, x)
        return z

    def smooth(self, x, sr, band):
        '''
        y = smooth(x, band)
            lowpass filter the signal with Butterworth filter to smooth it
        paramters:
            x (array_like) - signal trace
            band (float) - criticle freqency for lowpass filter
            sr (float) - sampling rate
        return:
            y (array_like) - smoothed trace
        '''

        b, a = signal.iirfilter(4, band / sr * 2, btype = 'lowpass', ftype = 'butter')
        y = signal.filtfilt(b, a, x)
        return y

class DataPrep:
    """
    Loading data and parameters for analysis.
    """
    def __init__(self, paramFile):
        self.loadParam(paramFile)
        pass

    def loadParam(self, paramFile):
        '''
        Load analysis parameters from a file
        '''
        self.params = {}
        with open(paramFile) as f:
            for line in f:
                name, val = line.split('=')
                self.params[name.rstrip()] = float(val)
        self.checkParam()
    
    def checkParam(self):
        '''
        Check the parameters to see if the paramters are correct. Implemented
        in derivative classes.
        '''
        pass

    def loadData(self, dataFolder, cellN, trialN):
        '''
        Load trace data from a file
        Arguments:
            dataFolder (String) - data file folder
            cellN (int) - cell number
            trialN (int) - trial number
        '''
        dataDir = dataFolder + '/' + util.gen_name(cellN, trialN)
        self.x, self.sr, stim = util.load_wave(dataDir)


def thmedfilt(x, wsize, thresh):
    '''
    median filter with threshold, only sample points with value larger 
    than a threshold are filtered.
    paramters:
        x (array_like) - input signal
        wsize (int) - median filter window size
        thresh (float) - threshold for filter
    return:
        z (array) - filterd signal
    '''

    y = signal.medfilt(x, wsize)
    z = np.where(thresh < abs(y - x), y, x)
    return z

def smooth(x, sr, band):
    '''
    y = smooth(x, band)
        lowpass filter the signal with Butterworth filter to smooth it
    paramters:
        x (array_like) - signal trace
        band (float) - criticle freqency for lowpass filter
        sr (float) - sampling rate
    return:
        y (array_like) - smoothed trace
    '''

    b, a = signal.iirfilter(4, band / sr * 2, btype = 'lowpass', ftype = 'butter')
    y = signal.filtfilt(b, a, x)
    return y


def resp_analysis(trace, sr, stim, slopethresh = 1e-8, sign = 1, slopewin = 1e-3, \
    lowpassband = 600, medfiltthresh = 0, endthresh = 0.04, risethresh = [0.001, 0.01], \
    durthresh = 1e-3, ampthresh = 10e-12, basewin = 5e-3, plotting = False):
    '''
    lat, amp, area, slp, f = resp_analysis(trace, sr, stim, sign = 1):
        Analyse responses in trace with to stimulation at,
        return the latencies and amplitudes of responses.
    parameters:
        trace (array_like) - signal trace with responses
        sr (float) - signal rate
        stim (list of float) - time points of stimulation
        slopethresh (float) - threshold of the slope to determine the start of the response
        sign (1 or -1) - direction of the responses
        slopewin (float) - window after the start of the response to measure the rising slope
        lowpassband (float) -threshold for lowpass band filter for response detection
        medfiltthresh (float) - threshold for median filter to remove single point noise,
            not applied if the value if 0
        endthresh (float) - threshold beyond which an endpoint is considered abnormal
        risethresh (list) - of 2 scalar, window for normal rise point
        durthresh (float) - threshold duration between rise and end below which is considered
            abnormal
        ampthresh (float) - amplitude threshold for rise point detection
        basewin (float) - window size of baseline trace
        plotting (boolean) - whether to return a matplotlib figure of of the trace with
            defined points shown.
    return:
        lat (float) - response latency
        amp (float) - response amplitude
        area (float) - integration over response period relative to start value
        slp (float) - slope of the rising of the signal
    '''

    # smooth the signal to reduce noise
    if medfiltthresh:
        ft = thmedfilt(trace, 5, medfiltthresh)
    else:
        ft = trace
    st = smooth(ft, sr, lowpassband)

    #lat, amp, area = np.empty(0), np.empty(0), np.empty(0)
    lat, amp, area, slp = [], [], [], []
    if plotting:
        points = []  # indices of key points of a peak
    
    diffst = np.diff(st) * sr # first derivative of the signal
    diffst_th = diffst * sign > slopethresh
    print('min diff: {:e}'.format(min(diffst[10000::])))
    # analyze response to each stimulation
    for stim_t in stim:
        stim_ind = int(stim_t * sr) # index of stimulation start
        if np.count_nonzero(diffst_th[stim_ind::]):
            rise_pre = int(np.nonzero(diffst_th[stim_ind + int(risethresh[0] * sr)::])[0][0] \
                + stim_ind + risethresh[0] * sr)
            #print('rise_pre ', rise_pre)
            # index of the point where the peak rise, derivative cross the threshold
            # start from stimulation time + a certain threshold value
            baseline = np.mean(ft[rise_pre - int(basewin * sr):rise_pre])
            #print('rise_pre value', ft[rise_pre])
            #print('base_pre', rise_pre - int(basewin * sr))
            #print('baseline ', baseline)
            #print('rise_end', stim_ind + risethresh[1] * sr)
            aboveth = (ft[rise_pre:stim_ind + int(sr * risethresh[1])] \
                - baseline) * sign > ampthresh
            rising = (ft[rise_pre + 1:stim_ind + int(sr * risethresh[1]) + 1] - \
                ft[rise_pre:stim_ind + int(sr * risethresh[1])]) * sign > 0
            post = np.nonzero(rising * aboveth)[0]
            if len(post):
                #print('in')
                rise_post = int(post[0]) + rise_pre
                #print('rise_post ', rise_post)
                #print('rise_post value', ft[rise_post])
                rise_start  = np.nonzero((ft[rise_pre + 1:rise_post + 1] - \
                    ft[rise_pre:rise_post]) * sign > 0)[0]
                if len(rise_start):
                    rise_ind = int(rise_start[-1]) + rise_pre
                else:
                    rise_ind = rise_pre
                #print('rise_ind', rise_ind)
            else:
                rise_ind = -1
        else:
            rise_ind = -1  # label -1 if no rise above threshold found
        if(rise_ind == -1 or (rise_ind - stim_ind) / sr > risethresh[1]):
            # if no peak found or latency too large, manual interference
            discarded, rise_ind  = assign_point(ft, sr, st, \
                [stim_ind, stim_ind + (risethresh[1] + endthresh) * sr], \
                'rise point', rise_ind)
            if discarded:
                lat.append(-1)
                amp.append(-1)
                area.append(-1)
                slp.append(0)
                continue
        bl = np.mean(st[stim_ind:rise_ind])
        ends = np.nonzero((sign * (st[rise_ind:-1:] - bl) < 0) * \
            (sign * diffst[rise_ind::] < 0))
        # index of point where the signal return to baseline
        # first time lower than value at the rise point
        if len(ends[0]):
            end_ind = int(ends[0][0] + rise_ind)
        else:
            end_ind = len(st) - 2
        if endthresh < (end_ind - rise_ind) / sr or (end_ind - rise_ind) / sr < durthresh:
            # if responding time is too large
            discarded, end_ind  = assign_point(ft, sr, st, \
                [stim_ind, end_ind + 1], 'end point', end_ind)
            if discarded:
                lat.append(-1)
                amp.append(-1)
                area.append(-1)
                slp.append(0)
                continue
        print('rise_ind', rise_ind)
        print('end_ind', end_ind)
        peak_ind = int(np.argmax(sign * st[rise_ind:end_ind]) + rise_ind) # peak position

        if sign * (st[peak_ind] - st[rise_ind]) < ampthresh:
            lat.append(-1)
            amp.append(-1)
            area.append(-1)
            slp.append(0)
        else:
            lat.append((rise_ind - stim_ind) / sr)
            amp.append(sign * (st[peak_ind] - st[rise_ind]))
            area.append(sum(ft[rise_ind:end_ind] - st[rise_ind]) / sr)
            linf = lambda x, a, b: a * x + b
            xdata = np.arange(rise_ind, rise_ind + int(slopewin * sr)) / sr
            ydata = ft[rise_ind:rise_ind + int(slopewin * sr)]
            popt, pcov = optimize.curve_fit(linf, xdata, ydata)
            slp.append(popt[0])
        if plotting:
            print('stim: {0:d}; rise: {1:d}; end: {2:d}, peak: {3:d}'.format(stim_ind, \
                rise_ind, end_ind, peak_ind))
            points.extend([stim_ind, rise_ind, end_ind, peak_ind])

    if plotting:
        f = plot.plot_trace(trace, sr, st, points)
        return lat, amp, area, slp, f
    else:
        return lat, amp, area, slp



def assign_point(trace, sr, st, win, point_name, point):
    '''
    discarded, apoint = assign_point(trace, sr, st, point_name, point)
        When the value of a time point the the peak is unusual, remind user to 
        assign it manually or discard this peak
    '''

    apoint = point
    if point < 0:
        print(point_name + ' is not found.')
        f = plot.plot_trace_v(trace, sr, st, win = [d / sr for d in win])
    else:
        print(point_name + ' ({:.3f}) is abnormal.'.format(point / sr))
        f = plot.plot_trace_v(trace, sr, st, [point / sr], win = [d / sr for d in win])
    f.show()

    while True:
        print('Do you want to keep (k), discard (d) or manually assign (a) the value?')
        dec = input('--> ')
        if dec == 'k':
            discarded = False
            break
        elif dec == 'd':
            discarded = True
            break
        elif dec == 'a':
            confirmed = 'n'
            while confirmed != 'y':
                try:
                    assigned = float(input('New x value: '))
                    if assigned < win[0] / sr or win[1] / sr < assigned:
                        raise ValueError
                    else:
                        apoint = int(assigned * sr)
                        f1 = plot.plot_trace_v(trace, 1, st, [apoint], win = win)
                        f1.show()
                        confirmed = input('Confirmed?(y/n): ')
                        #plt.close(f1)
                        del(f1)
                except ValueError:
                    print('Invalid value')
            discarded = False
            break
    #plt.close(f)
    del(f)
    return (discarded, apoint)

def gen_stim(start, freq, num):
    '''
    stim = gen_stim(start, freq, num)
        generate a list of stimulation times of a stimulation train
    parameters:
        start - time of first stimulation
        freq - stimulation frequency
        num - num of stimulation pulses
    retrun:
        stim - a list of time point of each stimulation pulses
    '''
    return np.arange(start, start + num / freq, 1 / freq)

def dyssync_resp_analysis(trace, sr, stim, win, slopethresh = 1e-8, sign = 1, \
    lowpassband = 600, medfiltthresh = 0):

    # Medium filter to remove single point noise and smooth to remove backgroud noise
    if medfiltthresh:
        ft = thmedfilt(trace, 5, medfiltthresh)
    else:
        ft = trace
    st = smooth(ft, sr, lowpassband)
    dst = np.diff(st)

    for stim_i in stim:
        left = int(stim_i * sr) 
        right = int((stim_i + win) * sr)
        begins = np.nonzero(slope * sign < st[left:right] * sign)
        peaks = np.nonzero((st[left - 1:right - 1] * sign < 0) * \
            0 < (st[left + 1:right + 1] * sign)) + left
        
