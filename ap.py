#!/usr/local/python3

import os
import re
import numpy as np
import pandas as pd
import util
import plot

def spike_detect(trace, sr, params, begin = 0, end = -1, plotting = False): 
    '''
    start_ind = spike_detect(trace, sr, params, begin = 0, end = -1, plotting = False) 
        Detect action potential spikes and return the spike rising time points.
        Find start time of spikes by finding point with slope over slope_th followed by a peak
        of relative amplitude above peak_th.  The peak is defined as the first point reversing 
        slope after the start point. 
    parameters:
        trace (array_like) - voltage trace
        sr (float) - sampling rate
        params (dictionary) - parameters
        begin (float) - begin of the time window to be analyzed
        end (float) - end of the time window to be analyzed, it represents the end of the 
            trace when it's -1
        plotting (boolean) - whether to plot the trace with starting point marked for
            inspection
    return:
        start_ind (array_like) - indices of spike starting points
    '''

    slope_th = params['spike_slope_threshold']
    peak_th = params['spike_peak_threshold']
    width_th = params['half_width_threshold']
    sign = params['sign']

    if sign < 0:
        trace = trace * sign
    trace_diff = np.diff(trace) * sr
    pstart = np.nonzero(trace_diff > slope_th)[0]  # possible start points
    reverse = np.nonzero(trace_diff < 0)[0] # possible peak points
    start_ind = []
    #print('start len = {:f}'.format(len(pstart)))
    #print('reverse len = {:f}'.format(len(reverse)))
    i = 0  # index in pstart
    j = 0  # index in reverse
    if end == -1:
        end = len(trace) / sr
    while i < len(pstart) and j < len(reverse) and pstart[i] < sr * end:
        if pstart[i] < sr * begin:
            i += 1
        elif pstart[i] < reverse[j]:
            #print('b: {0:f}, p: {1:f}'.format(trace[pstart[i]], trace[reverse[j]]))
            if peak_th < trace[reverse[j]] - trace[pstart[i]] and \
                reverse[j] - pstart[i] < width_th * sr:
                start_ind.append(pstart[i])
                while i < len(pstart) and pstart[i] < reverse[j]:
                    i += 1
            else:
                i += 1
        else:
            j += 1

    # plot trace with spike start points marked if needed
    #print(start_ind)
    start_ind = np.array(start_ind)  # transform to numpy array
    if plotting:
        w = plot.plot_trace_v(trace, sr, points = start_ind / sr)
        return start_ind, w
    else:
        return start_ind

def firing_rate(folder, cells, trials = [], out = 'firing_rate_data.txt', \
    param_file = 'param_spike_detect'):
    '''
    data = firing_rate(folder, cells, trials = [], data_out = 'firing_rate_data.txt')
        Measure firing rate data and output in two forms, one with firing rate of the same
        current input averaged and the other with raw firing rate for each trial. output a  
        dictionary with each element having data for one cell, the key is the cell index.
        data = {cell_idx: [[[stims], 
                            [averaged firing rate]],
                           [[trial indices],
                            [stims],
                            [firing rate]]]}
    parameters:
        folder (String) - directory to the folder of the trace data files
        cells (array_like) - indices of cells to analyze
        trials (array_like) - trial numbers, if not provided analyze all the trials in the
            folder
        out (String) - output data file directory
        param_file (String) - spike detection parameter file
    return:
        data (dictionary) - firing rate data
    '''
    
    data = {} 
    params = get_params(param_file)
    for cell in cells:
        print('Cell: ', cell)
        ctrials = trials[:]
        _stims = []
        _rates = []
        if not len(ctrials):
            data_files = os.listdir(folder)
            for data_file in data_files:
                matched = re.match('Cell' + \
                    '_{:04d}_0*([1-9][0-9]*)\.ibw'.format(int(cell)), data_file)
                if matched:
                    ctrials.append(int(matched.group(1)))
        for trial in ctrials:
            file_dir = folder + os.sep + util.gen_name(cell, trial)
            trace, sr, stim_i = util.load_wave(file_dir)
            _stims.append(stim_i[2])
            _rates.append(len(spike_detect(trace, sr, params, stim_i[1], \
                stim_i[0] + stim_i[1])))
        raw = [ctrials, _stims, _rates]
        _stims = np.array(_stims)
        _rates = np.array(_rates)
        stims = np.unique(_stims)
        ave = np.zeros(len(stims))
        for i, current in enumerate(stims):
            ave[i] = sum(_rates[np.nonzero(_stims == current)[0]]) / \
                len(np.nonzero(_stims == current)[0])
        ave_data = [stims.tolist(), ave.tolist()]
        data[cell] = [ave_data, raw]
    util.write_dict(out, data)
    return data

def get_params(param_file_name):
    '''
    params = get_params(param_file_name)
        Read ramp test parameters in a file
    Parameters:
        param_file_name (string) - directory to the file with seal test parameters
    Returns:
        params (dictionary) - all parameters as values in a dictionary with the
            parameter names as keys
    '''

    try:
        # Read peak detection paramter files
        param_file = open(param_file_name, 'r')
        params = {'spike_slope_threshold': None, \
            'spike_peak_threshold': None, \
            'half_width_threshold': None, \
            'sign': None}
        for line in param_file:
            matchObj = re.match(r'(\S*)\s*=\s*(.*)', line)
            params[matchObj.group(1)] = float(matchObj.group(2))
        param_file.close()
        if any([i is None for k, i in params.items()]):
            raise ValueError

    except IOError:
        print('File reading error')
    except KeyError:
        print('Wrong parameter name')
        param_file.close()
    except ValueError:
        print('Missing parameter')
    else:
        return params

class Aps:
    '''   
    Template class for trial choose in action potential analysis.
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
        Attributes:
            trial_data (array_like) - cell_num * trial_num * 3 3D array, each trial has
                has the trial index, firing rate and stimulation current data
        '''
        cell_num = len(self.data.index)
        self.trial_data = np.zeros((cell_num, trial_num, 3))  # chosen trials 
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

class Ahp(Aps):
    '''
    Analyze afterhyperpolarizations with current clamp spike trains.
    '''
    def calc_mahp(self, lat_win, spk_win, param_file = 'param_spike_detect', \
        output_file = 'medium_ahps.txt'):
        '''
        Calculate medium ahps in chosen trials after spikes in the spk_win. mAHPs are 
        defined as hyperpolarization peak value between two spikes relative to the spiking
        threshold of the prior spike, excluding the fast AHPs range right after the 
        prior spike. Choose spikes to avoid Ih current and sAHP.
        Parameters:
            lat_win (array_like) - 2 scalars, mininum latency for mAHP, window for fAHP, 
                and minimum interspike interval to include ahp.
            spk_win (array_like) - 2 scalars, begin and end of the window of spikes
            param_file (String) - directory to parameter files for spike detection
            output_file (String) - dirctory to output file of medium ahp data of all
                spikes
        Returns:
            mean_ahps (array_like) - 1D array of mean ahps for each cell
        '''
        cell_num = self.trial_data.shape[0] 
        trial_num = self.trial_data.shape[1] 
        spike_num = spk_win[1] - spk_win[0]
        ahps = np.zeros((cell_num, trial_num, spike_num))
        spike_params = get_params(param_file)
        for c in range(cell_num):
            for t in range(trial_num):
                cind = self.data['No'][c]
                tind = self.trial_data[c][t][0]
                if tind != 0:
                    trace, sr, stim = util.load_wave(self.folder + \
                        util.gen_name(cind, tind))
                    starts = spike_detect(trace, sr, spike_params)  # start of all spikes
                    for s in range(spike_num):
                        if spk_win[0] + s < len(starts): 
                            # only calculate if it is not the last spike 
                            if sr * lat_win[1] < \
                                starts[spk_win[0] + s + 1] - starts [spk_win[0] + s]:
                                # and the ISI is larger that the minumum threshold
                                ahps[c][t][s] = trace[starts[spk_win[0] + s]] - \
                                    np.min(trace[(starts[spk_win[0] + s] + \
                                    int(sr * lat_win[0])):starts[spk_win[0] + s + 1]])
                            else:
                                print('Irregular trial: cell', cind, ',trial', tind)
        util.write_arrays(output_file, ['mahp', ahps])
        mean_ahps = np.zeros(cell_num)
        for i in range(cell_num):
            mean_ahps[i] = np.mean(ahps[i][np.nonzero(ahps[i])])
        return mean_ahps

    def calc_sahp(self, baseline_win, lat, win, output_file = 'slow_ahps.txt'):
        '''
        Calculate the slow ahp in chosen trials. In current clamp AP trains, slow AHP is 
        defined as the hyperpolarization at lat time point after the end of the current
        step relative to the baseline before the current step. Try to avoid the Ih current.
        Parameters:
            baseline_win (float) - window size of baseline ahead of the current step
            lat (float) - latency of sAHP measurement point after the current step
            win (float) - window size of the sAHP measurement
        Return:
            mean_ahps (array_like) - mean slow ahps for each cell
        '''
        cell_num = self.trial_data.shape[0] 
        trial_num = self.trial_data.shape[1] 
        ahps = np.zeros((cell_num, trial_num))
        for c in range(cell_num):
            for t in range(trial_num):
                cind = self.data['No'][c]
                tind = self.trial_data[c][t][0]
                if tind != 0:
                    trace, sr, stim = util.load_wave(self.folder + \
                        util.gen_name(cind, tind))
                    baseline = np.mean(trace[int((stim[0] - baseline_win) * sr):\
                        int(stim[0] * sr)])
                    ahps[c][t] = baseline - np.mean(trace[int((stim[0] + stim[1] + lat) * sr):\
                        int((stim[0] + stim[1] + lat + win) * sr)])
        util.write_arrays(output_file, ['sahp', ahps])
        mean_ahps = np.zeros(cell_num)
        for i in range(cell_num):
            mean_ahps[i] = np.mean(ahps[i][np.nonzero(ahps[i])])
        return mean_ahps

class Ap_prop(Aps):
    '''
    Analyze the action potential properties in current clamp spike trains.
    '''
    def get_spike_time(self, param_file = 'param_spike_detect'):
        '''
        Calculate the spike start time for all the chosen traces.
        Parameters:
            param_file (string) - directory to spike detection parameter file
        '''
        spike_params = get_params(param_file)
        cell_num = self.trial_data.shape[0] 
        trial_num = self.trial_data.shape[1] 
        self.starts = []  # start times of all spikes in all trials
        for c in range(cell_num):
            _starts = []
            for t in range(trial_num):
                cind = self.data['No'][c]
                tind = self.trial_data[c][t][0]
                if tind != 0:
                    trace, sr, stim = util.load_wave(self.folder + \
                        util.gen_name(cind, tind))
                    _starts.append(spike_detect(trace, sr, spike_params))
            self.starts.append(_starts)

    def calc_slope(self, spk_win, output_file = 'ap_slope.txt'):
        '''
        ave_slopes = calc_slope(self, spk_win, output_file = 'ap_slope.txt'):
        Caclulate largest slope of the rising of an action potential
        Parameters
            spk_win (array_like) - 2 scalars, begin and end of the window of spikes
        Return
            ave_slopes (array_like) - average slopes for each cell
        '''
        cell_num = self.trial_data.shape[0] 
        trial_num = self.trial_data.shape[1] 
        spike_num = spk_win[1] - spk_win[0]
        slopes = np.zeros((cell_num, trial_num, spike_num))
        for c in range(cell_num):
            for t in range(trial_num):
                cind = self.data['No'][c]
                tind = self.trial_data[c][t][0]
                if tind != 0:
                    starts = self.starts[c][t]
                    trace, sr, stim = util.load_wave(self.folder + \
                        util.gen_name(cind, tind))
                    for s in range(spk_win[0], spk_win[1]):
                        if s < len(starts) - 1:
                            peak_point = np.argmax(trace[starts[s]:starts[s + 1]])
                            slopes[c, t, s - spk_win[0]] = np.max(np.diff(trace[starts[s]: \
                                starts[s] + peak_point])) * sr
                        elif s == len(starts) - 1:
                            peak_point = np.argmax(trace[starts[s]:\
                                int((stim[0] + stim[1]) * sr)])
                            slopes[c, t, s - spk_win[0]] = np.max(np.diff(trace[starts[s]: \
                                starts[s] + peak_point])) * sr
        util.write_arrays(output_file, ['slopes', slopes])
        ave_slopes = np.zeros(cell_num)
        for i in range(cell_num):
            ave_slopes[i] = np.mean(slopes[i][np.nonzero(slopes[i])])
        return ave_slopes
        # return slopes
    
    def calc_v(self, win, output_file = 'vhold.txt'):
        '''
        ave_vhold = calc_v(self, output_file = 'vhold.txt'):
            Calculate holding membrane potential before the spike train.
        parameter:
            win (float) - window size before the depolarization to average.
            output_file (string) - output file directory.
        return:
            ave_vhold (array_like) - averaged holding potential for each neuron
        '''
        cell_num = self.trial_data.shape[0] 
        trial_num = self.trial_data.shape[1] 
        vhold = np.zeros((cell_num, trial_num))
        for c in range(cell_num):
            for t in range(trial_num):
                cind = self.data['No'][c]
                tind = self.trial_data[c][t][0]
                if tind != 0:
                    trace, sr, stim = util.load_wave(self.folder + \
                        util.gen_name(cind, tind))
                    vhold[c, t] = np.mean(trace[int((stim[0] - win) * sr):\
                        int(stim[0] * sr)])
        util.write_arrays(output_file, ['vhold', vhold])
        ave_vhold = np.zeros(cell_num)
        for i in range(cell_num):
            ave_vhold[i] = np.mean(vhold[i][np.nonzero(vhold[i])])
        return ave_vhold
        
def FI_slope(data_file, cells):
    '''
    slope = FI_slope(data_file, cells, stims = []):
        Calculate firing slope using averaged firing rate data.
    parameters:
        data_file (String) - directory to firing rate data file
        cells (array_like) - indices of cells to be analyzed
    return:
        slope (array_like) - FI curve slope of all the cells
    '''
    data = util.read_dict(data_file, 'int')
    slope = []
    for cell in cells:
        stims = np.array(data[cell][0][0])
        rates = np.array(data[cell][0][1])
        firing_ind = np.nonzero(rates)[0]  # indices of point with firing rate above zero
        p = np.polyfit(stims[firing_ind], rates[firing_ind], 1)
        slope.append(p[0])
    return slope

