#!/usr/local/python3

import re
import numpy as np
import json
import csv
from pydoc import locate
from igor import binarywave

paramF = "/Users/lch/Google Drive/prg/data_analysis/"

def gen_name(cell, trial, format_file_dir = paramF + 'file_name_format.txt'):
    '''
    file_name = gen_name(cell, trial, format_file_dir = 'file_name_format.txt')
        Generate recorded file name for trial of cell with the format in format_file
    parameters:
        cell (int) - cell index
        trial (int) - trial index
        format_file_dir (String) - file with format parmeters
    return:
        file_name (String) = formated file name
    '''
    params = get_name_params(format_file_dir)
    cell = int(cell)
    trial = int(trial)
    file_name = (params['prefix'] + params['link'] + '{0:0' + params['pad'] + 'd}' + \
        params['link'] + '{1:0' + params['pad'] + 'd}' + params['suffix']).format(cell, \
        trial)
    return file_name

def get_name_params(format_file_dir = paramF + 'file_name_format.txt'):
    '''
    params = get_name_params(format_file_dir = 'file_name_format.txt')
        Read file name format parameters
    parameters:
        format_file_dir (String) - file with the parameters
    return:
        params (dictionary) - parameters
    '''
    try:
        format_file = open(format_file_dir, 'r')
        params = {'prefix': None, \
            'pad': None, \
            'link': None, \
            'suffix': None}
        for line in format_file:
            matchObj = re.match(r'(\S*)\s*=\s*(.*)', line)
            params[matchObj.group(1)] = matchObj.group(2)
        format_file.close()

    except IOError:
        print('File reading error')
    except KeyError:
        print('Wrong parameter name')
        param_file.close()
    except ValueError:
        print('Missing parameter')
    else:
        return params

def load_wave(file_name):
    '''
    trace, sr, stim = load_wave(file_name)
        Load trace from an igor data file, as well as sampleing rate and stimulation
        amplitude.
    parameters:
        file_name (string) - directory of the igor data file
    return:
        trace (array_like) - data trace in the file, return as numpy array
        sr (float) - sampling rate
        stim (array_like)- stimulation step properties in an array, including start time,
            duration and amplitude
    '''
        
    try:
        sr, stim_amp, stim_dur, stim_start = 10000, 0, 0, 0
        data = binarywave.load(file_name)
        trace = data['wave']['wData']
        # Search for sampling rate
        searched = re.search(r'XDelta\(s\):(.*?);', data['wave']['note'].decode())
        if(searched != None):
            sr = 1 / float(searched.group(1))
        # Search for stimulation amplitude
        searched = re.search(r';Stim Amp.:(.+?);', data['wave']['note'].decode())
        if(searched != None):
            stim_amp = float(searched.group(1))
        # Search for stimulation duration
        searched = re.search(r';Width:(.+?);', data['wave']['note'].decode())
        if(searched != None):
            stim_dur = float(searched.group(1))
        # Search for stimulation strat
        searched = re.search(r';StepStart\(s\):(.+?);', data['wave']['note'].decode())
        if(searched != None):
            stim_start = float(searched.group(1))
        return (trace, sr, [stim_start, stim_dur, stim_amp])
    except IOError:
        print('Igor wave file (' + file_name + ') reading error')
        raise IOError

def input_col(end = ''):
    '''
    arr = input_col(end = ''):
        Take a column vector of data input from keyboard and return as a numpy array
    parameters:
        end (string) - string showing the end of input
    return:
        arr (array like) - a vector of input data
    '''

    try:
        arr = []
        data = input()
        while(data != end):
            arr.append([float(data)])
            data = input()
        return np.array(arr)
    except ValueError:
        print('Only numeric input accepted')

def write_arrays(file_dir, *args):
    '''
    write_arrays(file_dir, *args):
        Write arrays in a file. Each array is stored as name in one line and the array in
        the next line.
    parameters:
        file_dir (string) - directory of output file
        *args - array name (string) and array pairs
    '''
    fo = open(file_dir, 'w') 
    for pair in args:
        fo.write(pair[0])
        fo.write('\n')
        fo.write(json.dumps(pair[1].tolist()))
        fo.write('\n')
    fo.close()

def write_dict(file_dir, d):
    fo = open(file_dir, 'w') 
    for k in d:
        fo.write(str(k))
        fo.write('\n')
        fo.write(json.dumps(d[k]))
        fo.write('\n')
    fo.close()

def read_arrays(file_dir, *args):
    '''
    write_arrays(file_dir, *args):
        Read arrays from a file. Each array is stored as name in one line and the array in
        the next line.
    parameters:
        file_dir (string) - directory of input file
        *args - array names (string)
    return:
        data (tuple) - of arrays
    '''
    fi = open(file_dir, 'r') 
    data = [None] * len(args)
    line = fi.readline().rstrip('\n')
    while line:
        if line in args:
            data[args.index(line)] = np.array(json.loads(fi.readline().rstrip('\n')))
        line = fi.readline().rstrip('\n')
    fi.close()
    if len(data) == 1:
        return data[0]
    else:
        return tuple(data)

def read_dict(file_dir, key_type):
    var_type = locate(key_type)
    fi = open(file_dir, 'r')
    data = {}
    line = fi.readline().rstrip('\n')
    while line:
        k = var_type(line)
        data[k] = np.array(json.loads(fi.readline().rstrip('\n')))
        line = fi.readline().rstrip('\n')
    fi.close()
    return data

def read_csv(file_dir):
    '''
    data = read_csv(file_dir):
        Read from csv file a numpy 2D array, empty cells in the file stored as 0
    parameters:
        file_dir (string) - directory of data file
    return:
        data (array_like) - 2D array of the data in the file
    '''
    with open(file_dir, newline = '') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            if 'data' in locals():
                data = np.concatenate((data, [[float(d) if d else 0  for d in row]]), axis = 0)
            else:
                data = np.array([[float(d) if d else 0  for d in row]])
    return data

def write_csv(file_dir, data):
    '''
    write_csv(file_dir, data):
        Write a 2D array matrix in a csv file
    parameters:
        file_dir (string) - directory of output file
        data (array_like) - output data
    '''
    datafile = open(file_dir, 'w')
    writer = csv.writer(datafile)
    for row in data:
        writer.writerow(row)
    return 0

def intensity2file(folder, cells, intensities, trials, intensity_data, group_size):
    '''
    file_dir = intensity2file(folder, cells, intensities, trials, intensity_data, group_size):
        Generate directory to files with the data collected under provided conditions.
    parameters:
        folder (string) - directory of folder with the data files
        cells (array_like) - indices of cells of interest
        intensitities (array_like) - intensities (conditions) of interests
        trials (array_like) - indices of trials under each condition, starting from 0
        intensity_data (string) - intensity data file with all the intensities (conditions) 
            of each cell. It's a csv file with the 1st column having cell indices, the 2nd
            column having cell types and the 3rd column having first indice of all response 
            trial and the following columns having cell specific intensities (conditions)
        group_size (int) - number of trials under each intensity (condition)
    return:
        file_dir (array_like) - of data file direcotries
    '''
    file_dir = []
    idata = read_csv(intensity_data)
    for c in cells:
        if len(np.nonzero(idata[:, 0] == c)[0]):
            row = idata[np.nonzero(idata[:, 0] == c)[0][0], :]
            first = int(row[2])
            for i in intensities:
                if len(np.nonzero(row[3:] == i)[0]):
                    t0 = (np.nonzero(row[3:] == i)[0][0]) * group_size + first
                    for t in trials:
                        file_dir += [folder + gen_name(c, t0 + t)]
    return file_dir

def intensity_trials(intensity_sel):
    '''
    cells, intensities = intensity_trials(intensity_sel):
        Read cell indices and intensities to be process from a csv file.
    parameters:
        intensity_sel (String) - directory to file with the selected cell and intensity data
    return:
        cells (array_like) - indices of cells
        intensities (list) - list of numpy arrays, intensities for each cell
    '''

    data = read_csv(intensity_sel)
    cells = data[:, 0].astype(int)
    intensities = [d[np.nonzero(d)][1:] for d in data]
    return (cells, intensities)

def permutationTest(x, y, N = 10000):
    '''
    permutate samples in x and y and test if the mean is different.
    parameters:
        x, y
        N (int) - number of permutaions
    return:
        p (float) - p value of significance test
    '''
    delta = x.mean() - y.mean()
    estimates = np.zeros(N)
    pooled = np.hstack((x, y)) 
    xlen = len(x)
    ylen = len(y)
    for i in range(N):
        np.random.shuffle(pooled)
        estimates[i] = pooled[:xlen].mean() - pooled[-ylen:].mean()
    if delta < 0:
        p = len(np.where(estimates < delta)[0]) / N
    elif 0 < delta:
        p = len(np.where(delta < estimates)[0]) / N
    else:
        p = 1
    return p

def readTrials(trialFile):
    '''
    Read trial nunmbers selected from a file.
    arguments:
        trialFile (string) - directory to the file.
    returns:
        trials (dictionary) - trial numbers for cells with cell number as index
    '''

    trials = {}
    with open(trialFile) as f:
        for line in f:
            c, t = line.split(':')
            t = t.rstrip().lstrip()
            if not len(t):
                trials[int(c)] = []
            else:
                trials[int(c)] = list(map(int, t.split(',')))
    return trials

