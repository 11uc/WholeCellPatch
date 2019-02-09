#!/usr/local/bin/python3

import math
import re
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pyqtgraph as pg
import joblib
from igor import binarywave
from process import DataPrep, SignalProc
import plot
import util

class SealTest(DataPrep, SignalProc):
	"""
	Analyze seal tests (short current or voltage steps) to calculate 
	access resistance, input resistance and membrane capacitance of
	of neurons.
	"""
	def __init__(self, paramFile):
		DataPrep.__init__(self, paramFile)

	def checkParam(self):
		required = set(['baseline_start', 'baseline_end', 'steady_state_start', \
            'steady_state_end', 'seal_test_start', 'dV', 'dI', \
			"scaleV", "scaleI", "minTau"])
		if set(self.params.keys()) != required:
			raise ValueError

	def seal_test_v(self, verbose = False):
		'''
		Analysze the seal test part of the trace and calculate the basic properties
		Parameters:
			plot (string) - name of the figure of the analysis process, default is empty 
				string and no figure will be saved
		Properties:
			Rs (float) - access resistance
			Ri (float) - input resistance
			C (float) - membrane compacitance
		'''
		t_baseline1 =  self.params['baseline_start']
		t_baseline2 = self.params['baseline_end']
		t_steady1 = self.params['steady_state_start']
		t_steady2 = self.params['steady_state_end']
		t_0 = self.params['seal_test_start']
		dV = self.params['dV']
		scale = self.params["scaleV"]  # scale up for better fitting
		minTau = self.params["minTau"]  # minimum tau accepted
		trace = self.x * scale
		sr = self.sr

		I_b = np.mean(trace[int(t_baseline1 * sr):int(t_baseline2 * sr)])
			# baseline current before seal test
		I_s = np.mean(trace[int(t_steady1 * sr):int(t_steady2 * sr)])
			# steady state current after the voltage step
		std_Is = np.std(trace[int(t_steady1 * sr):int(t_steady2 * sr)])
			# standard deviation of current at the steady state
			# used to define boundary for steady state current in fitting

		fit_fun = lambda t, _I0, _tau, _Is, _t0: _Is + (_I0 - _Is) * \
			np.exp(-(t - _t0) / _tau)  # exponential function to fit
			# I0 - current spike right after the voltage step
			# tau - decay time constant
			# Is - steady state current of the decay
			# t0 - start of the decay
		mf = None  # medium fitting threshold
		trapped = True
		while trapped:
			if mf is not None:
				trace = self.thmedfilt(trace, 5, mf)
			# fit the exponential decay after the voltage step
			ft1 = np.argmax(trace[int(t_0 * sr):int(t_steady1 * sr)] * np.sign(dV)) \
				+ int(t_0 * sr)
				# index of peak value after the voltage step start of curve fitting
			ft2 = int(t_steady1 * sr)
				# index of the start of steady state, end of curve fitting
			fit_time = np.arange(ft1, ft2) / sr  # time array of exponential fit
			fit_trace = np.array(trace[ft1:ft2]) # trace within the fit time period
			g_I0 = trace[ft1]  # initial guess of I0
			g_tau_ind = np.argwhere(np.sign(dV) * (trace[ft1:ft2] - I_s) < \
				np.sign(dV) * (g_I0 - I_s) / np.e)
			g_tau = g_tau_ind[0] / sr  # initial guess of tau
			p_0 = [g_I0, g_tau, I_s, t_0]  # initial guess
			bnd = ([-np.inf, 0, I_s - 3 * std_Is, t_0], \
					[I_s, t_steady1 - t_0, I_s + 3 * std_Is, t_steady1])  # bounds
			try:
				popt, pcov = curve_fit(fit_fun, fit_time, fit_trace, p0 = p_0)
				fit_I0, tau, fit_Is, fit_t0 = popt
				I_0 = fit_fun(ft1 / sr, fit_I0, tau, fit_Is, fit_t0)
				# Use the current of the fitted curve at the peak time as I0
				if verbose or tau < minTau:
					verbose = True
					print('I0 = ', I_0)
					print('tau = ', tau)
					print('Ib = ', I_b)
					print('fit_Is = ', fit_Is)
					print('t0 = ', fit_t0)

					# pt1 = int(t_0 * sr)
					# pt2 = int(t_steady1 * sr)
					pt1 = ft1
					pt2 = ft2
					plot_time = np.arange(pt1, pt2) / sr
					if mf is None:
						plot_trace = np.array(trace[pt1:pt2])
					else:
						plot_trace = self.thmedfilt(np.array(trace[pt1:pt2]), 5, mf)
					w = pg.GraphicsWindow()
					ax = w.addPlot()
					ax.plot(plot_time, plot_trace)
					ax.plot(plot_time, \
							[fit_fun(i, fit_I0, tau, fit_Is, fit_t0) \
							for i in plot_time], pen = pg.mkPen('r'))
					print("Median filter threshold?")
					ins = input()
					mf = float(ins)
					del(w)
					if mf <= 0:
						self.R_s, self.R_i, self.C = 0, 0, 0
						break
				else:
					trapped = False
				self.R_s = dV / (I_0 - I_b) * scale
				self.R_i = dV / (I_s - I_b) * scale - self.R_s
				self.C = tau * (self.R_i + self.R_s) / self.R_i / self.R_s
			except TypeError:
				self.R_s, self.R_i, self.C = 0, 0, 0
				break
			except ValueError:  # when input is not a number
				self.R_s = dV / (I_0 - I_b) * scale
				self.R_i = dV / (I_s - I_b) * scale - self.R_s
				self.C = tau * (self.R_i + self.R_s) / self.R_i / self.R_s
				break
		if verbose:
			print('Rs = {0:.3e}, Ri = {1:.3e}, C = {2:.3e}'.format(self.R_s, \
					self.R_i, self.C))

	def seal_test_i(self, verbose = False):
		'''
		Seal test in current clamp, calculate access resistance, 
		input resistance, capacitance.
		Parameters: 
			trace (array_like) - signal trace
			sr (float) - sampling rate
			params (dictionary) - parameters used for seal test analysis
			comp (boolean) - whether or not the Rs has been compensated
		Properties:
			Rs (float) - Access resistance
			Rin (float) - Input resistance
			C (float) - Membrane capacitance
		'''

		t_baseline1 =  self.params['baseline_start']
		t_baseline2 = self.params['baseline_end']
		t_steady1 = self.params['steady_state_start']
		t_steady2 = self.params['steady_state_end']
		t_0 = self.params['seal_test_start']
		dI = self.params['dI']
		scale = self.params["scaleI"]
		trace = self.x * scale
		sr = self.sr
		minTau = self.params["minTau"]

		V_b = np.mean(trace[int(t_baseline1 * sr):int(t_baseline2 * sr)])
			# baseline current before seal test
		V_s = np.mean(trace[int(t_steady1 * sr):int(t_steady2 * sr)])
			# steady state current after the voltage step
		std_Vs = np.std(trace[int(t_steady1 * sr):int(t_steady2 * sr)])
			# standard deviation of current at the steady state
			# used to define boundary for steady state current in fitting

		trapped = True
		mf = None
		while trapped:
			if mf is not None:
				trace = self.thmedfilt(trace, 5, mf)
			# fit the exponential decay after the voltage step
			ind1 = int(t_0 * sr)
				# index of start of curve fit, assume charging of pipette 
				# capacitance takes no time
			ind2 = int(t_steady1 * sr)
				# index of the start of steady state, end of curve fitting
			fit_time = np.arange(ind1, ind2) / sr  # time array of exponential fit
			fit_trace = np.array(trace[ind1:ind2])  # trace within the fit time period
			fit_fun = lambda t, _V0, _tau, _Vs, _t0: _Vs + (_V0 - _Vs) * \
				np.exp(-(t - _t0) / _tau)  # exponential function to fit
				# V0 - volrage after the voltage jump due to the access resistance
				# tau - decay time constant
				# Vs - steady state potential of the decay
				# t0 - start of the decay
			g_V0 = trace[ind1]  # initial guess of V0
			g_tau_ind = np.argwhere(np.sign(dI) * (trace[ind1:ind2] - V_s) > \
				np.sign(dI) * (g_V0 - V_s) / np.e)
			g_tau = g_tau_ind[0][0] / sr  # initial guess of tau
			p_0 = [g_V0, g_tau, V_s, t_0]  # initial guess
			bnd = ([-np.inf, 0, V_s - 3 * std_Vs, t_0], \
				[V_s, t_steady1 - t_0, V_s + 3 * std_Vs, t_steady1])  # bounds
			try:
				popt, pcov = curve_fit(fit_fun, fit_time, fit_trace, p0 = p_0)
				fit_V0, tau, fit_Vs, fit_t0 = popt
				V_0 = fit_fun(ind1 / sr, fit_V0, tau, fit_Vs, fit_t0)
				if verbose or tau < minTau:
					verbose = True
					print('Vs = ', V_s)
					print('V0 = ', V_0)
					print('tau = ', tau)
					print('Vb = ', V_b)
					print('fit_Vs = ', fit_Vs)
					print('t0 = ', fit_t0)
					# ind1 = int(t_0 * sr)
					# ind2 = int(t_steady1 * sr)
					plot_time = np.arange(ind1, ind2) / sr
					if mf is None:
						plot_trace = np.array(trace[ind1:ind2])
					else:
						plot_trace = self.thmedfilt(np.array(trace[ind1:ind2]), 5, mf)
					w = pg.GraphicsWindow()
					ax = w.addPlot()
					ax.plot(plot_time, plot_trace)
					ax.plot(plot_time, \
							[fit_fun(i, fit_V0, tau, fit_Vs, fit_t0) \
							for i in plot_time], pen = pg.mkPen('r'))
					ins = input()
					mf = float(ins)
					del(w)
					if mf <= 0:  # input 0 or negative values to drop the results
						self.R_s, self.R_i, self.C = 0, 0, 0
						break
				else:
					trapped = False
				self.R_s = (V_0 - V_b) / dI / scale
				self.R_i = (V_s - V_b) / dI / scale - self.R_s
				self.C = tau / self.R_i 
			except TypeError:
				self.R_s, self.R_i, self.C = 0, 0, 0
				break
			except ValueError:  # when input is not a number, accept the result
				self.R_s = (V_0 - V_b) / dI / scale
				self.R_i = (V_s - V_b) / dI / scale - self.R_s
				self.C = tau / self.R_i 
				break
		if verbose:
			print('Rs = {0:.3e}, Ri = {1:.3e}, C = {2:.3e}'.format(self.R_s, \
					self.R_i, self.C))

	def getProperties(self, pName):
		"""
		Retrieve seal tests properties.
		"""
		if pName == "Rin":
			return self.R_i
		elif pName == "Rs":
			return self.R_s
		elif pName == "C":
			return self.C
		elif pName == "tau":
			return self.R_i * self.C
	
	def batchAnalyze(self, dataFolder, output, reportF = "",  \
			dumpF = "", vclamp = True, verbose = 1):
		"""
		Batch mini analysis.
		Arguments:
			dataFolder (string) - raw trace data folder.
			output (string) - csv file with the indices of neurons to analyze.
			reportF (string) - output text file with seal tests results for all trials.
			dumpF (string) - data file to store analyzed data to for future use.
			vclamp (bool) - whether the data is from voltage clamp or currrent clamp.
			verbose (int) - verbose levels
				0: no cli output
				1: print cell numbers
				2: print cell, trial number, plot fitting result
		"""
		if dataFolder[-1] != os.sep:
			dataFolder += os.sep

		name_params = util.get_name_params()
		data_files = os.listdir(dataFolder)
		df = pd.read_csv(output, index_col = 0)
		if len(dumpF):
			try:
				dp = joblib.load(dumpF)
			except (EOFError, IOError):
				dp = {}
		if len(reportF):
			rp = open(reportF, "w")
		for c in df.index:
			if verbose:
				print("Cell: {:d}".format(c))
			rss = []
			ris = []
			cs = []
			rp.write('\n' + name_params['prefix'] + ('_{0:0'+ name_params['pad'] + \
				'd}:').format(c) + '\n')
			for data_file in data_files:
				matched = re.match(name_params['prefix'] + name_params['link'] + \
						'{:04d}'.format(c) + name_params['link'] + \
						'0*([1-9][0-9]*)' + name_params['suffix'] , data_file)
				if matched:
					t = int(matched.group(1))
					if verbose > 1:
						print("Trial: {:d}".format(t))
					if "dp" in locals() and self.stored(dp, c, t):
						self.retrieve(dp, c, t)
					else:
						self.loadData(dataFolder, c, t)
						if vclamp:
							self.seal_test_v(verbose > 1)
						else:
							self.seal_test_i(verbose > 1)
					if "dp" in locals() and not self.stored(dp, c, t):
						self.store(dp, c, t)
					if self.getProperties("Rin") > 0:
						rss.append(self.getProperties("Rs"))
						ris.append(self.getProperties("Rin"))
						cs.append(self.getProperties("C"))
						rp.write('\t{0:d} Rs:{1:3.2f}; Ri:{2:3.2f}; C:{3:4.3f}\n'.
								format(t, rss[-1] / 1e6, ris[-1] / 1e6, cs[-1] * 1e9))
			if len(rss):
				mrs = np.mean(rss)
				mri = np.mean(ris)
				mc = np.mean(cs)
				mtau = np.mean(np.array(ris) * np.array(cs))
			else:
				mrs, mri, mc, mtau = np.nan, np.nan, np.nan, np.nan
			df.loc[c, "Rin"] = mri / 1e6
			# Update Rs and tau only in voltage clamp because I usually have
			# Rs compensation on in current clamp and the seal test results 
			# are not accurate.
			if vclamp:  
				df.loc[c, "Rs"] = mrs / 1e6
				df.loc[c, "mTau"] = mtau
			rp.write('\tMean:\n')
			rp.write('\tRs:{0:3.2f}; Ri:{1:3.2f}; C:{2:4.3f}\n'.format(\
				mrs / 1e6, mri / 1e6, mc * 1e9))
		if len(dumpF):
			joblib.dump(dp, dumpF)
		if len(reportF):
			rp.close()
		df.to_csv(output)
	
	def store(self, dp, c, t):
		if c not in dp:
			dp[c] = {}
		dp[c][t] = [self.R_s, self.R_i, self.C]

	def stored(self, dp, c, t):
		return len(dp) and c in dp and len(dp[c]) and t in dp[c]

	def retrieve(self, dp, c, t):
		[self.R_s, self.R_i, self.C] = dp[c][t]
		
	def filterRs(self, dumpF, maxRs, output, trialOut, filterBy = "trial"):
		'''
		For voltage clamp seal tests, remove trials with access resistance above maxRs,
		write the remaining trial numbers to file output, and return mean Rs and Rin for
		each cell of the remaining trials.
		arguments:
			dumpF (string) - stored batch seal test results.
			maxRs (float) - upper threshold of Rs.
			output (String) - directory to summary output file, mean values
			trialOut (string) - output file for selected trials
			filterBy (string) - filter criterion
				"trial": remove trials with to high Rs
				"cells": remove cells with one trial that is above threshold
		'''
		selected = {}
		dp = joblib.load(dumpF)
		df = pd.read_csv(output, index_col = 0)
		for c, vc in dp.items():
			srs, sri, stau = 0, 0, 0
			selected[c] = []
			for t, vt in vc.items(): 
				if vt[0] < maxRs and 0 < vt[0]:
					selected[c].append(t)
					srs += vt[0]
					sri += vt[1]
					stau += vt[2] * vt[1]
				elif filterBy == "cell":
					srs = np.nan
					sri = np.nan
					stau = np.nan
			if len(selected[c]):
				df.loc[c, "Rs"] = srs / len(selected[c]) / 1e6
				df.loc[c, "Rin"] = sri / len(selected[c]) / 1e6
				df.loc[c, "mTau"] = stau / len(selected[c])
			else:
				df.loc[c, "Rs"] = np.nan
				df.loc[c, "Rin"] = np.nan
				df.loc[c, "mTau"] = np.nan
		df.to_csv(output)
		# write file
		with open(trialOut, 'w') as f:
			for k in sorted(selected):
				s = "{:d}: ".format(k)
				for t in selected[k]:
					s += str(t)
					s += ","
				s = s[:-1] + "\n"
				f.write(s)
		
def ICnoise(trace, sr, begin, end):
    '''
    noise r ICnoise(trace, sr, begin, end)
        Calculate std of sample points in trace within a certein time range to represent noise.
    Parameters:
        trace (array_like) - signal data
        sr (float) - sampling rate
        begin (float) -  start point of the time range
        end (float) -  end point of the time range
    Return:
        noise (float) - std as noise level
    '''
    return np.std(trace[int(begin * sr):int(end * sr)])
