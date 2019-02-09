# Analyze miniEPSCs (or probably miniEPSPs)

import functools
import operator
import numpy as np
import pyqtgraph as pg
import pandas as pd
import joblib
from scipy.optimize import curve_fit
import util
import plot
from process import SignalProc, DataPrep

class MinisAnalyzer(SignalProc, DataPrep):
	# Detect minis and analyze their properties
	# Fistly look at miniEPSCs

	def __init__(self, paramFile = ''):
		'''
		Read and load analysis parameters from a file.
		'''
		super().__init__()
		self.scale = 1e12  # scale up the data for better fitting
		self.params = {}
		if len(paramFile) != 0:
			self.loadParam(paramFile)
		# miniEPSC parameters after analysis
		self.miniRises = []  # valid minis' rise times
		self.miniPeaks = []  # valid mini's peak index
		self.miniAmps = []	# valid mini's peak amplitude
		self.miniDecayTaus = []  # valid mini's decay time constant

	def checkParam(self):
		'''
		Check the parameters to see if the paramters are correct.
		'''
		self.params["medianFilterWinSize"] = \
				int(self.params["medianFilterWinSize"])
		self.params["riseSlope"] = self.params["riseSlope"] * self.scale
		self.params["minAmp"] = self.params["minAmp"] * self.scale
		self.params["residual"] = self.params["residual"] * self.scale

	def analyze(self, verbose = 0):
		'''
		Detect the spikes of the minis and analyze them
		Criterions:
		  1. Rise time short enough
		  2. Amplitude large enough
		  3. Decay fit exponential curve with low enough residual
		  4. Decay time constant big enough
		'''
		# miniEPSC parameters after analysis
		self.miniRises = []  # valid minis' rise times
		self.miniPeaks = []  # valid mini's peak index
		self.miniAmps = []	# valid mini's peak amplitude
		self.miniDecayTaus = []  # valid mini's decay time constant
		x = self.x[int(self.sr * self.params['start']): \
				int(self.sr * self.params['end'])] * self.params['sign']
		# rig defect related single point noise
		x = self.thmedfilt(x, self.params['medianFilterWinSize'], \
				self.params['medianFilterThresh'])
		# scale
		x = x * self.scale
		# remove linear shifting baseline
		p = np.polyfit(np.arange(len(x)), x, 1)
		x = (x - np.polyval(p, np.arange(len(x))))
		# low pass filter
		fx = self.smooth(x, self.sr, self.params['lowBandWidth'])
		dfx = np.diff(fx) * self.sr
		peaks = (0 < dfx[0:-1]) & (dfx[1:] < 0)
		troughs = (dfx[0:-1] < 0) & (0 < dfx[1:])
		# points with local maximum slope, which is also larger than threshold
		rises = (dfx[0:-1] < self.params["riseSlope"]) & \
				(self.params["riseSlope"] < dfx[1:])
		'''
		rises = np.zeros(peaks.shape)
		rises = (dfx[0:-2] < dfx[1:-1]) & (dfx[2:] < dfx[1:-1]) & \
				(self.params['riseSlope'] < dfx[1:-1])
		'''
		# indices of either rises or peaks
		ptrInds = np.concatenate((np.nonzero(peaks | rises | troughs)[0], \
				[int(self.params['end'] * self.sr)]), axis = None)
		lastRise = -self.params["riseTime"] * self.sr  # last rise point index
		last2Rise = 0  # the rise point index before last rise point
		baseline = 0  # current baseline level
		peakStack = []	# peaks stacked to close to each other
		for i in range(len(ptrInds) - 1):
			if peaks[ptrInds[i]]:
				if ptrInds[i] - lastRise < self.params['riseTime'] * self.sr or \
						len(peakStack):
					if (len(peakStack) and ptrInds[i + 1] - peakStack[0] \
								< self.params["stackWin"] * self.sr):
						peakStack.append(ptrInds[i])
					else:
						if last2Rise < lastRise - \
							int(self.params['baseLineWin'] * self.sr):
							baseline = np.mean(x[lastRise - \
									int(self.params['baseLineWin'] * self.sr):\
									lastRise])
						amp = fx[ptrInds[i]] - baseline
						if self.params['minAmp'] < amp or len(peakStack):
							if not len(peakStack) and ptrInds[i + 1] - ptrInds[i] < \
									self.params["stackWin"] * self.sr and \
									i + 3 < len(ptrInds) and not rises[ptrInds[i + 2]]:
								peakStack.append(ptrInds[i])
							else:
								if len(peakStack):
									amp = np.max(fx[peakStack] - baseline)
									peakStack = []
								sample = x[lastRise:ptrInds[i + 1]]
								# exponential function to fit the decay
								fun = lambda x, t1, t2, a, b, c: \
										a * np.exp(-x / t1) - b * np.exp(-x / t2) + c
								# initial parameter values
								p0 = [self.params["offTauIni"], self.params["onTauIni"], \
										fx[lastRise] + amp - baseline, amp, baseline]
								# boundaries
								bounds = ([-np.inf, -np.inf, 0, 0, -np.inf], \
										[np.inf, np.inf, np.inf, np.inf, np.inf])
								try:
									popt, pcov = curve_fit(fun, np.arange(len(sample)), \
											sample, p0, bounds = bounds, \
											loss = "linear", max_nfev = 1e3 * len(sample))
									tau_rise = popt[1] / self.sr
									tau_decay = popt[0] / self.sr
									res = np.sqrt(np.sum((fun(np.arange(len(sample)), \
											*popt) - sample) ** 2))
									if verbose > 1:
										print("popt: ", popt)
										print("tau rise: ", tau_rise, "tau decay: ", \
												tau_decay, "res: ", res, "time:", \
												lastRise / self.sr)
										fm = pg.GraphicsWindow()
										ax = fm.addPlot(0, 0)
										plot.plot_trace_v(x[lastRise:ptrInds[i + 1]], \
												self.sr, ax = ax)
										plot.plot_trace_v(fx[lastRise:ptrInds[i + 1]], \
												self.sr, ax = ax, cl = 'g')
										plot.plot_trace_v(\
												fun(np.arange(len(sample)), *popt), \
												self.sr, ax = ax, cl = 'r')
										print("Continue (c) or step ([s])")
										if input() == 'c':
											verbose = 1
									if self.params['minTau'] < tau_decay \
											and res < self.params['residual']:
										'''
										self.miniPeaks.append(self.params['start'] + \
												ptrInds[i] / self.sr)
										self.miniRises.append(self.params["start"] + \
												lastRise / self.sr)
										'''
										self.miniPeaks.append(ptrInds[i] / self.sr)
										self.miniRises.append(lastRise / self.sr)
										self.miniAmps.append(amp / self.scale)
										self.miniDecayTaus.append(tau_decay)
								except RuntimeError as e:
									print("Fit Error")
									print(e)
								except ValueError as e:
									print("Initialization Error")
									print(e)
			elif rises[ptrInds[i]]:
				last2Rise = lastRise
				lastRise = ptrInds[i]
		if verbose > 0:
			fs = pg.GraphicsWindow()
			ax = [fs.addPlot(i, 0) for i in range(3)]
			plot.plot_trace_v(x, self.sr, ax = ax[0])
			plot.plot_trace_v(fx, self.sr, ax = ax[0], cl = 'g')
			plot.plot_trace_v(fx, self.sr, ax = ax[1], pcl = 'r', \
					points = np.nonzero(rises)[0] / self.sr)
			plot.plot_trace_v(fx, self.sr, ax = ax[1], pcl = None, \
					points = np.nonzero(peaks)[0] / self.sr)
			plot.plot_trace_v(fx, self.sr, points = self.miniRises, \
					ax = ax[1], pcl = 'b')
			ax[0].setXLink(ax[1])
			ax[0].setYLink(ax[1])
			plot.plot_trace_v(dfx, self.sr, ax = ax[2])
			return fs

	def getNoise(self):
		"""
		Analyze the noise level of the voltage clamp. Use variance
		of the entire trace and scale up to increase accuracy.
		"""
		x = self.x[int(self.sr * self.params['start']): \
				int(self.sr * self.params['end'])] * self.params['scale']
		self.noise = np.var(x)

	def getSpan(self):
		"""
		Smooth the trace and analyze the range of all the time point, for
		stability estimation.
		"""
		x = self.x[int(self.sr * self.params['start']): \
				int(self.sr * self.params['end'])]
		x = self.thmedfilt(x, self.params['medianFilterWinSize'], \
				self.params['medianFilterThresh'])
		x = x * self.params["scale"]
		fx = self.smooth(x, self.sr, self.params['lowBandWidth'])
		self.span = fx.max() - fx.min()

	def getProperties(self, prop_name):
		'''
		Calculate and return the analyzed properties' values
		Arguments:
			prop_name (String) - name of the property to get
		'''
		if prop_name == "frequency":
			freq = len(self.miniPeaks) / (self.params["end"] - self.params["start"])
			return freq
		elif prop_name == "amplitude":
			return self.miniAmps;
		elif prop_name == "decayTau":
			return self.miniDecayTaus;
		else:
			print("Error: Unknown property name.")

	def getMeanProperties(self, prop_name):
		'''
		Calculate the return the mean values of the analyzed properties
		Arguments:
			prop_name (String) - name of the properties to get
		'''
		if prop_name == "amplitude":
			return np.mean(self.miniAmps);
		elif prop_name == "decayTau":
			return np.mean(self.miniDecayTaus);
		else:
			print("Error: Unknown property name.")

	def stored(self, dp, c, t):
		return len(dp) and c in dp and len(dp[c]) and t in dp[c]

	def store(self, dp, c, t):
		if c not in dp:
			dp[c] = {}
		dp[c][t] = [self.miniRises, self.miniPeaks, \
			self.miniAmps, self.miniDecayTaus]

	def retrieve(self, dp, c, t):
		[self.miniRises, self.miniPeaks, \
			self.miniAmps, self.miniDecayTaus] = dp[c][t]

	def batchAnalyze(self, dataFolder, output, trialFile = "", trials = [], \
			dumpF = "", verbose = 1, ignoreMissing = False, minTrialN = 4):
		'''
		Analyze minis in a group of cells,
		arguments:
			dataFolder (string) - folder with the raw data.
			output (string) - data output file with cell indeices.
			trialFile (string) - file with selected trials, use trials if not provided.
			trials (array_like) - trial number to use for all cells.
			dumpF (string) - data file with a dictionary of previously caculated
				minis data, used to avoid redundant calculation.
			verbose (int) - 0, no std output, 1, output cell, 2 output trial.
			ingoreMissing (bool) - whether to ignore missing file.
			minTrialN (int) - minimum number of valid trials for a cell to be analyzed.
		''' 

		out = pd.read_csv(output, header = 0, index_col = 0)
		if len(dumpF):
			try:
				dp = joblib.load(dumpF)
			except (IOError, EOFError):
				dp = {}
		if len(trialFile):
			selectedTrials = util.readTrials(trialFile)
		for c in out.index:
			if verbose > 0:
				print("Cell: ", c)
			props = {"frequency": [], "amplitude": [], "decayTau": []}
			if len(trialFile):
				trials = selectedTrials[c]
			if len(trials) > minTrialN - 1:
				for t in trials:
					if verbose > 1:
						print("Trial: ", t)
					try:
						if "dp" in locals() and self.stored(dp, c, t):
							self.retrieve(dp, c, t)
						else:
							self.loadData(dataFolder, c, t)
							self.analyze()
						if "dp" in locals() and not self.stored(dp, c, t):
							self.store(dp, c, t)
						props["frequency"].append(self.getProperties("frequency"))
						props["amplitude"].append(self.getMeanProperties("amplitude"))
						props["decayTau"].append(self.getMeanProperties("decayTau"))
					except IOError:
						print("Cell ", c, "trial ", t, "not found.")
						if ignoreMissing:
							continue
						else:
							print("Continues? (Y/N)")
							i = input()
							if i == 'y' or i == 'Y':
								continue
							else:
								raise IOError
			for k, v in props.items():
				if len(v):
					out.loc[c, k] = np.mean(v)
		out.to_csv(output)
		if len(dumpF):
			joblib.dump(dp, dumpF)

	def noiseFilter(self, dataFolder, output, trialFile = "", trials = [], \
			cutoff = [-1., -1.]):
		"""
		Analyze minis recording trace noise level and exclude too noisy ones.
		Write kept trials to a text trial file.
			dataFolder (string) - folder with the raw data.
			output (string) - data output file with cell indeices.
			trialFile (string) - file with selected trials, use trials if not provided.
			trials (array_like) - trila number to use for all cells.
			cutoff (float) - noise and span cutoff threshold
		"""
		# calculate noises
		out = pd.read_csv(output, header = 0, index_col = 0)
		noises = {}
		spans = {}
		if len(trialFile):
			selectedTrials = util.readTrials(trialFile)
		for c in out.index:
			noises[c] = []
			spans[c] = []
			if len(trialFile):
				trials = selectedTrials[c]
			for t in trials:
				self.loadData(dataFolder, c, t)
				self.getNoise()
				self.getSpan()
				noises[c].append(self.noise)
				spans[c].append(self.span)
		# determine cutoff value if not provided
		cutoff = np.array(cutoff)
		confirm = False
		if any(cutoff < 0.):
			win = pg.GraphicsWindow()
			axN = win.addPlot()
			axS = win.addPlot()
			flatNoises = [item for sublist in noises.values() for item in sublist]
			flatSpans = [item for sublist in spans.values() for item in sublist]
		while any(cutoff < 0.) or not confirm:
			print("Number of bins: ")
			ins = input()
			try:
				nbins = int(ins)
			except ValueError: # when wrong value is given, use 10 bins
				nbins = 10
			y, x = np.histogram(flatNoises, nbins)
			axN.plot(x, y, stepMode = True, fillLevel = 0)
			axS.plot(flatNoises, flatSpans, pen = pg.mkPen(None), \
					symbol = 'o', symbplBrush = pg.mkBrush(None))
			axS.setLabel("bottom", "noise")
			axS.setLabel("left", "span")
			print("Cutoff thresholds: ")
			ins = input()
			try:
				cutoff = np.array(list(map(float, ins.split(","))))
			except ValueError: # when wrong value is given, use 10 bins
				pass
			if not any(cutoff < 0.):
				axS.plot([cutoff[0], 0], [0, cutoff[1]])
				print("Confirm? (y/n)")
				ins = input()
				confirm = (ins == "y" or ins == "Y")
			axN.clear()
			axS.clear()
		if "win" in locals():
			win.close()
		# write the kept trials based on cutoff threshold
		if len(trialFile) == 0:
			trialFile = "tmp.txt"
		with open(trialFile, "w") as f:
			for k, v in noises.items():
				s = "{:d}: ".format(k)
				if "selectedTrials" in locals():
					trials = selectedTrials[k]
				for i, n in enumerate(zip(noises[k], spans[k])):
					if np.sum(np.array(n) / cutoff) < 1:
						s += str(trials[i])
						s += ","
				s = s[:-1] + "\n"
				f.write(s)
