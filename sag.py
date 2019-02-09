#!/usr/local/python3

import re
import numpy as np
from process import DataPrep

class SagRatio(DataPrep):
	"""
	Calculate sag ratio from sag test data.
	"""
	def checkParam(self):
		required = set(['baseline_window', 'peak_window', 'steady_state_window', \
				"hyp_begin", "hyp_end", "I_step"])
		if set(self.params.keys()) != required:
			raise ValueError

	def getProperties(self, pName):
		"""
		Retrieve seal tests properties.
		"""
		if pName == "sagRatio":
			return self.r
		elif pName == "Rin":
			return self.Rin

	def analyze(self):
		'''
		Analyze membrane potential change to hyperpolarization current step and 
		calculate the sag ratio.  Estimate input resistance for neurons with 
		Ih current.
		parameters:
			hyp_begin (float) - time point of the beginning of hyperpolarization
			hyp_end (float) - time point of the end of hyperpolarizaion
			I_step (float) - amplitude of current step
			params (dictionay) - parameters for sag ratio calculation
		properties:
			r (float) - sag ratio
			Rin (float) - input resistance
		'''
		trace = self.x
		sr = self.sr
		b_win = params['baseline_window']  # baseline window size
		p_win = params['peak_window']  # peak value window size
		s_win = params['steady_state_window']  # steady state window size
		hyp_begin = self.params["hyp_begin"]
		hyp_end = self.params["hyp_end"]
		I_step = self.params["I_step"]
		baseline = np.mean(trace[int((hyp_begin - b_win) * sr): \
			int(hyp_begin * sr)])
		peak = np.mean(trace[int(np.argmin(trace) - p_win / 2 * sr): \
			int(np.argmin(trace) + p_win / 2 * sr)])
		steady_state = np.mean(trace[int((hyp_end - s_win) * sr): \
			int(hyp_end * sr)])

		self.r = (peak - steady_state) / (peak - baseline)
		self.Rin = (peak - baseline) / I_step
	
	def batchAnalyze(self, dataFolder, output, trialFile = "", trials = [], \ 
			ignoreMissing = False):
		"""
		Analyze a group of cells and selected traces to calculate the rheobases
		Arguments:
			dataFolder (string) - folder with the raw trace data.
			output (string) - csv file with the mean values for cells to analyze
			trialFile (string) - optional, text file with the target trials
				for each cell.
			trials (list) - optional, unified target trials.
			ignoreMissing (bool) - whether to ingore missing trials
		"""

        out = pd.read_csv(output, header = 0, index_col = 0)
        if len(trialFile):
            selectedTrials = util.readTrials(trialFile)
        for c in out.index:
            if verbose > 0:
                print("Cell: ", c)
            sagr = []
			rins = []
            if len(trialFile):
                trials = selectedTrials[c]
			for t in trials:
				if verbose > 1:
					print("Trial: ", t)
				try:
					self.loadData(dataFolder, c, t)
					self.analyze()
					sagr.append(self.r)
					rins.append(self.Rin)
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
			out.loc[c, "sagRatio"] = np.mean(sagr)
        out.to_csv(output)
