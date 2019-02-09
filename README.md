## Introduction

This is a collection of Python scripts used for basic analyze whole-cell
patch-clamp recording results. Applied to raw data collected using Igor,
i.e. igor binary files (\*.ibw) files.

### Functions
  * Analyze seal tests (in voltage clamp or current clamp) to calculate
    (*seal_test.py*)
  * Detect spontaneous miniature post synaptic currents (miniPSCs) 
    recorded in voltage clamp and calculate their amplitudes/frequencies.
	(*minis.py*)
  * Analyze voltage clamp response to hyperpolarization and calculate
    sag ratio. (*sag.py*)
  * Analyze current clamp firing rate and action potential properties 
    responding to current steps. (*ap.py*, *ahp.py*)

### General Usage

There are several class objects for each of the functions mentioned 
above. Without a GUI, I usually write a specific script for each 
set of experiments. Parameters for signal analysis and experimental
protocol parameters, such as stimulation parameters and recording 
time lines are read from text files, which are specified when 
declaring those objects. Then raw data folder and output file need
to be provided for the analysis. Usually data is output to csv files
ready for further statistical analysis.
