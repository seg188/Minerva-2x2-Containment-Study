#!/usr/bin/env python3
import util
import acceptance
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time
import h5py
import sys
import scipy as sp
import timeit
import os
from acceptance import datatype as data_type

_defualt_plot_dir = 'plots'\


def initHDF5File(output_file, name, dtype=data_type):
	with h5py.File(output_file, 'a') as f:
	   	f.create_dataset(name, (0,), dtype=dtype, maxshape=(None,))

def updateHDF5File(output_file, name, data):
	if len(data):
		with h5py.File(output_file, 'a') as f:
			ndata = len(f[name])
			f[name].resize((ndata+len(data),))
			f[name][ndata:] = data

def main(dirname, plot_dir=_defualt_plot_dir, data_file_prefix='data_'):

	files = util.get_list_of_h5_files(dirname)
	data = dict()
	plotdir = plot_dir + '/'
	prefix = plotdir

	for ifile, file in enumerate(files):
		filename = prefix+data_file_prefix+file.split('/')[-1]
		print(filename)
		if not os.path.exists(filename): 
			initHDF5File(filename, 'data')
		start = time.time()
		full_start = time.time()
		print(file)
		if True:
		#try:
			datalog = h5py.File(file, 'r')
			all_trajectories = datalog['trajectories']
			all_segments = datalog['segments']
			all_vertices = datalog['vertices']
			all_stack = datalog['particle_stack']
			this_result = acceptance.main(all_segments, all_trajectories, all_stack, all_vertices)
			print('full time:', time.time()-full_start)
			updateHDF5File(filename, 'data', this_result)
		#except:
		#	print('error processing file')
		#	continue

	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dirname', '-i', type=str, help='''Directory containing edep-sim files converted to hdf5''')
	parser.add_argument('--plot_dir', type=str, default=_defualt_plot_dir, help='''Directory name to write plots''')
	#parser.add_argument('--tag_muons', action='store_true', default=False, help='''Flag to do muon-tagging analysis. Default to false''')
	args = parser.parse_args()
	c = main(**vars(args))