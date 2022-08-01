import os
import h5py
import datetime

import argparse

def Merge(dict_list):
	d = dict()
	for dd in dict_list:
		for key in dd:
			if not key in d: d[key] = []
			d[key]+=dd[key]
	return d

def get_list_of_h5_files(file_or_dir_name):
	if os.path.isfile(file_or_dir_name):
		return [file_or_dir_name]
	else:
		base = file_or_dir_name
		print(os.listdir(base))
		return sorted([base + "/" + name for name in os.listdir(base) if (name[-3:] == ".h5")])

def get_list_of_root_files(file_or_dir_name):
	if os.path.isfile(file_or_dir_name):
		return [file_or_dir_name]
	else:
		base = file_or_dir_name
		print(os.listdir(base))
		return sorted([base + "/" + name for name in os.listdir(base) if (name[-5:] == ".root")])
	
