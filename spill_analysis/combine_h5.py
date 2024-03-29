import util
import argparse
import time
import h5py
import sys
import timeit
import os
import numpy as np

_defualt_plot_dir = 'plots'\

data_type = np.dtype( \
		[("eventID","u4"),("contained","i4"),("W","f8"), ("lt_10cm","i4"), ("n_vtx_particles","i4"), 
		 ("fs_e", "f8"), ("fs_theta", "f8"), ("v_x", "f8"), ("v_y", "f8"), ("v_z", "f8"), 
         ("visible_energy","f8"), ("n_pions", "i4"), ("n_pi0", "i4"), ("n_protons", "i4"), ("nu_i", "i4"), ##add neutral pion count
         ("nu_i_energy", "f8"), ("q", "f8"), ("q2", "f8"), ("p1_p", "f8"), ("p1_theta", "f8"), ("p1_phi", "f8"), ("p2_p", "f8"), ("p2_theta", "f8"), ("p2_phi", "f8"),  
         ("pions_contained", "i4"), ("pr1_p", "f8"), ("pr1_theta", "f8"), ("cs", "i4"), ("containment_threshold", "f8"), ("containment_threshold_2x2", "f8"), 
         ("fs", "i4"), ("fs_energy_sum", "f8"), ("fs_contained", "i4"), ("all_contained", "i4"), ("all_but_fs_contained_2x2_only", "i4"), ("all_contained_2x2_only", "i4")]) 


def initHDF5File(output_file, name, dtype=data_type):
	with h5py.File(output_file, 'a') as f:
	   	f.create_dataset(name, (0,), dtype=dtype, maxshape=(None,))

def updateHDF5File(output_file, name, data):
	if len(data):
		with h5py.File(output_file, 'a') as f:
			ndata = len(f[name])
			f[name].resize((ndata+len(data),))
			f[name][ndata:] = data

def main(dirname, name='data', combined_file_name='combined.h5'):

	files = util.get_list_of_h5_files(dirname)
	data = dict()
	combined_file=dirname+'/'+combined_file_name

	initHDF5File(combined_file, name)

	for ifile, file in enumerate(files):
		try:
			print(file)
			with h5py.File(file, 'r') as f:
				data = f[name]
				test_data = data[0:10]
				updateHDF5File(combined_file, name, data)
		except:
			print('cant use:', file)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dirname', '-i', type=str, help='''Directory containing edep-sim files converted to hdf5''')
	#parser.add_argument('--tag_muons', action='store_true', default=False, help='''Flag to do muon-tagging analysis. Default to false''')
	args = parser.parse_args()
	c = main(**vars(args))