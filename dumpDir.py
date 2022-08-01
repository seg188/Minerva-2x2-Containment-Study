import util
import argparse
import time
import h5py
import sys
import dumpTree

_default_output_dirname = 'dump/'


def transform_file_name(name):
	dir_and_name = name.split('/')
	name = dir_and_name[-1]
	name_and_ext = name.split('.')
	return name_and_ext[0]+'.h5'

def transform_file_names(file_name_list):
	return [transform_file_name(file) for file in file_name_list]

def main(input_dirname, output_dirname=_default_output_dirname):
	input_files = util.get_list_of_root_files(input_dirname)
	output_files = transform_file_names(input_files)
	for ifile, file in enumerate(input_files):
		start = time.time()
		print('dumping', file, 'to', output_files[ifile])
		dumpTree.dump(file, output_dirname + '/' +  output_files[ifile])
		print('time:', round(time.time()-start, 2) )
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dirname', '-i', type=str, help='''Directory containing edep-sim files converted to hdf5''')
	parser.add_argument('--output_dirname', '-o', type=str, default=_default_output_dirname, help='''Directory name to write output''')
	#parser.add_argument('--tag_muons', action='store_true', default=False, help='''Flag to do muon-tagging analysis. Default to false''')
	args = parser.parse_args()
	c = main(**vars(args))