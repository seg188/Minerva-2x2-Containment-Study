import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt
import util



_defualt_plot_dir = 'plots'

def main(filename, plot_dir=_defualt_plot_dir):
	prefix = plot_dir + '/'
	files = util.get_list_of_h5_files(filename)
	pion_counts = []
	nu_energies = []

	for file in files:
		f = h5py.File(file)
		particle_stack = f['particle_stack']
		eventIDs = particle_stack['eventID']
		for eventID in set(eventIDs):
			pion_count = 0
			event_mask= particle_stack['eventID']==eventID
			stack = particle_stack['pdgId', 'status', 'p4'][event_mask]
			#plot n pions in events
			nu_energy = 0
			for ipart in range(len(stack)):
				if (stack['status'][ipart]==0) and (stack['pdgId'][ipart] in [14, -14, 12, -12]): 
					nu_energy = stack['p4'][ipart][3]

				if stack['pdgId'][ipart] in [111, 211, -211] and stack['status'][ipart]==1: pion_count += 1

			pion_counts.append(pion_count)
			nu_energies.append(nu_energy)

	pion_bins=[0, 1, 3, 20]
	all_energies = []
	for inp in range(len(pion_bins)-1):
		energies = []
		for iEvent in range(len(pion_counts)):
			if pion_counts[iEvent] >= pion_bins[inp] and pion_counts[iEvent] < pion_bins[inp+1]:
				energies.append(nu_energies[iEvent])
		all_energies.append(energies)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Pion Production vs. Nu Energy')
	for iplt in range(len(all_energies)):
		plt.hist(all_energies[iplt], bins=40, density=True, label=str(round(pion_bins[iplt])) + '-' + str(round(pion_bins[iplt+1])) + ' pions', range=(0, 20.0), alpha=0.4 )
	ax.legend()
	ax.set_xlabel('Nu Energy [GeV]')
	fig.savefig(prefix+'npions.png')
	
	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Beam Neutrino Energy')
	plt.hist(nu_energies, bins=40, density=True, range=(0.1, 25.0))
	ax.set_xlabel('Nu Energy [GeV]')
	fig.savefig(prefix+'beam_energy.png')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', '-i', type=str, help=''' edep-sim file converted to hdf5''')
	parser.add_argument('--plot_dir', type=str, default=_defualt_plot_dir, help='''Directory name to write plots''')
	args = parser.parse_args()
	c = main(**vars(args))