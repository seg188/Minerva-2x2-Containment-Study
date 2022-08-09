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

def plot(filename, plotdir, fieldname='data'):

	f = h5py.File(filename, 'r')
	
	print('opening')
	data = f['data'][0:595000]
	print('got W')

	e_min = -0.5
	e_max = 15000
	n_ebins = 100

	q_min = -0.5
	q_max = 5000
	n_qbins = 100
	prefix = plotdir + '/'

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Containment vs. Nu Energy')
	vals, edges = np.histogram(data['nu_i_energy'], weights=np.array(data['all_contained']).astype(int),bins=n_ebins, range=(e_min, e_max))
	norms, edges = np.histogram(data['nu_i_energy'], bins=n_ebins, range=(e_min, e_max))
	mdpnts = edges[:-1] + np.diff(edges)/2
	normalized = np.divide(vals,norms)
	print(sum(vals), sum(norms))
	print(sum(normalized[np.logical_not(np.isnan(normalized))]))
	ax.plot(mdpnts, normalized)
	ax.set_xlabel('Nu Energy [MeV]')
	ax.set_ylabel('Containment Fraction')
	fig.savefig(prefix+'containment_vs_enu.png')
	plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Containment vs. Energy Transfer')
	vals, edges = np.histogram(data['q'], weights=np.array(data['all_contained']).astype(int), bins=n_qbins, range=(q_min, q_max))
	norms, edges = np.histogram(data['q'], bins=n_qbins, range=(q_min, q_max))
	mdpnts = edges[:-1] + np.diff(edges)/2
	normalized = np.divide(vals,norms)
	print(sum(vals), sum(norms))
	print(sum(normalized[np.logical_not(np.isnan(normalized))]))
	ax.plot(mdpnts, normalized)
	ax.set_xlabel('q [MeV]')
	ax.set_ylabel('Containment Fraction')
	fig.savefig(prefix + 'containment_vs_q.png')
	plt.close(fig)

	mask = np.logical_or(data['q'] > q_max, data['q'] < 0)
	print(len(data['q']))
	print(len(data['q'][mask]))
	print(len(data['q'][np.logical_not(mask)]))



##################################################################
	#2d histogram of truth energy vs active edeps
	fig = plt.figure()
	ax = fig.add_subplot()
	ehist2d, eedges2d = np.histogramdd([np.array(data['visible_energy'])/1000., np.array(data['fs_energy_sum'])], bins=n_ebins, range=( (e_min, e_max), (e_min, e_max)  ))

	#normalize vertically
	scaled_ehist2d = []
	for energy_bin_data in np.transpose(ehist2d): 
		s = sum(energy_bin_data)
		if s == 0:
			scaled_ehist2d.append(np.array([0]*len(energy_bin_data)))
			continue
		scaled_ehist2d.append(energy_bin_data/s)
		
	im = plt.imshow(np.transpose(scaled_ehist2d), interpolation='nearest', extent=[min(eedges2d[0]), max(eedges2d[0]), min(eedges2d[1]), max(eedges2d[1])],  aspect='auto',origin='lower')
	ax.set_xlabel('Truth Energy [MeV]')
	ax.set_ylabel('Total Active Edeps [MeV]')
	fig.suptitle('Energy Resolution -- Full 2x2 + Minerva ')
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label('log10(n events)', rotation=-90)
	fig.savefig(prefix + 'active_edeps_2d.png')
	plt.close(fig)





	e_min = 0
	e_max = 6000
	e_bins = 150
	
	pion_counts = data['n_pions']
	nu_energies = data['q']

	pion_bins=[0, 1, 2, 3, 4, 20]
	all_energies = []
	for inp in range(len(pion_bins)-1):
		energies = []
		for iEvent in range(len(pion_counts)):
			if pion_counts[iEvent] >= pion_bins[inp] and pion_counts[iEvent] < pion_bins[inp+1]:
				energies.append(nu_energies[iEvent])
		all_energies.append(energies)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Pion Production vs. Energy Transfer -- NC, RHC')
	labels = []
	for iplt in range(len(all_energies)):
		_label = str(pion_bins[iplt]) + ' pions' if iplt < len(all_energies)-1 else  str(pion_bins[iplt]) + '+ pions'
		labels.append(_label)
	plt.hist(all_energies, bins=int(e_bins/4), stacked=True, label=labels, range=(e_min, e_max), alpha=0.4 )
	ax.legend()
	ax.set_xlabel('Energy Transfer [MeV]')
	fig.savefig(prefix+'npions.png')
	plt.close(fig)


	#################
	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('W')
	w_mask = np.array(data['W']) >= 0
	w_mask = np.logical_and(w_mask, np.array(data['fs'])==13)
	w_data = np.array(data['W'])[w_mask]
	plt.hist(w_data, range=(e_min, e_max), bins=e_bins)
#	ax.legend()
	ax.set_xlabel('W [MeV]')
	fig.savefig(prefix+'W.png')
	plt.close(fig)

	#################

	pion_bins=[0, 1, 2, 3, 20]
	all_energies = []
	pion_counts = np.array(data['n_pions'])[w_mask]
	for inp in range(len(pion_bins)-1):
		energies = []
		for iEvent in range(len(pion_counts)):
			if pion_counts[iEvent] >= pion_bins[inp] and pion_counts[iEvent] < pion_bins[inp+1]:
				energies.append(w_data[iEvent])
		all_energies.append(energies)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Pion Production vs. W -- CC')
	labels = []
	for iplt in range(len(all_energies)):
		_label = 'cc-' + str(pion_bins[iplt]) + 'pi' if iplt < len(all_energies)-1 else  'cc-' + str(pion_bins[iplt]) + '+pi'
		labels.append(_label)
	plt.hist(all_energies, bins=e_bins, stacked=False, histtype='step', label=labels, range=(e_min, e_max))
	plt.hist(w_data, range=(e_min, e_max), bins=e_bins, alpha=0.3, label='cc-inclusive')
	ax.legend()
	ax.set_xlabel('W [MeV]')
	fig.savefig(prefix+'W_n_pions.png')
	plt.close(fig)

	#2d histogram of truth energy vs active edeps
	w_min, w_max = 500, 4500
	w_nbins = 70
	q2_min, q2_max = 0, 4e6
	q2_nbins = 70
	fig = plt.figure()
	ax = fig.add_subplot()
	qwhist2d, qwedges2d = np.histogramdd([np.array(data['W']), np.array(data['q2'])], weights=np.array(data['all_contained'].astype(int)), bins=(int(w_nbins/2), int(q2_nbins/2)), range=( (w_min, w_max), (q2_min, q2_max) ))
	norm_qwhist2d, qwedges2d = np.histogramdd([np.array(data['W']), np.array(data['q2'])], bins=(int(w_nbins/2), int(q2_nbins/2)), range=( (w_min, w_max), (q2_min, q2_max) ))		
	scaled_qwhist2d = np.divide(qwhist2d, norm_qwhist2d)
	im = plt.imshow(np.transpose(scaled_qwhist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
	ax.set_xlabel('W [MeV]')
	ax.set_ylabel('q^2 [MeV^2]')
	fig.suptitle('Containment -- W vs. q^2')
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label('contained fraction', rotation=-90)
	fig.savefig(prefix + 'w_q2_containment.png')
	plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot()
	norm_qwhist2d, qwedges2d = np.histogramdd([np.array(data['W']), np.array(data['q2'])], bins=(int(w_nbins*1.5), int(q2_nbins*1.5) ), range=( (w_min, w_max), (q2_min, q2_max) ))		
	im = plt.imshow(np.transpose(norm_qwhist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
	ax.set_xlabel('W [MeV]')
	ax.set_ylabel('q^2 [MeV^2]')
	fig.suptitle('W vs. q^2')
	fig.savefig(prefix + 'w_q2_distribution.png')
	plt.close(fig)


	fig = plt.figure()
	ax = fig.add_subplot()
	ax.hist(data['q2'], bins=100)
	fig.suptitle('q^2')
	ax.set_xlabel('q^2 [MeV]')
	fig.savefig(prefix+'q2.png')



	f.close()

	#plot 2d containment vs W and q



	return

def main(filename, plotdir):
	plot(filename, plotdir)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', '-i', type=str, help='''hdf5 file with containment run data''')
	parser.add_argument('--plotdir', '-w', default='.', type=str, help='''directory to plot to''')
	args = parser.parse_args()
	c = main(**vars(args))






