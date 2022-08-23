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
	
	data = f['data']

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
	ax.plot(mdpnts, normalized, drawstyle='steps-mid')
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
	ax.plot(mdpnts, normalized, drawstyle='steps-mid')
	ax.set_xlabel('$\omega$ [MeV]')
	ax.set_ylabel('Containment Fraction')
	fig.savefig(prefix + 'containment_vs_q.png')
	plt.close(fig)


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
	fig.suptitle('Pion Production vs. Energy Transfer -- NC, FHC')
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

	### plotting as a function of n pions produced
	pion_bins=[0, 1, 2, 3, 20]
	all_w = []
	pion_counts = np.array(data['n_pions'])[w_mask]
	
	#W
	for inp in range(len(pion_bins)-1):
		ws = []
		for iEvent in range(len(w_data)):
			if pion_counts[iEvent] >= pion_bins[inp] and pion_counts[iEvent] < pion_bins[inp+1]:
				ws.append(w_data[iEvent])
		all_w.append(ws)

	#Q, E
	pion_counts = np.array(data['n_pions'])
	all_contained = data['all_contained']
	print('starting q, e loop')
	print('pion counts length:', len(pion_counts))
	print('initial nu energy len:', len(data['nu_i_energy']))
	all_q = []
	all_energies = []
	all_contained_q_e = []
	energy_data = data['nu_i_energy']
	q_data = data['q']
	for inp in range(len(pion_bins)-1):
		qs = []
		energies = []
		contained_q_e = []
		for iEvent in range(len(pion_counts)):
			if pion_counts[iEvent] >= pion_bins[inp] and pion_counts[iEvent] < pion_bins[inp+1]:
				qs.append(q_data[iEvent])
				energies.append(energy_data[iEvent])
				contained_q_e.append(all_contained[iEvent])
		all_q.append(qs)
		all_energies.append(energies)
		all_contained_q_e.append(contained_q_e)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Pion Production vs. W -- CC')
	labels = []
	for iplt in range(len(pion_bins)-1):
		_label = 'cc-' + str(pion_bins[iplt]) + 'pi' if iplt < len(all_energies)-1 else  'cc-' + str(pion_bins[iplt]) + '+pi'
		labels.append(_label)
	plt.hist(all_w, bins=e_bins, stacked=False, histtype='step', label=labels, range=(e_min, e_max))
	plt.hist(w_data, range=(e_min, e_max), bins=e_bins, alpha=0.3, label='cc-inclusive')
	ax.legend()
	ax.set_xlabel('W [MeV]')
	fig.savefig(prefix+'W_n_pions.png')
	plt.close(fig)

	labels = []
	for iplt in range(len(all_energies)):
		_label = str(pion_bins[iplt]) + 'pi' if iplt < len(all_energies)-1 else str(pion_bins[iplt]) + '+pi'
		labels.append(_label)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Energy Transfer by Pion Production')
	plt.hist(all_q, bins=e_bins, stacked=False, histtype='step', label=labels, range=(0, 8000))
	ax.legend()
	ax.set_xlabel('$\omega$ [MeV]')
	fig.savefig(prefix+'Q__pion_production.png')
	plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Containment vs. Energy Transfer by Pion Production')
	q_c_min, q_c_max = 0, 3000
	q_c_bins = 25
	for iplt in range(len(all_q)):
		norms, edges = np.histogram(all_q[iplt], bins=q_c_bins, range=(q_c_min, q_c_max))
		contained, edges = np.histogram(all_q[iplt], bins=q_c_bins, range=(q_c_min, q_c_max), weights=np.array(np.array(all_contained_q_e[iplt]).astype(int) ))
		fraction = np.divide(contained, norms)
		mdpnts = edges[:-1] + np.diff(edges)/2
		plt.plot(mdpnts, fraction, label=labels[iplt], drawstyle='steps-mid')
	ax.legend()
	ax.set_xlabel('$\omega$ [MeV]')
	fig.savefig(prefix+'Q_n_pions_containment.png')
	plt.close(fig)

	
	#2d histogram of truth energy vs active edeps
	w_min, w_max = 500, 4500
	w_nbins = 70
	q2_min, q2_max = 0, 4e6
	q2_nbins = 70
	fig = plt.figure()
	ax = fig.add_subplot()
	qwhist2d, qwedges2d = np.histogramdd([np.array(data['W']), np.array(data['q2'])], weights=np.array(data['all_contained'].astype(int)), bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
	norm_qwhist2d, qwedges2d = np.histogramdd([np.array(data['W']), np.array(data['q2'])], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
	scaled_qwhist2d = np.divide(qwhist2d, norm_qwhist2d)
	im = plt.imshow(np.transpose(scaled_qwhist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
	ax.set_xlabel('W [MeV]')
	ax.set_ylabel('$Q^2$ [MeV$^2$]')
	fig.suptitle('Containment -- W vs. $Q^2$')
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label('contained fraction', rotation=-90)
	fig.savefig(prefix + 'w_q2_containment.png')
	plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot()
	norm_qwhist2d, qwedges2d = np.histogramdd([np.array(data['W']), np.array(data['q2'])], bins=(int(w_nbins*1.5), int(q2_nbins*1.5) ), range=( (w_min, w_max), (q2_min, q2_max) ))		
	im = plt.imshow(np.transpose(norm_qwhist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
	ax.set_xlabel('W [MeV]')
	ax.set_ylabel('$Q^2$ [MeV^2]')
	fig.suptitle('W vs. $Q^2$')
	fig.savefig(prefix + 'w_q2_distribution.png')
	plt.close(fig)


	fig = plt.figure()
	ax = fig.add_subplot()
	ax.hist(data['q2'], bins=100)
	fig.suptitle('$Q^2$')
	ax.set_xlabel('$Q^2$ [MeV]')
	fig.savefig(prefix+'q2.png')
	plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.hist(data['n_protons'], bins=[i for i in range(20)] )
	fig.suptitle('n protons')
	ax.set_xlabel('n')
	fig.savefig(prefix+'n_protons.png')
	plt.close(fig)

	proton_counts = np.array(data['n_protons'])
	all_contained = data['all_contained']
	all_q = []
	all_energies = []
	all_contained_q_e = []
	energy_data = data['nu_i_energy']
	q_data = data['q']
	for inp in range(len(pion_bins)-1):
		qs = []
		energies = []
		contained_q_e = []
		for iEvent in range(len(proton_counts)):
			if proton_counts[iEvent] >= pion_bins[inp] and proton_counts[iEvent] < pion_bins[inp+1]:
				qs.append(q_data[iEvent])
				energies.append(energy_data[iEvent])
				contained_q_e.append(all_contained[iEvent])
		all_q.append(qs)
		all_energies.append(energies)
		all_contained_q_e.append(contained_q_e)

	fig = plt.figure()
	ax = fig.add_subplot()
	fig.suptitle('Containment vs. Energy Transfer by Proton Production')
	q_c_min, q_c_max = 0, 3000
	q_c_bins = 25
	for iplt in range(len(all_q)):
		norms, edges = np.histogram(all_q[iplt], bins=q_c_bins, range=(q_c_min, q_c_max))
		contained, edges = np.histogram(all_q[iplt], bins=q_c_bins, range=(q_c_min, q_c_max), weights=np.array(np.array(all_contained_q_e[iplt]).astype(int) ))
		fraction = np.divide(contained, norms)
		mdpnts = edges[:-1] + np.diff(edges)/2
		plt.plot(mdpnts, fraction, label=str(iplt)+'p' if iplt < len(all_q)-1 else str(iplt)+'+p', drawstyle='steps-mid')
	ax.legend()
	ax.set_xlabel('$\omega$ [MeV]')
	fig.savefig(prefix+'Q_n_protons_containment.png')
	plt.close(fig)


	fig = plt.figure(figsize=(10,8))
	fig.suptitle('Containment by Pion Production, W-$Q^2$ Plane')
	ax0 = fig.add_subplot(221)
	ax0.set_title('0 Pion Production')
	ax0.set_xlabel('W')
	ax0.set_ylabel('$Q^2$')
	ax1 = fig.add_subplot(222)
	ax1.set_title('1 Pion Production')
	ax1.set_xlabel('W')
	ax1.set_ylabel('$Q^2$')
	ax2 = fig.add_subplot(223)
	ax2.set_title('2 Pion Production')
	ax2.set_xlabel('W')
	ax2.set_ylabel('$Q^2$')
	ax3 = fig.add_subplot(224)
	ax3.set_title('3 Pion Production')
	ax3.set_xlabel('W')
	ax3.set_ylabel('$Q^2$')
	axs = [ax0, ax1, ax2, ax3]
	for pion_count in [0, 1, 2, 3]:
		correct_pion_mask = np.array(data['n_pions'])==pion_count
		w_data = np.array(data['W'])[correct_pion_mask]
		q2_data = np.array(data['q2'])[correct_pion_mask]
		hist2d, edges2d = np.histogramdd([w_data, q2_data], weights=np.array(data['all_contained'].astype(int))[correct_pion_mask], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
		norm_hist2d, edges2d = np.histogramdd([w_data, q2_data], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
		scaled_hist2d = np.divide(hist2d, norm_hist2d)
		im = axs[pion_count].imshow(np.transpose(scaled_hist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
		cbar = fig.colorbar(im, ax=axs[pion_count])

	plt.savefig(prefix+'full_containment_by_pions.png')
	plt.close(fig)


	fig = plt.figure(figsize=(10,8))
	fig.suptitle('2x2 Only Containment by Pion Production, W-$Q^2$ Plane')
	ax0 = fig.add_subplot(221)
	ax0.set_title('0 Pion Production')
	ax0.set_xlabel('W')
	ax0.set_ylabel('$Q^2$')
	ax1 = fig.add_subplot(222)
	ax1.set_title('1 Pion Production')
	ax1.set_xlabel('W')
	ax1.set_ylabel('$Q^2$')
	ax2 = fig.add_subplot(223)
	ax2.set_title('2 Pion Production')
	ax2.set_xlabel('W')
	ax2.set_ylabel('$Q^2$')
	ax3 = fig.add_subplot(224)
	ax3.set_title('3 Pion Production')
	ax3.set_xlabel('W')
	ax3.set_ylabel('$Q^2$')
	axs = [ax0, ax1, ax2, ax3]
	for pion_count in [0, 1, 2, 3]:
		correct_pion_mask = np.array(data['n_pions'])==pion_count
		w_data = np.array(data['W'])[correct_pion_mask]
		q2_data = np.array(data['q2'])[correct_pion_mask]
		hist2d, edges2d = np.histogramdd([w_data, q2_data], weights=np.array(data['all_contained_2x2_only'].astype(int))[correct_pion_mask], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
		norm_hist2d, edges2d = np.histogramdd([w_data, q2_data], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
		scaled_hist2d = np.divide(hist2d, norm_hist2d)
		im = axs[pion_count].imshow(np.transpose(scaled_hist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
		cbar = fig.colorbar(im, ax=axs[pion_count])

	plt.savefig(prefix+'2x2_only_containment_by_pions.png')
	plt.close(fig)


	fig = plt.figure(figsize=(10,8))
	fig.suptitle('Containment by Pi0 Production, W-$Q^2$ Plane')
	ax0 = fig.add_subplot(221)
	ax0.set_title('0 Pi0 Production')
	ax0.set_xlabel('W')
	ax0.set_ylabel('$Q^2$')
	ax1 = fig.add_subplot(222)
	ax1.set_title('1 Pi0 Production')
	ax1.set_xlabel('W')
	ax1.set_ylabel('$Q^2$')
	ax2 = fig.add_subplot(223)
	ax2.set_title('2 Pi0 Production')
	ax2.set_xlabel('W')
	ax2.set_ylabel('$Q^2$')
	ax3 = fig.add_subplot(224)
	ax3.set_title('3 Pi0 Production')
	ax3.set_xlabel('W')
	ax3.set_ylabel('$Q^2$')
	axs = [ax0, ax1, ax2, ax3]
	for pion_count in [0, 1, 2, 3]:
		correct_pion_mask = np.array(data['n_pi0'])==pion_count
		w_data = np.array(data['W'])[correct_pion_mask]
		q2_data = np.array(data['q2'])[correct_pion_mask]
		hist2d, edges2d = np.histogramdd([w_data, q2_data], weights=np.array(data['all_contained'].astype(int))[correct_pion_mask], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
		norm_hist2d, edges2d = np.histogramdd([w_data, q2_data], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
		scaled_hist2d = np.divide(hist2d, norm_hist2d)
		im = axs[pion_count].imshow(np.transpose(scaled_hist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
		cbar = fig.colorbar(im, ax=axs[pion_count])

	plt.savefig(prefix+'full_containment_by_n_pi0.png')
	plt.close(fig)


	fig = plt.figure(figsize=(10,8))
	fig.suptitle('2x2 Only Containment by Pi0 Production, W-$Q^2$ Plane')
	ax0 = fig.add_subplot(221)
	ax0.set_title('0 Pi0 Production')
	ax0.set_xlabel('W')
	ax0.set_ylabel('$Q^2$')
	ax1 = fig.add_subplot(222)
	ax1.set_title('1 Pi0 Production')
	ax1.set_xlabel('W')
	ax1.set_ylabel('$Q^2$')
	ax2 = fig.add_subplot(223)
	ax2.set_title('2 Pi0 Production')
	ax2.set_xlabel('W')
	ax2.set_ylabel('$Q^2$')
	ax3 = fig.add_subplot(224)
	ax3.set_title('3 Pi0 Production')
	ax3.set_xlabel('W')
	ax3.set_ylabel('$Q^2$')
	axs = [ax0, ax1, ax2, ax3]
	for pion_count in [0, 1, 2, 3]:
		correct_pion_mask = np.array(data['n_pi0'])==pion_count
		w_data = np.array(data['W'])[correct_pion_mask]
		q2_data = np.array(data['q2'])[correct_pion_mask]
		hist2d, edges2d = np.histogramdd([w_data, q2_data], weights=np.array(data['all_contained_2x2_only'].astype(int))[correct_pion_mask], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
		norm_hist2d, edges2d = np.histogramdd([w_data, q2_data], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
		scaled_hist2d = np.divide(hist2d, norm_hist2d)
		im = axs[pion_count].imshow(np.transpose(scaled_hist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
		cbar = fig.colorbar(im, ax=axs[pion_count])

	#plt.show()
	plt.savefig(prefix+'2x2_only_containment_by_n_pi0.png')
	plt.close(fig)


	fig = plt.figure(figsize=(10,8))
	fig.suptitle('Containment by Charged Pion Production, W-$Q^2$ Plane')
	ax0 = fig.add_subplot(221)
	ax0.set_title('0 Pi Production')
	ax0.set_xlabel('W')
	ax0.set_ylabel('$Q^2$')
	ax1 = fig.add_subplot(222)
	ax1.set_title('1 Pi Production')
	ax1.set_xlabel('W')
	ax1.set_ylabel('$Q^2$')
	ax2 = fig.add_subplot(223)
	ax2.set_title('2 Pi Production')
	ax2.set_xlabel('W')
	ax2.set_ylabel('$Q^2$')
	ax3 = fig.add_subplot(224)
	ax3.set_title('3 Pi Production')
	ax3.set_xlabel('W')
	ax3.set_ylabel('$Q^2$')
	axs = [ax0, ax1, ax2, ax3]
	for pion_count in [0, 1, 2, 3]:
		correct_pion_mask = (np.array(data['n_pions']) - np.array(data['n_pi0']))==pion_count
		w_data = np.array(data['W'])[correct_pion_mask]
		q2_data = np.array(data['q2'])[correct_pion_mask]
		hist2d, edges2d = np.histogramdd([w_data, q2_data], weights=np.array(data['all_contained'].astype(int))[correct_pion_mask], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
		norm_hist2d, edges2d = np.histogramdd([w_data, q2_data], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
		scaled_hist2d = np.divide(hist2d, norm_hist2d)
		im = axs[pion_count].imshow(np.transpose(scaled_hist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
		cbar = fig.colorbar(im, ax=axs[pion_count])

	plt.savefig(prefix+'full_containment_by_n_charged_pions.png')
	plt.close(fig)


	fig = plt.figure(figsize=(10,8))
	fig.suptitle('2x2 Only Containment by Charged Pion Production, W-$Q^2$ Plane')
	ax0 = fig.add_subplot(221)
	ax0.set_title('0 Pi Production')
	ax0.set_xlabel('W')
	ax0.set_ylabel('$Q^2$')
	ax1 = fig.add_subplot(222)
	ax1.set_title('1 Pi Production')
	ax1.set_xlabel('W')
	ax1.set_ylabel('$Q^2$')
	ax2 = fig.add_subplot(223)
	ax2.set_title('2 Pi Production')
	ax2.set_xlabel('W')
	ax2.set_ylabel('$Q^2$')
	ax3 = fig.add_subplot(224)
	ax3.set_title('3 Pi Production')
	ax3.set_xlabel('W')
	ax3.set_ylabel('$Q^2$')
	axs = [ax0, ax1, ax2, ax3]
	for pion_count in [0, 1, 2, 3]:
		correct_pion_mask = (np.array(data['n_pions']) - np.array(data['n_pi0']))==pion_count
		w_data = np.array(data['W'])[correct_pion_mask]
		q2_data = np.array(data['q2'])[correct_pion_mask]
		hist2d, edges2d = np.histogramdd([w_data, q2_data], weights=np.array(data['all_contained_2x2_only'].astype(int))[correct_pion_mask], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))
		norm_hist2d, edges2d = np.histogramdd([w_data, q2_data], bins=(int(w_nbins/1.5), int(q2_nbins/1.5)), range=( (w_min, w_max), (q2_min, q2_max) ))		
		scaled_hist2d = np.divide(hist2d, norm_hist2d)
		im = axs[pion_count].imshow(np.transpose(scaled_hist2d), interpolation='nearest', extent=[min(qwedges2d[0]), max(qwedges2d[0]), min(qwedges2d[1]), max(qwedges2d[1])],  aspect='auto',origin='lower')
		cbar = fig.colorbar(im, ax=axs[pion_count])

	#plt.show()
	plt.savefig(prefix+'2x2_only_containment_by_charged_pions.png')
	plt.close(fig)


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






