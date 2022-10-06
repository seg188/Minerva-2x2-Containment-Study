#!/usr/bin/env python3
import h5py
import numpy as np
#import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from consts import *
import time
from sklearn.decomposition import PCA
import scipy
import particle
import numba
'''
h5 file format for edep sim output
file['segments'] --> 

([('eventID', '<u4'), ('z_end', '<f4'), ('trackID', '<u4'), ('tran_diff', '<f4'), ('z_start', '<f4'), ('x_end', '<f4'), ('y_end', '<f4'), 
('n_electrons', '<u4'), ('pdgId', '<i4'), ('x_start', '<f4'), ('y_start', '<f4'), ('t_start', '<f4'), ('dx', '<f4'), ('long_diff', '<f4'), 
('pixel_plane', '<i4'), ('t_end', '<f4'), ('dEdx', '<f4'), ('dE', '<f4'), ('t', '<f4'), ('y', '<f4'), ('x', '<f4'), ('z', '<f4'), ('n_photons', '<f4')])

'''

datatype = np.dtype( \
		[("eventID","u4"),("contained","i4"),("W","f8"), ("lt_10cm","i4"), ("n_vtx_particles","i4"), 
		 ("fs_e", "f8"), ("fs_theta", "f8"), ("v_x", "f8"), ("v_y", "f8"), ("v_z", "f8"), 
         ("visible_energy","f8"), ("n_pions", "i4"), ("n_pi0", "i4"), ("n_protons", "i4"), ("nu_i", "i4"), ##add neutral pion count
         ("nu_i_energy", "f8"), ("q", "f8"), ("q2", "f8"), ("p1_p", "f8"), ("p1_theta", "f8"), ("p1_phi", "f8"), ("p2_p", "f8"), ("p2_theta", "f8"), ("p2_phi", "f8"),  
         ("pions_contained", "i4"), ("pr1_p", "f8"), ("pr1_theta", "f8"), ("cs", "i4"), ("containment_threshold", "f8"), ("containment_threshold_2x2", "f8"), 
         ("fs", "i4"), ("fs_energy_sum", "f8"), ("fs_contained", "i4"), ("all_contained", "i4"), ("all_but_fs_contained_2x2_only", "i4"), ("all_contained_2x2_only", "i4")]) 


def energy(p, m):
	return np.sqrt(p**2+m**2)

def is_muon(pdg):
	return np.absolute(pdg) in [-13, 13]

def get_available_energy(p, m, stable_flag):
	if stable_flag:
		if m > 0:
			return np.sqrt(p**2+m**2)-m
		else:
			return p
	else:
		return np.sqrt(p**2+m**2)

unknown_particles = set()

def get_particle_energy(p, pdgid):
	part = None 
	stable_flag = True 
	try:
		part = particle.Particle.from_pdgid(pdgid)
	except:
		return get_available_energy(np.linalg.norm(p), 0, stable_flag)
	try:
		if part.lifetime < 0.001: stable_flag=False    #p.lifetime gives lifetime in ns
		return get_available_energy(np.linalg.norm(p), part.mass, stable_flag)
	except:
		try: 
			return get_available_energy(np.linalg.norm(p), part.mass, stable_flag)
		except:
			return get_available_energy(np.linalg.norm(p), 0, stable_flag)

def get_punch_through_mask(x, y, z):
	mask = np.logical_and(np.logical_and( z > MINERVA_2_MAX_Z - 10, z < MINERVA_2_MAX_Z + 10), \
	 		np.sqrt(np.add(np.square(x),np.square(y))) < MINERVA_HCAL_OUTER_RADIUS)
	return mask

def trajectory_energy(tr):
	return get_particle_energy(tr['pxyz_start'], tr['pdgId'])

def contained_in_list_mask(data, _list):
	if len(_list)==0: return np.array([False]*len(data))
	mask = data==_list[0] 
	for val in _list[1:]:
		mask = np.logical_or(mask, data==val)
	return mask

def get_all_lineage(trackID, trajectories):
	daughter_trajectories_mask = trajectories['parentID']==trackID
	lineage_trajectories = trajectories[daughter_trajectories_mask]
	lineage = lineage_trajectories['trackID']
	all_lineage = set(lineage)
	for daughter_id in lineage:
		all_lineage.update(get_all_lineage(daughter_id, trajectories))
	return all_lineage

#@vectorize
def v_get_all_lineage(trackIDs, trajectories):
	total_lineage = set()
	for trackID in trackIDs:
		total_lineage.update(get_all_lineage(trackID, trajectories))

	return total_lineage


def get_neutron_energy(p):
	neutron_mass = 939.6 #Mev
	return np.sqrt(np.linalg.norm(p)**2 + neutron_mass**2) - neutron_mass


#@vectorize
def v_get_neutron_energy(ps):
	return np.vectorize(get_neutron_energy)(ps)

def in_fiducial_region(x, y, z):
	return (x < ACTIVE_VOLUME_CENTER_X + ACTIVE_VOLUME_LEN_X/2 - FIDUCIAL_LEN and x > ACTIVE_VOLUME_CENTER_X - ACTIVE_VOLUME_LEN_X/2 + FIDUCIAL_LEN) and \
		   (y < ACTIVE_VOLUME_CENTER_Y + ACTIVE_VOLUME_LEN_Y/2 - FIDUCIAL_LEN and y > ACTIVE_VOLUME_CENTER_Y - ACTIVE_VOLUME_LEN_Y/2 + FIDUCIAL_LEN) and \
		   (z < ACTIVE_VOLUME_CENTER_Z + ACTIVE_VOLUME_LEN_Z/2 - FIDUCIAL_LEN and z > ACTIVE_VOLUME_CENTER_Z - ACTIVE_VOLUME_LEN_Z/2 + FIDUCIAL_LEN) 

def get_active_lar_mask(x, y, z):
	return np.logical_and(np.logical_and(np.logical_and(x < ACTIVE_VOLUME_CENTER_X + ACTIVE_VOLUME_LEN_X/2,x > ACTIVE_VOLUME_CENTER_X - ACTIVE_VOLUME_LEN_X/2), \
		   np.logical_and(y < ACTIVE_VOLUME_CENTER_Y + ACTIVE_VOLUME_LEN_Y/2,y > ACTIVE_VOLUME_CENTER_Y - ACTIVE_VOLUME_LEN_Y/2)), \
		   np.logical_and(z < ACTIVE_VOLUME_CENTER_Z + ACTIVE_VOLUME_LEN_Z/2,z > ACTIVE_VOLUME_CENTER_Z - ACTIVE_VOLUME_LEN_Z/2))


def get_active_mask(x, y, z):
	minerva_1_mask = np.logical_and(np.logical_and( z > MINERVA_1_MIN_Z, z < MINERVA_1_MAX_Z), \
	 		np.sqrt(np.add(np.square(x),np.square(y))) < MINERVA_HCAL_OUTER_RADIUS)
	minerva_2_mask = np.logical_and(np.logical_and( z > MINERVA_2_MIN_Z, z < MINERVA_2_MAX_Z), \
	 		np.sqrt(np.add(np.square(x),np.square(y))) < MINERVA_HCAL_OUTER_RADIUS)
	minerva_mask = np.logical_or(minerva_1_mask, minerva_2_mask)
	return np.logical_or(minerva_mask, get_active_lar_mask(x, y, z))


def get_fiducial_mask(x, y, z):
	return np.logical_and(np.logical_and(np.logical_and(x < ACTIVE_VOLUME_CENTER_X + ACTIVE_VOLUME_LEN_X/2 - FIDUCIAL_LEN,x > ACTIVE_VOLUME_CENTER_X - ACTIVE_VOLUME_LEN_X/2 + FIDUCIAL_LEN), \
		   np.logical_and(y < ACTIVE_VOLUME_CENTER_Y + ACTIVE_VOLUME_LEN_Y/2 - FIDUCIAL_LEN,y > ACTIVE_VOLUME_CENTER_Y - ACTIVE_VOLUME_LEN_Y/2 + FIDUCIAL_LEN)), \
		   np.logical_and(z < ACTIVE_VOLUME_CENTER_Z + ACTIVE_VOLUME_LEN_Z/2 - FIDUCIAL_LEN,z > ACTIVE_VOLUME_CENTER_Z - ACTIVE_VOLUME_LEN_Z/2 + FIDUCIAL_LEN))

def get_detector_containment_mask(x, y, z):
	mask = np.logical_and(np.logical_and( z > MINERVA_1_MIN_Z, z < MINERVA_2_MAX_Z), \
	 		np.sqrt(np.add(np.square(x),np.square(y))) < MINERVA_HCAL_OUTER_RADIUS)
	return mask

def get_minerva_mask(x, y, z):
	return np.logical_and(np.logical_or(np.logical_and(z > MINERVA_2_MIN_Z, z < MINERVA_2_MAX_Z), np.logical_and(z > MINERVA_1_MIN_Z, z < MINERVA_1_MAX_Z)), \
			np.sqrt(np.add(np.square(x),np.square(y))) < MINERVA_HCAL_OUTER_RADIUS)

def get_muon_tag(segments, eventN):
	mask = segments['eventID']==eventN
	return any( mask ) and sum(mask.astype(int) < NHITS_TRACK_LIKE_CUT)

def get_theta(px, py, pz):#azimuthal angle, taking z as beam direction
	return np.arccos( pz/np.sqrt(px**2+py**2+pz**2) )

def get_theta_p(p):
	return get_theta(p[0], p[1], p[2])

def get_phi(px, py, pz):#azimuthal angle, taking z as beam direction
	return np.arctan2(py, px)

def floor(num):
	return int(np.floor(num))

def midpoint(arr):
	arr1 = arr[:-1]
	arr2 = arr[1:]
	return (arr1+arr2)/2

def is_in_active_lar(x, y, z):
	return (x < ACTIVE_VOLUME_CENTER_X + ACTIVE_VOLUME_LEN_X/2 and x > ACTIVE_VOLUME_CENTER_X-ACTIVE_VOLUME_LEN_X/2) \
	and (y < ACTIVE_VOLUME_CENTER_Y + ACTIVE_VOLUME_LEN_Y/2 and y > ACTIVE_VOLUME_CENTER_Y-ACTIVE_VOLUME_LEN_Y/2) \
	and (z < ACTIVE_VOLUME_CENTER_Z + ACTIVE_VOLUME_LEN_Z/2 and z > ACTIVE_VOLUME_CENTER_Z-ACTIVE_VOLUME_LEN_Z/2)

def is_in_minerva_tracker(x, y, z):
	return (np.sqrt(x**2 + y**2) < MINERVA_TRACKER_RADIUS) and (z > MINERVA_2_MIN_Z and z < MINERVA_2_MIN_Z+MINERVA_2_TRACKER_DEPTH) 

def is_in_minerva_ecal(x, y, z):
	return (np.sqrt(x**2 + y**2) < MINERVA_ECAL_OUTER_RADIUS) and (z > MINERVA_2_MIN_Z+MINERVA_2_TRACKER_DEPTH and z < MINERVA_2_MIN_Z+MINERVA_2_TRACKER_DEPTH+MINERVA_2_ECAL_DEPTH) 

def is_in_minerva_hcal(x, y, z):
	return (np.sqrt(x**2 + y**2) < MINERVA_HCAL_OUTER_RADIUS) and (z > MINERVA_2_HCAL_ZMIN and z < MINERVA_2_MAX_Z) 

def is_in_minerva_side_hcal(x, y, z):
	return (np.sqrt(x**2 + y**2) < MINERVA_HCAL_OUTER_RADIUS and np.sqrt(x**2 + y**2) > MINERVA_HCAL_INNER_RADIUS)

def is_in_minerva_side_ecal(x, y, z):
	return (np.sqrt(x**2 + y**2) < MINERVA_ECAL_OUTER_RADIUS and np.sqrt(x**2 + y**2) > MINERVA_ECAL_INNER_RADIUS)

def is_in_minerva_veto(x, y, z):
	return z < MINERVA_1_MAX_Z


#@vectorize
def v_get_2x2_minerva_subdetector(x, y, z):
	return np.vectorize(get_2x2_minerva_subdetector)(x, y, z)
	
def get_2x2_minerva_subdetector(x, y, z):
	if is_in_active_lar(x, y, z): return 'LAr'
	if is_in_minerva_tracker(x, y, z): return 'tracker'
	if is_in_minerva_hcal(x, y, z): return 'hcal'
	if is_in_minerva_ecal(x, y, z): return 'ecal'
	if is_in_minerva_veto(x, y, z): return 'm-veto'
	if is_in_minerva_side_ecal(x, y, z): return 'side-ecal'
	if is_in_minerva_side_hcal(x, y, z): return 'side-hcal'
	return 'unknown'

def get_muon_tagger_mask(z_start, z_end):
	return np.logical_or(z_start > MINERVA_2_MAX_Z-MUON_TAG_CUT, z_end > MINERVA_2_MAX_Z-MUON_TAG_CUT)


def analyze_muon_tag(truth_energies, _thetas, _muon_tags):
	nbins_energy = 50 # 200 MeV energy bins
	nbins_theta = 31
	theta_energy_bin_counts_full_range, theta_energy_edges_full_range = np.histogramdd([_thetas, truth_energies], bins=(nbins_theta, nbins_energy))
	muon_tag_theta_energy_means, _duplicate_theta_energy_edges = np.histogramdd([_thetas, truth_energies], bins=(nbins_theta, nbins_energy), weights=_muon_tags.astype(int))
	muon_tag_theta_energy_means = np.divide(muon_tag_theta_energy_means, theta_energy_bin_counts_full_range)
	return muon_tag_theta_energy_means

def analyze_containment(truth_energies,  _thetas, total_edeps, containment_cuts=[0.5], not_contained_flags=None):
	fractional_edeps = np.divide(total_edeps, truth_energies)
	contained_masks = []
	for cut in containment_cuts:
		if not_contained_flags is None:
			contained_masks.append(fractional_edeps >= cut)
		else:
			contained_masks.append(np.logical_and(fractional_edeps >= cut, np.logical_not(not_contained_flags)))
	
	nbins_energy = 50 # 100 MeV energy bins
	nbins_theta = 31
	MAX_SR_ENERGY = 5000

	contained_masks = np.array(contained_masks)
	energy_containment = []
	theta_containment = []
	th_energy_containment = []

	#Get bin counts for the 2d histograms to normalize

	thntotal, thetas = np.histogram(_thetas, bins=nbins_theta)
	thetas = midpoint(thetas)
	theta_energy_bin_counts_full_range, theta_energy_edges_full_range = np.histogramdd([_thetas, truth_energies], bins=(nbins_theta, nbins_energy))
	theta_energy_bin_counts_small_range, theta_energy_edges_small_range = np.histogramdd([_thetas, truth_energies], bins=(nbins_theta, nbins_energy), range=((0, 3.14),(0, MAX_SR_ENERGY)))
	
	ntotal_full_range, energies_full_range = np.histogram(truth_energies, bins=nbins_energy)
	energies_full_range = midpoint(energies_full_range)

	ntotal_small_range, energies_small_range = np.histogram(truth_energies, bins=nbins_energy, range=(0, MAX_SR_ENERGY))
	energies_small_range = midpoint(energies_small_range)

	for i, mask in enumerate(contained_masks):
		_containment, _duplicate_energies = np.histogram(truth_energies, weights=mask.astype(int), bins=nbins_energy, range=(0, MAX_SR_ENERGY))
		_containment = np.divide(_containment, ntotal_small_range)

		print('cut', containment_cuts[i])
		print('total contained:', sum(mask))
		print('array len', len(_thetas))

		_thcontainment, _duplicate_thetas = np.histogram(_thetas, weights=mask.astype(int), bins=nbins_theta)
		_thcontainment = np.divide(_thcontainment, thntotal)

		energy_containment.append(_containment)
		theta_containment.append(_thcontainment)

		##2d energy, theta map for containment
		theta_energy_means, _duplicate_theta_energy_edges = np.histogramdd([_thetas, truth_energies], bins=(nbins_theta, nbins_energy), weights=mask.astype(int), range=((0, 3.14),(0, MAX_SR_ENERGY)))
		theta_energy_means = np.divide(theta_energy_means, theta_energy_bin_counts_small_range)

		th_energy_containment.append(theta_energy_means)
		th_energy_containment.append(theta_energy_means)

	##2d energy, theta map for muon tag

	return np.array(energy_containment), energies_full_range, energies_small_range, np.array(theta_containment), thetas, th_energy_containment

#@vectorize
def v_distance_to_line(_line_direction, midpt, pt):
	line_direction = _line_direction/np.linalg.norm(_line_direction)
	def distance_to_line(pt):
		p = pt-midpt
		vector_to_line = p - np.dot(line_direction, p) * line_direction
		return np.linalg.norm(vector_to_line)
	return np.vectorize(distance_to_line)(pt)


def single_particle_contained(tr, _segments, _trajectories):
	#all decedent particles from initial particle
	lineage = get_all_lineage(tr['trackID'], _trajectories)
	#is_decendent_mask = contained_in_list_mask(_trajectories['trackID'], lineage)
	#trajectories = _trajectories[is_decendent_mask]

	#get all segments corresponding to those particles
	edeps_mask = contained_in_list_mask(_segments['trackID'], list(lineage)+[tr['trackID']])
	segments = _segments[edeps_mask]
	contained_mask = get_detector_containment_mask(segments['x_start'], segments['y_start'], segments['z_start'])

	uncontained_trids = list(set(segments['trackID'][np.logical_not(contained_mask)]))

	es = [-1]
	for trid in uncontained_trids:
		traj = _trajectories[_trajectories['trackID']==trid]
		es.append(np.linalg.norm(traj['pxyz_start'][0]))

	offending_track = max(es) 

	not_contained_flag = any(np.logical_not(contained_mask))

	contained_mask_2x2_only = get_active_lar_mask(segments['x_start'], segments['y_start'], segments['z_start'])

	uncontained_trids_2x2 = list(set(segments['trackID'][np.logical_not(contained_mask_2x2_only)]))

	es = [-1]
	for trid in uncontained_trids_2x2:
		traj = _trajectories[_trajectories['trackID']==trid]
		es.append(np.linalg.norm(traj['pxyz_start'][0]))

	offending_track_2x2 = max(es) 

	contained_2x2_only_flag = not any(np.logical_not(contained_mask_2x2_only))
	active_minerva_mask = get_minerva_mask(segments['x_start'], segments['y_start'], segments['z_start'])

	if not not_contained_flag: #marked as contained
		if any(np.logical_not(contained_mask_2x2_only)) and (not any(active_minerva_mask)):
			not_contained_flag = True
			outside_active = np.logical_and(np.logical_not(contained_mask_2x2_only), np.logical_not(active_minerva_mask)) #segments outside of minerva and 2x2
			inside_active = active_minerva_mask

			outside_trajectories = set(segments['trackID'][outside_active])
			inside_trajectories = set(segments['trackID'][inside_active])

			uncontained = list(outside_trajectories- inside_trajectories)
			es = [-1]
			for trid in uncontained:
				traj = _trajectories[_trajectories['trackID']==trid]
				es.append(np.linalg.norm(traj['pxyz_start'][0]))

			offending_track= max(es)

			if offending_track==-1: offending_track==9999999


			#get the energies of these trajectories--the max of these needs to be the containment threshold, now



########################################################
	mu_tag = False 

	if (tr['pdgId'] in [-13, 13]) and not_contained_flag: #muon, look for muon tag
		segments = segments[get_detector_containment_mask(segments['x_start'], segments['y_start'], segments['z_start'])]
		mu_tag = True
		punch_through_mask = get_punch_through_mask(segments['x_start'], segments['y_start'], segments['z_start'])
		punch_through = any(punch_through_mask)
		pca_done = False

		if punch_through:
			if len(segments) < 5:
				mu_tag = False
			else:
				pca = PCA(n_components=3)
				pca.fit(np.transpose([segments['x_start'], segments['y_start'], segments['z_start']]))
				max_pca = max(pca.explained_variance_ratio_)
				if max_pca < 0.85:
					mu_tag = False
					return not not_contained_flag, contained_2x2_only_flag, offending_track, offending_track_2x2

			pca_done = False
			if mu_tag:
				subdetectors = v_get_2x2_minerva_subdetector(segments['x_start'],segments['y_start'], segments['z_start'])
				#charge weighted pca
				for subdetector in set(subdetectors):
					if not subdetector in ['hcal', 'side-hcal']: continue
					subdet_mask = subdetectors==subdetector
					hits = segments[subdet_mask]
					if True:
						if len(hits) < 5: continue
						total_edep_this_subdet = np.sum(hits['dE'])
						if total_edep_this_subdet > 4000: 
							mu_tag = False
							return not not_contained_flag, contained_2x2_only_flag, offending_track, offending_track_2x2

					#fit the pts to a line, ensure least squares is low, average dE/dx in line direction is low
						hits_array = np.array( [hits['x_start'], hits['y_start'], hits['z_start']] )
						data = np.transpose(hits_array)
						datamean = data.mean(axis=0)
						uu, dd, vv = np.linalg.svd(data - datamean)
						direction = vv[0]

						sum_of_squares = np.linalg.norm(v_distance_to_line(direction, datamean, hits_array) )
					#	all_dedx.append(sum_of_squares)
						if sum_of_squares > 20000:
							mu_tag = False
							return not not_contained_flag, contained_2x2_only_flag, offending_track, offending_track_2x2

						if len(hits) > 5:
							if pca_done: second_pca_done = True
							pca_done = True
							pca = PCA(n_components=3)
							normed_charges = hits['dE']/sum(hits['dE'])
							pca.fit(np.transpose([np.multiply(hits['x_start'], normed_charges), np.multiply(hits['y_start'], normed_charges), np.multiply(hits['z_start'], normed_charges)]))
							max_ratio = max(pca.explained_variance_ratio_)
							if max_ratio < 0.9:
								mu_tag = False
								return not not_contained_flag, contained_2x2_only_flag, offending_track, offending_track_2x2
		else:
			mu_tag = False



	return ( (not not_contained_flag) or mu_tag ), contained_2x2_only_flag, offending_track, offending_track_2x2

	
def v4_norm(v4):
	return np.sqrt( np.absolute(v4[3]**2 - np.linalg.norm(v4[0:3])**2 ) ) 


def containment_study(_all_segments, _all_trajectories, _all_particle_stack, _all_vertices):
	#need to fill: ("containment_threshold", "f8"), ("containment_threshold_2x2", "f8"),

	ignore_particle_ids = [2112, 12, 14, 16, -12, -14, -16]
	all_segments = _all_segments['eventID', 'trackID', 'dE', 'dEdx', 'x_start', 'y_start', 'z_start', 'z_end']
	all_trajectories = _all_trajectories['eventID', 'trackID', 'parentID', 'xyz_start', 'pxyz_start', 'pdgId']
	all_particle_stack = _all_particle_stack['eventID', 'p4', 'status', 'pdgId']
	all_vertices = _all_vertices['eventID','x_vert', 'y_vert', 'z_vert']
	all_event_ids = set(all_trajectories['eventID'])
	data = np.empty( len(all_event_ids), dtype=datatype)
	counter = 0


	max_evdid = len(all_event_ids)
	nevents = 0.0
	n2x2=0
	total_times = 0.0
	for idata, eventN in enumerate(all_event_ids):
		start = time.time()
	#	if eventN > 100: return data

		_trajectories = all_trajectories[all_trajectories['eventID']==eventN]
		_trajectories = _trajectories[ np.logical_not(_trajectories['pdgId']==2112) ] #IGNORE NEUTRONS
		_trajectories = _trajectories[ [ True if np.linalg.norm(traj['pxyz_start']) > 10.0 else False for traj in _trajectories ] ]
		#for traj in _trajectories:
		#	print(np.linalg.norm(traj['pxyz_start']))

		_segments = all_segments[all_segments['eventID']==eventN]
		_stack = all_particle_stack[all_particle_stack['eventID']==eventN]
		_vertices = all_vertices[all_vertices['eventID']==eventN]

		data[idata]['v_x'] = _vertices['x_vert'][0]
		data[idata]['v_y'] = _vertices['y_vert'][0]
		data[idata]['v_z'] = _vertices['z_vert'][0]

		fs_stack = _stack[_stack['status']==1]
		i_stack = _stack[_stack['status']==0]

		data[idata]['fs_energy_sum'] = np.sum([part['p4'][3] for part in fs_stack])

		found_nu = False
		initial_energy = 0
		nu_p4 = None
		for ipart, pdg in enumerate(i_stack['pdgId']):
			if pdg in [12, -12, 14, -14]:
				data[idata]['nu_i'] = pdg
				found_nu = True
				initial_energy = i_stack[ipart]['p4'][3]*GeV
				nu_p4 = np.array(i_stack[ipart]['p4'])*GeV
				data[idata]['nu_i_energy'] = initial_energy
		if not found_nu: data[idata]['nu_i'] = 0

		found_fs = False
		fs_p4 = None
		n_pions = 0
		n_pi0 = 0
		n_protons = 0
		final_energy = 0
		pion_dict = {}
		proton_dict = {}
		data[idata]['cs'] = 0

		for ipart,pdg in enumerate(fs_stack['pdgId']):
			if pdg in strange_and_charmed: data[idata]['cs'] = 1
			if pdg in [12, -12, 14, -14, 11, -11, 13, -13]:
				final_energy = fs_stack[ipart]['p4'][3]*GeV
				fs_p4 = fs_stack[ipart]['p4']*GeV
				mass = particle.Particle.from_pdgid(pdg).mass
				theta = get_theta_p(fs_stack['p4'][ipart][:-1])

				if not found_fs: #
					data[idata]['fs'] = pdg #FIX ME!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
					data[idata]['fs_e'] = final_energy
					data[idata]['fs_theta'] = theta

				else:
					data[idata]['fs'] = -1
					data[idata]['fs_e'] = -1
					data[idata]['fs_theta'] = -1

				found_fs = True
				
			if pdg in [211, -211, 111]: n_pions += 1
			if np.absolute(pdg)==211:
				pion_dict[fs_stack['p4'][ipart][3]] = [get_theta_p(fs_stack['p4'][ipart][:-1]), np.arctan(fs_stack['p4'][ipart][1]/fs_stack['p4'][ipart][0]), ipart]
			if pdg == 111: n_pi0 += 1
			if pdg == 2212: 
				proton_dict[fs_stack['p4'][ipart][3]] = get_theta_p(fs_stack['p4'][ipart][:-1])
				n_protons += 1

		sorted_pions = sorted(list(pion_dict.keys()), reverse=True)
		sorted_protons = sorted(list(proton_dict.keys()), reverse=True)
		if len(sorted_pions) == 0:
			data[idata]['p1_p'] = -1
			data[idata]['p2_p'] = -1
			data[idata]['p1_theta'] = -1
			data[idata]['p2_theta'] = -1
			data[idata]['p1_phi'] = -1
			data[idata]['p2_phi'] = -1
		elif len(sorted_pions) == 1:
			data[idata]['p1_p'] = sorted_pions[0]
			data[idata]['p1_theta'] = pion_dict[sorted_pions[0]][0]
			data[idata]['p1_phi'] = pion_dict[sorted_pions[0]][1]
			data[idata]['p2_p'] = -1
			data[idata]['p2_theta'] = -1
			data[idata]['p2_phi'] = -1
		else:
			data[idata]['p1_p'] = sorted_pions[0]
			data[idata]['p1_theta'] = pion_dict[sorted_pions[0]][0]
			data[idata]['p1_phi'] = pion_dict[sorted_pions[0]][1]
			data[idata]['p2_p'] = sorted_pions[1]
			data[idata]['p2_theta'] = pion_dict[sorted_pions[1]][0]
			data[idata]['p2_phi'] = pion_dict[sorted_pions[1]][1]

		if len(sorted_protons)==0:
			data[idata]['pr1_p'] = -1
			data[idata]['pr1_theta'] = -1
		else:
			data[idata]['pr1_p'] = sorted_protons[0]
			data[idata]['pr1_theta'] = proton_dict[sorted_protons[0]]

		if not found_fs: data[idata]['fs'] = 0

		data[idata]['n_vtx_particles']=len(fs_stack['pdgId'])
		
		nucleon_stack =  _stack[_stack['status']==11]
		nucleon = None
		if len(nucleon_stack)==0:
			nucleon = None
			data[idata]['W'] = -1
		else:
			nucleon = nucleon_stack[0]
			nuc_mass = 0
			if nucleon['pdgId'] in [2000000200, 2000000201, 2000000202]: #NN, NP, PP pairs
				nuc_mass = 938.2*MeV #rough approximation
			else:
				nuc_mass = particle.Particle.from_pdgid(nucleon['pdgId']).mass
			nuc_p4 = np.array([0, 0, 0, nuc_mass])
			hadronic_p4 = nu_p4 + nuc_p4 - fs_p4
			data[idata]['W'] = v4_norm(hadronic_p4)
			#print(data['W'][-1])

		data[idata]['n_pions'] = n_pions
		data[idata]['n_pi0'] = n_pi0
		data[idata]['n_protons'] = n_protons

		data[idata]['q'] = np.absolute(final_energy-initial_energy)

		data[idata]['visible_energy'] = np.sum(_segments['dE'][get_active_mask(_segments['x_start'], _segments['y_start'], _segments['z_start'])])

		diff = fs_p4 - nu_p4
		data[idata]['q2'] =  v4_norm(diff)**2 

		fs_trajectory_mask = _trajectories['parentID']==-1
		fs_trajectories = _trajectories[fs_trajectory_mask]

		contained = []
		contained_2x2_only = []
		fs_contained  = False
		pions_contained = []
		
		all_es = []
		all_es_2x2 = []
		for tr in fs_trajectories:
			cont, cont_2x2, off_e, off_e_2x2 = single_particle_contained(tr, _segments, _trajectories)
			all_es_2x2.append(off_e_2x2)
			all_es.append(off_e)
			contained.append(cont)
			if (is_muon(tr['pdgId']) and cont): cont_2x2 = True 
			contained_2x2_only.append(cont_2x2)
			if np.absolute(tr['pdgId']) in [12, 13, 14]: fs_contained = cont
			if np.absolute(tr['pdgId'])==211: pions_contained.append(cont_2x2)

		#if not all(contained):
		#	print('***', eventN, '***')
		#	for itr, tr in enumerate(fs_trajectories):
		#		print('p:', np.linalg.norm(tr['pxyz_start']), ' --- pdg:', tr['pdgId'], '--- contained?:', contained[itr])
		data[idata]['all_contained'] = all(contained)
		data[idata]['all_contained_2x2_only'] = all(contained_2x2_only)
		if all(contained_2x2_only): n2x2+=1
		data[idata]['fs_contained'] = fs_contained
		data[idata]['all_but_fs_contained_2x2_only'] = sum(np.array(np.logical_not(contained_2x2_only)).astype(int))==1 and (not fs_contained)
		data[idata]['pions_contained']=all(pions_contained)
		data[idata]['containment_threshold'] = max(all_es)
		data[idata]['containment_threshold_2x2'] = max(all_es_2x2)

		#print(data[idata]['all_but_fs_contained_2x2_only'], contained_2x2_only, fs_contained)
		total_times += time.time()-start
		nevents+=1

		#print(data[idata]['all_contained'])

#		if data[idata]['containment_threshold_2x2'] < 20 and data[idata]['containment_threshold_2x2'] > 0:
#			print(eventN, data[idata]['containment_threshold_2x2'])

		if (eventN % 100 == 0): 
			print(eventN, '/', max_evdid)
			print('avg loop time:', total_times/nevents )


	return data

def main(_all_segments, _all_trajectories, _all_particle_stack, _all_vertices):
	return containment_study(_all_segments, _all_trajectories, _all_particle_stack, _all_vertices)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', '-i', type=str, help='''edep-sim file converted to hdf5''')
	args = parser.parse_args()
	c = main(**vars(args))