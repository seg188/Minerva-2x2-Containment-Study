import numpy as np


#UNITS
cm = 1.0
mm = 1e-1
m = 1e2
ev = 1e-6
MeV = 1
GeV = 1e3
s=1
ns=1e-9

#Constants
c = 2.99e8*m/s

#Particle mass dictionary by pdg

#Detector config
#active volume 134 cm centered on 0.0 43.0 0.0 cm (x, y, z)

ACTIVE_VOLUME_LEN_X = 134*cm
ACTIVE_VOLUME_LEN_Y = 134*cm
ACTIVE_VOLUME_LEN_Z = 134*cm

ACTIVE_VOLUME_CENTER_X = 0.0*cm
ACTIVE_VOLUME_CENTER_Y = 43.0*cm
ACTIVE_VOLUME_CENTER_Z = 0.0*cm

MINERVA_2_MIN_Z = 130*cm + 10*cm
MINERVA_2_MAX_Z = 336*cm + 10*cm

MINERVA_1_MAX_Z = -130*cm - 10*cm
MINERVA_1_MIN_Z = -167*cm - 10*cm

MINERVA_2_ECAL_DEPTH = 45.58*cm
MINERVA_2_TRACKER_DEPTH = 69.68*cm

MINERVA_2_HCAL_ZMIN = MINERVA_2_MIN_Z+MINERVA_2_TRACKER_DEPTH+MINERVA_2_ECAL_DEPTH

MINERVA_TRACKER_RADIUS = 104.8*cm

MINERVA_ECAL_INNER_RADIUS = 104.8*cm
MINERVA_ECAL_OUTER_RADIUS = 123.7*cm

MINERVA_HCAL_INNER_RADIUS = 134.5*cm
MINERVA_HCAL_OUTER_RADIUS = 199.3*cm

MUON_TAG_CUT = 3*cm
FIDUCIAL_LEN = 28*cm
NHITS_TRACK_LIKE_CUT = 5