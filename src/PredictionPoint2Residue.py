import os
import sys
import csv
import json
import random
import numpy as np
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, average_precision_score, confusion_matrix, \
    matthews_corrcoef, precision_recall_curve

from getcontactEpipred import getcontactbyabag, getsppider2, getalign
from scipy.special import expit

from pathlib import Path

dir_path=sys.argv[1]
pdb = Path(sys.argv[1]).stem[0:4]

rfile = os.path.join(dir_path,pdb + '-r.pts')
rcoord = np.transpose(np.loadtxt(rfile))[0:3, :]

lfile = os.path.join(dir_path,pdb + '-l.pts')
lcoord = np.transpose(np.loadtxt(lfile))[0:3, :]

profile = os.path.join(dir_path,pdb + '_prob_r_l.seg')
prolabel = np.loadtxt(profile)

pror = prolabel[0:rcoord.shape[1]]
prol = prolabel[rcoord.shape[1]:]

rcoord = np.transpose(rcoord)
lcoord = np.transpose(lcoord)

nn = 3
dt = 2
cutoff = 0.5
tol = [6, 6, 6]

rl, nl, cl = getsppider2(os.path.join(dir_path,pdb + '-l.pdb'))
rr, nr, cr = getsppider2(os.path.join(dir_path,pdb + '-r.pdb'))

cencoordr = np.asarray(nr)

cencoordl = np.asarray(nl)

clfr = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(rcoord)
distancesr, indicesr = clfr.kneighbors(cencoordr)

probr = [0] * len(cr)
for ii, ind in enumerate(indicesr):
    for jj, sind in enumerate(ind):
        if distancesr[ii][jj] > dt:
            continue
        probr[rr[ii]] = max(probr[rr[ii]], pror[sind])

np.savetxt(os.path.join(dir_path,pdb + '_resi_r.seg'), np.array(probr))

clfl = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(lcoord)
distancesl, indicesl = clfl.kneighbors(cencoordl)

probl = [0] * len(cl)
for ii, ind in enumerate(indicesl):
    for jj, sind in enumerate(ind):

        if distancesl[ii][jj] > dt:
            continue

        probl[rl[ii]] = max(probl[rl[ii]], prol[sind])
np.savetxt(os.path.join(dir_path,pdb + '_resi_l.seg'), np.array(probl))
