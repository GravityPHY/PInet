import os
import glob

import numpy as np
from sklearn import neighbors

from utils.parser import read_wrl2,parsefile
from utils.conversion import pdb_to_wrl,wrl_to_pts
from utils.features import get_pqr_from_pdb,findvalue,getcontactbyabag


def add_EH_to_pts(pdb_id,pdb_path,pts_path,apbs_path,save_dir):
    newdict, labeldict = getcontactbyabag(pdb_path)
    file=np.loadtxt(pts_path)[:,0:3]
    coord=np.asarray(newdict)
    label=np.transpose(np.asarray(labeldict))
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(coord, label * 10)
    dist, indices=clf.kneighbors(file)
    apbs=open(apbs_path,"r")
    gl, orl, dl, vl = parsefile(apbs)
    av = findvalue(file,gl,orl, dl,vl)
    pred=np.sum(label[indices] * np.expand_dims(dist, 2), 1) / np.expand_dims(np.sum(dist, 1), 1) / 10
    np.savetxt(os.path.join(save_dir, f'{pdb_id}-l.pts'),
               np.concatenate((file, np.expand_dims(av, 1), np.expand_dims(pred[:, 0], 1)), axis=1))

pdbid = '6P50'
ab_ch = 'H L'
ag_ch = 'C'

#pdb_dir="/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound"
#output_folder="/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound"
#pdb_to_wrl(pdb_dir,output_folder)
#wrl_to_pts(pdb_dir,output_folder)
#get_pqr_from_pdb(pdb_dir,output_folder)
save_folder="/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound"
pdb_path = "/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_l_b.pdb"
pts_path = "/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_l_b.pts"
apbs_path = "/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_l_b.pqr.dx"

add_EH_to_pts(pdbid, pdb_path, pts_path, apbs_path,save_folder)
#newdickl, labeldickl = getcontactbyabag('/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_l_b.pdb')
#lfile = np.loadtxt("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_l_b.pts")[:, 0:3]
#lcoord = np.asarray(newdickl)
#llabel = np.transpose(np.asarray(labeldickl))
#clfl = neighbors.KNeighborsClassifier(3)
#clfl.fit(lcoord, llabel * 10)
#distl, indl = clfl.kneighbors(lfile)
#apbsl = open("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_l_b.pqr.dx", 'r')
#gl, orl, dl, vl = parsefile(apbsl)
#avl = findvalue(lfile, gl, orl, dl, vl)
#lpred = np.sum(llabel[indl] * np.expand_dims(distl, 2), 1) / np.expand_dims(np.sum(distl, 1), 1) / 10
#np.savetxt(os.path.join(save_folder,f'{pdbid}-l.pts'),
#           np.concatenate((lfile, np.expand_dims(avl, 1), np.expand_dims(lpred[:, 0], 1)), axis=1))

"""
newdickr, labeldickr = getcontactbyabag('/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_r_b.pdb')
rfile = np.loadtxt("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_r_b.pts")[:, 0:3]
rcoord = np.asarray(newdickr)
rlabel = np.transpose(np.asarray(labeldickr))
clfr = neighbors.KNeighborsClassifier(3)
clfr.fit(rcoord, rlabel * 10)
distr, indr = clfl.kneighbors(rfile)
apbsr = open("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/6P50_bound/6P50_r_b.pqr.dx", 'r')
gr, orr, dr, vr = parsefile(apbsr)
avr = findvalue(rfile, gr, orr, dr, vr)
rpred = np.sum(rlabel[indr] * np.expand_dims(distr, 2), 1) / np.expand_dims(np.sum(distr, 1), 1) / 10
np.savetxt(os.path.join(save_folder,f'{pdbid}-r.pts'),
           np.concatenate((rfile, np.expand_dims(avr, 1), np.expand_dims(rpred[:, 0], 1)), axis=1))
"""