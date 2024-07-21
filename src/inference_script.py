# cpu only version
from __future__ import print_function
import argparse
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from Bio.PDB import *

from scipy.special import expit
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, average_precision_score, confusion_matrix, \
    matthews_corrcoef, precision_recall_curve

from getcontactEpipred import getcontactbyabag, getsppider2, getalign

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from pathlib import Path
from pinet.model import PointNetDenseCls12, feature_transform_regularizer

import warnings
warnings.filterwarnings("ignore")


random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))

def predict_label(filel_path, filer_path, save_dir, model_path,num_classes = 2):
    """

    :param filel_path: path to ligand point cloud (e.g. ABCD-l.pts)
    :param filer_path: path to receptor point cloud (e.g. ABCD-r.pts)
    :param save_dir: path to saving directory
    :param model_path: path to model parameters
    :param num_classes: number of prediction classes
    :return: None
    """
    # prepare classifier
    num_classes = 2
    classifier = PointNetDenseCls12(k=num_classes, feature_transform=False, pdrop=0.0, id=5)
    PATH=model_path
    classifier.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    classifier.eval()

    try:
        pointsr = np.loadtxt(filer_path).astype(np.float32)
        pointsl = np.loadtxt(filel_path).astype(np.float32)
    except Exception as e:
        print(pdb_id)
        return

    coordsetr = pointsr[:, 0:3]
    featsetr = pointsr[:, 3:]

    coordsetl = pointsl[:, 0:3]
    featsetl = pointsl[:, 3:]

    featsetr = featsetr / np.sqrt(np.max(featsetr ** 2, axis=0))
    featsetl = featsetl / np.sqrt(np.max(featsetl ** 2, axis=0))

    coordsetr = coordsetr - np.expand_dims(np.mean(coordsetr, axis=0), 0)  # center
    coordsetl = coordsetl - np.expand_dims(np.mean(coordsetl, axis=0), 0)  # center

    pointsr[:, 0:5] = np.concatenate((coordsetr, featsetr), axis=1)
    pointsl[:, 0:5] = np.concatenate((coordsetl, featsetl), axis=1)

    pointsr = torch.from_numpy(pointsr).unsqueeze(0)
    pointsl = torch.from_numpy(pointsl).unsqueeze(0)

    memlim = 120000
    if pointsl.size()[1] + pointsr.size()[1] > memlim:
        lr = pointsl.size()[1] * memlim // (pointsl.size()[1] + pointsr.size()[1])
        rr = pointsr.size()[1] * memlim // (pointsl.size()[1] + pointsr.size()[1])
        ls = np.random.choice(pointsl.size()[1], lr, replace=False)
        rs = np.random.choice(pointsr.size()[1], rr, replace=False)

        pointsr = pointsr[:, rs, :]
        pointsl = pointsl[:, ls, :]

    pointsr = pointsr.transpose(2, 1)  # .cuda()
    pointsl = pointsl.transpose(2, 1)  # .cuda()

    classifier = classifier.eval()

    pred, _, _ = classifier(pointsr, pointsl)

    pred = pred.view(-1, 1)

    np.savetxt(os.path.join(save_dir, Path(filel_path).stem[0:4] + '_prob_r_l.seg'),
               torch.sigmoid(pred).view(1, -1).data.cpu())

def seperate_label_r_l(filel_path, filer_path,
                       ligand_pdb_path,receptor_pdb_path,
                       labelfile_path, save_dir, save_id):
    """

    :param filel_path: path to ligand point cloud (e.g. ABCD-l.pts)
    :param filer_path: path to receptor point cloud (e.g. ABCD-r.pts)
    :param labelfile_path: path to predict label file (e.g. ABCD_prob_r_l.seg)
    :param save_dir: path to directory that you want to save the out put
    :return:
    """
    lcoord = np.transpose(np.loadtxt(filel_path))[0:3, :]
    rcoord = np.transpose(np.loadtxt(filer_path))[0:3, :]
    prolabel = np.loadtxt(labelfile_path)
    pror = prolabel[0:rcoord.shape[1]]
    prol = prolabel[rcoord.shape[1]:]
    rcoord = np.transpose(rcoord)
    lcoord = np.transpose(lcoord)

    nn = 3
    dt = 2
    cutoff = 0.5
    tol = [6, 6, 6]

    rl, nl, cl = getsppider2(ligand_pdb_path)
    rr, nr, cr = getsppider2(receptor_pdb_path)

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
    np.savetxt(os.path.join(save_dir, save_id + '_resi_r.seg'), np.array(probr))

    clfl = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(lcoord)
    distancesl, indicesl = clfl.kneighbors(cencoordl)
    probl = [0] * len(cl)
    for ii, ind in enumerate(indicesl):
        for jj, sind in enumerate(ind):
            if distancesl[ii][jj] > dt:
                continue
            probl[rl[ii]] = max(probl[rl[ii]], prol[sind])
    np.savetxt(os.path.join(save_dir, save_id + '_resi_l.seg'), np.array(probl))

def getsppider2(file):
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', file)
    residic = []
    # tempo=[]
    newdic = []
    labeldic = []
    bdic = []
    cd = []
    mark = 0
    for c in structure[0]:
        for resi in c:
            # residic.append(resi._id[1])
            cen = [0, 0, 0]
            count = 0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                # if 'H' in atom.get_name():
                #     continue
                cen[0] += atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count += 1

                # residic.append(resi._id[1])
                residic.append(mark)
                newdic.append([atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]])
            cen = [coor * 1.0 / count for coor in cen]
            mark += 1
            cd.append(cen)
            # labeldic.append(1)

    # print len(residic)
    # print len(bdic)
    return residic, np.asarray(newdic), cd

def set_bfactor(pdb_file, res_file, save_path):
    r, n, c = getsppider2(pdb_file)
    r_file = open(res_file, 'r').readlines()
    #print(len(r_file))
    #print(len(c))

    assert len(r_file) == len(c)
    parser = PDBParser()
    structure = parser.get_structure('C', pdb_file)
    for c in structure[0]:
        for i, resi in enumerate(c):
            for atom in resi:
                atom.set_bfactor(float(r_file[i]))
                #print(r_file[i])
    if True:
        io = PDBIO()
        io.set_structure(structure)
        io.save(os.path.join(save_path,pdb_file))


for folder in os.listdir("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest"):
    pdb_id = folder[0:4]
    save_folder = os.path.join("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/", folder)
    print(save_folder)


    filel = os.path.join(save_folder,f"{pdb_id}-l.pts")#sys.argv[1]
    filer = os.path.join(save_folder,f"{pdb_id}-r.pts")#sys.argv[2]
    pdbl = os.path.join(save_folder, f"{pdb_id}_l_b.pdb")
    pdbr = os.path.join(save_folder, f"{pdb_id}_r_b.pdb")
    labelrl=os.path.join(save_folder, f"{pdb_id}_prob_r_l.seg")

    #num_classes = 2
    #classifier = PointNetDenseCls12(k=num_classes, feature_transform=False, pdrop=0.0, id=5)
    #try:
    #    seperate_label_r_l(filel,filer, pdbl, pdbr, labelrl, save_folder, pdb_id)
    #except Exception as e:
    #    print(e)
    try:
        pdb_file = os.path.join(save_folder, f"{pdb_id}_l_b.pdb")
        res_file = os.path.join(save_folder, f"{pdb_id}_resi_l.seg")
        set_bfactor(pdb_file, res_file, save_folder)
    except Exception as e:
        print(pdb_id,e)

# classifier.cuda()


    PATH = '../models/dbd_aug.pth'

    #predict(filel,filer, save_folder, PATH)

"""
    classifier.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    classifier.eval()
    try:
        pointsr = np.loadtxt(filer).astype(np.float32)
        pointsl = np.loadtxt(filel).astype(np.float32)
    except Exception as e:
        print(pdb_id)
        continue

    coordsetr = pointsr[:, 0:3]
    featsetr = pointsr[:, 3:]

    coordsetl = pointsl[:, 0:3]
    featsetl = pointsl[:, 3:]

    featsetr = featsetr / np.sqrt(np.max(featsetr ** 2, axis=0))
    featsetl = featsetl / np.sqrt(np.max(featsetl ** 2, axis=0))

    coordsetr = coordsetr - np.expand_dims(np.mean(coordsetr, axis=0), 0)  # center
    coordsetl = coordsetl - np.expand_dims(np.mean(coordsetl, axis=0), 0)  # center

    pointsr[:, 0:5] = np.concatenate((coordsetr, featsetr), axis=1)
    pointsl[:, 0:5] = np.concatenate((coordsetl, featsetl), axis=1)

    pointsr = torch.from_numpy(pointsr).unsqueeze(0)
    pointsl = torch.from_numpy(pointsl).unsqueeze(0)

    memlim = 120000
    if pointsl.size()[1] + pointsr.size()[1] > memlim:
        lr = pointsl.size()[1] * memlim // (pointsl.size()[1] + pointsr.size()[1])
        rr = pointsr.size()[1] * memlim // (pointsl.size()[1] + pointsr.size()[1])
        ls = np.random.choice(pointsl.size()[1], lr, replace=False)
        rs = np.random.choice(pointsr.size()[1], rr, replace=False)

        pointsr = pointsr[:, rs, :]
        pointsl = pointsl[:, ls, :]

    pointsr = pointsr.transpose(2, 1)  # .cuda()
    pointsl = pointsl.transpose(2, 1)  # .cuda()

    classifier = classifier.eval()

    pred, _, _ = classifier(pointsr, pointsl)

    pred = pred.view(-1, 1)

    np.savetxt(os.path.join(save_folder, Path(filel).stem[0:4] + '_prob_r_l.seg'),
               torch.sigmoid(pred).view(1, -1).data.cpu())
# save_folder="/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/5ZUF_bound"
# np.savetxt(os.path.join(save_folder,Path(filel).stem[0:4]+'_prob_r_l.seg'),torch.sigmoid(pred).view(1, -1).data.cpu())
# print("Prediction results save to",os.path.join(save_folder,Path(filel).stem[0:4]+'_prob_r_l.seg'))

"""