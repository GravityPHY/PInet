import os
import glob
import numpy as np
from pathlib import Path

from Bio.PDB import PDBParser


hfile = open('/projectnb2/docking/imhaoyu/24_epitope_mapping/PInet/src/utils/hydro.csv', 'r')
hdic = {}
for line in hfile:
    ll = line.split(':')
    if len(ll) > 1:
        hdic[ll[0].upper()] = float(ll[1])
# print hdic
# print len(hdic)
edic = {}
charge = ['LYS', 'ARG', 'HIS']
necharge = ['ASP', 'GLU']
for aa in hdic:
    if aa in charge:
        edic[aa] = 1
    elif aa in necharge:
        edic[aa] = -1
    else:
        edic[aa] = 0

def get_pqr_from_pdb(input_dir,output_dir):
    pdb2pqr = 'pdb2pqr'  # '/path/to/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
    apbsflag = '--whitespace --ff=AMBER --apbs-input'
    apbs = '/projectnb2/docking/imhaoyu/.conda/pkgs/apbs-1.5-hd8ae8d1_0/bin/apbs'
    # necessary!
    # https://groups.google.com/g/apbs-users/c/IFNWd_G42gc
    os.chdir(input_dir)
    for f in glob.glob(os.path.join(input_dir, "*.pdb")):
        print(f)
        pdb_basename = Path(f).stem
        command_format = f"pdb2pqr {apbsflag} {os.path.join(output_dir, pdb_basename + '.in')} {f} {os.path.join(output_dir, pdb_basename + '.pqr')}"
        try:
            os.system(command_format)
        except:
            print('pqr:' + f)
        # if f[-5]=='l':
        #    continue
        command_format = f"apbs {pdb_basename}.in"
        try:
            os.chdir(input_dir)
            os.system(command_format)
        except:
            print('apbs: ' + f)

def findvalue(dotcloud, gridsize, origin, delta, value):
    ind3d = np.floor((dotcloud - origin) / delta)
    ind1d = ind3d[:, 2] + ind3d[:, 1] * gridsize[2] + ind3d[:, 0] * gridsize[1] * gridsize[2]
    # print value.shape
    #     print ind3d
    # print value[ind1d.astype(int)]
    temp = value.reshape(gridsize)
    #     print temp[ind3d[:,0].astype(int),ind3d[:,1].astype(int),ind3d[:,2].astype(int)]
    return value[ind1d.astype(int)]


def getcontactbyabag(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('C', pdb_path)
    allchain=[]
    for chain in structure.get_chains():
        allchain.append(chain.get_id())
    newdick = []
    labeldick = [[],[]]
    for chain in allchain:
        try:
            test = structure[0][chain]
        except Exception as e:
            print(e)
            continue
        for resi in structure[0][chain]:
            if resi.get_resname() not in hdic.keys():
                continue
            cen = [0, 0, 0]
            count = 0
            for atom in resi:
                    # print atom.get_coord()
                    # print list(atom.get_vector())
                cen[0] += atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count += 1
            cen = [coor * 1.0 / count for coor in cen]
            newdick.append(cen)
            labeldick[0].append(hdic[resi.get_resname()])
            labeldick[1].append(edic[resi.get_resname()])
    return newdick, labeldick