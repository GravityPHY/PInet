import os
import sys
import numpy as np
from Bio.PDB import *

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
    print(len(r_file))
    print(len(c))

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
        
if __name__ == "__main__":
    save_dir="/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest"
    for folder in os.listdir(save_dir):
        pdb_id=folder[0:4]
        pdb_file=os.path.join(os.path.join(save_dir,folder),f"{pdb_id}_l_b.pdb")
        res_file=os.path.join(os.path.join(save_dir,folder),f"{pdb_id}_resi_l.seg")
        save_path=os.path.join(save_dir,folder)
        set_bfactor(pdb_file, res_file, save_path)