import os
import random
import numpy as np
import argparse
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import rotmat, Vector

from sklearn import neighbors
from utils.parser import read_wrl2,parsefile
from utils.conversion import pdb_to_wrl,wrl_to_pts
from utils.features import get_pqr_from_pdb,findvalue,getcontactbyabag

def get_chain_names(pdb_path):
    """Get chain names from the structure."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    chain_names = set()
    for model in structure:
        for chain in model:
            chain_names.add(chain.id)
    return list(chain_names)

def get_ab_ag_names(ab_path,ag_path):
    ab_ch_list=get_chain_names(ab_path)
    ag_ch_list=get_chain_names(ag_path)
    ab_ch_str=" ".join(i for i in ab_ch_list)
    ag_ch_str=" ".join(i for i in ag_ch_list)
    return ab_ch_str,ag_ch_str

def add_EH_to_pts(pdb_id,pdb_path,pts_path,apbs_path,save_dir,type):
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
    np.savetxt(os.path.join(save_dir, f'{pdb_id}-{type}.pts'),
               np.concatenate((file, np.expand_dims(av, 1), np.expand_dims(pred[:, 0], 1)), axis=1))


def random_rotation_matrix():
    """Generate a random rotation matrix."""
    theta = random.uniform(0, 2 * np.pi)
    phi = random.uniform(0, np.pi)
    z = random.uniform(0, 2 * np.pi)

    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])

    return np.dot(Rz, np.dot(Ry, Rx))

def apply_rotation(structure, rotation_matrix):
    """Apply the given rotation matrix to the structure."""
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.transform(rotation_matrix, Vector(0, 0, 0))




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='point cloud preprocess')
    parser.add_argument('id', type=str, help='The input PDB id')

    # Parse the arguments
    args = parser.parse_args()
    pdbid=args.id
    #for pdbid in os.listdir("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/AGDB_labeled_epitopes/87cases/"):
        #if len(os.listdir(os.path.join("/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest",f"{pdbid}_bound")))!=2:
        #    continue


    ag_path=f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_l_b.pdb"
    ab_path=f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_r_b.pdb"
    ag_ch,ab_ch=get_ab_ag_names(ab_path, ag_path)

    pdb_dir = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound"
    output_folder = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound"
    pdb_to_wrl(pdb_dir, output_folder)
    wrl_to_pts(pdb_dir, output_folder)
    get_pqr_from_pdb(pdb_dir, output_folder)
    save_folder = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound"
    pdb_path = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_l_b.pdb"
    pts_path = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_l_b.pts"
    apbs_path = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_l_b.pqr.dx"
    add_EH_to_pts(pdbid, pdb_path, pts_path, apbs_path, save_folder, "l")

    save_folder = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound"
    pdb_path = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_r_b.pdb"
    pts_path = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_r_b.pts"
    apbs_path = f"/projectnb2/docking/imhaoyu/24_epitope_mapping/database/ABAGtest/{pdbid}_bound/{pdbid[0:4]}_r_b.pqr.dx"
    add_EH_to_pts(pdbid, pdb_path, pts_path, apbs_path, save_folder, "r")