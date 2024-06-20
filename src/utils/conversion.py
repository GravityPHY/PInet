import os
import glob
import numpy as np
from pathlib import Path

import pymol
from pymol import cmd, stored

from sklearn import neighbors
from Bio.PDB import *

from utils.parser import read_wrl2


def pdb_to_wrl(input_dir,output_dir):
    """
    convert all pdb under input_dir to wrl, and save them in the output_dir
    :param input_dir: path to an input directory
    :param output_dir: path to an output directory, usually the directory to save your files
    :return: None
    """
    files = glob.glob(os.path.join(input_dir, "*.pdb"))
    assert len(files)!=0, f"No pdb found in {input_dir}"
    for f in files:
        cmd.load(os.path.join(input_dir, f))
        cmd.set('surface_quality', '0')
        cmd.show_as('surface', 'all')
        cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
        cmd.save(os.path.join(output_dir, os.path.basename(f).replace('.pdb','.wrl')))
        cmd.delete('all')
    print(f"Files are saved in {os.path.abspath(output_dir)}")


def wrl_to_pts(input_dir,output_dir):
    """
    convert all wrl under input_dir to pts, and save them in output_dir
    :param input_dir: a path of directory contains wrl files
    :param output_dir: a path of directory to save pts files
    :return:
    """
    wrl_files = glob.glob(os.path.join(input_dir, "*.wrl"))
    assert len(wrl_files) != 0, f"No wrl file found in {input_dir}"
    for f in wrl_files:
        wrl_basename = Path(f).stem
        vb, _, cb, nb = read_wrl2(f)
        vecb = np.unique(vb, axis=0)
        np.savetxt(os.path.join(output_dir, f"{wrl_basename}.pts"), np.vstack((vecb)), delimiter=' ')