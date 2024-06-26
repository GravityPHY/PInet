import os
from Bio.PDB import *

# hfile=open('/Users/bowendai/Documents/ProteinEng/hydro.csv','r')
# hfile=open('/dartfs-hpc/rc/lab/C/CBKlab/Bdai/pythontools/hydro.csv','r')
hfile = open('hydro.csv', 'r')
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


def getcontactbyabag(folder, file, ab='', ag=''):
    if ab == '':
        i1 = file.find('-')
        i2 = file.find('-', i1 + 1)
        i3 = min(file.find('.pdb'), file.find('-', i2 + 1))
        ab = file[i1 + 1:i2]
        ag = file[i2 + 1:i3]

    atomset = ['CA', 'CB']
    chain = ab
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure_path = os.path.join(folder, file)
    structure = parser.get_structure('C', structure_path)
    allchain = []
    for c in ab:
        allchain.append(c)
    for c in ag:
        allchain.append(c)
    newdick = {}
    newdick['r'] = []
    newdick['l'] = []
    labeldick = {}
    labeldick['r'] = [[], []]
    labeldick['l'] = [[], []]

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
            if chain in ag:
                newdick['r'].append(cen)
                labeldick['r'][0].append(hdic[resi.get_resname()])
                labeldick['r'][1].append(edic[resi.get_resname()])
            else:
                newdick['l'].append(cen)
                labeldick['l'][0].append(hdic[resi.get_resname()])
                labeldick['l'][1].append(edic[resi.get_resname()])
    return newdick, labeldick, ab, ag


def gethydro(file):
    atomset = ['CA', 'CB']
    parser = PDBParser()
    structure = parser.get_structure('C', file)
    newdick = []
    labeldick = [[], []]

    for chain in structure[0]:
        for resi in chain:
            if resi.get_resname() not in hdic.keys():
                continue
            #             print resi.get_resname()
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
