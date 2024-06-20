import re
import numpy as np
from numpy import matrix, savetxt

def read_wrl2(filename, verbose=0):
    """
    load a mesh from a VRML file

    wrl format:
    Coordinate{ point [0 0 0, 1 1 1]} PointSet, IndexedLineSet, Indexed FaceSet
    :param filename: path to VRML file
    :param verbose:
    :return:
    np.array: coord: coordinates of 3D point vertices
    np.array: face: Faceset
    np.array: color: color of vertices
    np.array: normal: surface unit normal vectors
    """
    with open(filename, 'r') as fp:

        tempstr = ' '
        nodeCount = 0
        coord = []
        face = []
        color = []
        normal = []
        ln = ['a']
        endfile = False

        while not endfile:
            line = fp.readline()
            if len(line) == 0:
                endfile = True
                break
            ln = line.split()
            # point coordinates
            if (ln != []) and (ln[0] == 'point'):
                if verbose:
                    print('Reading vertex coordinates.')
                end_section = False
                while not end_section:
                    lns = fp.readline().split(',')
                    for tris in lns:
                        ln = tris.split()
                        if len(ln) == 0:
                            continue
                        if ln[0] == ']':
                            # coord = matrix(coord).astype(float)
                            end_section = True
                            break
                        if len(ln) > 2:
                            # ln[2] = ln[2][:-1]  # remove comma
                            coord.append(np.asarray(ln, dtype=float))

            # faces
            if (ln != []) and (ln[0] == 'coordIndex'):
                if verbose: print('Reading faces indexes.')
                while 1:
                    line = fp.readline()
                    ln = list(filter(None, re.split(' |,|\n', line)))
                    if ln == [']']:
                        # face = matrix(face)
                        break
                    if len(ln) > 2:
                        for i_ln in range(0, len(ln), 4):
                            face.append(np.asarray(ln[i_ln:i_ln + 3], dtype=int))
            # color
            if (ln != []) and (ln[0] == 'color'):

                if verbose: print('Reading color vertex.')
                end_section = False
                while not end_section:
                    lns = fp.readline().split(',')
                    for tris in lns:
                        ln = tris.split()
                        if len(ln) == 0:
                            continue
                        if ln[0] == ']':
                            # coord = matrix(coord).astype(float)
                            end_section = True
                            break
                        if len(ln) > 2:
                            # ln[2] = ln[2][:-1]  # remove comma
                            color.append(np.asarray(ln, dtype=float))

            # normal
            if (ln != []) and (ln[0] == 'vector') and ln[1] == '[':
                if verbose: print('Reading normal vertex.')
                while 1:
                    ln = fp.readline().split()
                    if ln[0] == ']':
                        # normal = matrix(normal)
                        endfile = True
                        break
                    if len(ln) > 2:
                        ln[-1] = ln[-1][:-1]
                        normal.append(np.asarray(ln[0:3], dtype=float))
    return np.array(coord), np.array(face), np.array(color), np.array(normal)


def parsefile(file):
    gridsize = []
    origin = []
    delta = [0, 0, 0]
    value = []
    for line in file:
        if line[0] == '#':
            continue
        if line[0:2] == 'ob':
            if line[7] == '1':
                # print literal_eval(line[line.find('counts')+7:])
                # break
                gridsize = [int(d) for d in line[line.find('counts') + 7:].split(' ')]
                # print gridsize
                continue
                # break
            if line[7] == '2' or line[7] == '3':
                continue
            #     print delta
            #     break
        if line[0:2] == 'or':
            origin = [float(d) for d in line[7:].split(' ')]
            # print origin
            continue
            # break
        if line[0:2] == 'de':
            for i, d in enumerate(line[6:].split(' ')):
                delta[i] += float(d)
            continue
        if line[0:3] == 'att':
            break
        # print line.strip().split(' ')
        value.extend([float(d) for d in line.strip().split(' ')])
        # for v in [float(d) for d in line.strip().split(' ')]:
        #     if v<0:
        #         print 1

    # print len(value)
    gridsize = np.asarray(gridsize)
    origin = np.asarray(origin)
    delta = np.asarray(delta)
    value = np.asarray(value)

    return gridsize, origin, delta, value