import time
import os
import numpy as np
import copy

def load_movement(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        dofs = eval(lines[0])
        postures = []
        for line in lines[1:]:
            angles = eval(line[:-1])
            posture = {
                dof : angle for dof, angle in zip(dofs, angles)
            }
            postures.append(posture)
        return postures
    raise(BaseException(filename+" does not exist"))

