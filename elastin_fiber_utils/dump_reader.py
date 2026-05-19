import numpy as np
import re

class dump_reader:
    def __init__(self, filelist):
        self.data = {}
        for file in filelist:
            ts = re.split('_|\\.', file)[-2]
            with open(file, "r") as f:
                print("reading {}".format(file))
                snap_shot =self.read_snapshot(f)
                self.data[int(ts)] = snap_shot.data

    def read_snapshot(self, f):
        a_snap = snap()
        f.readline()
        a_snap.time_step = int(f.readline().split()[0])
        f.readline()
        a_snap.number_of_entries = int(f.readline().split()[0])
        a_snap.number_of_entries
        for _ in range(5):
            f.readline()
        a_snap.data = np.zeros((a_snap.number_of_entries, 5))
        for i in range(a_snap.number_of_entries):
            a_snap.data[i, :] = f.readline().split()
        return a_snap


class snap():
    pass