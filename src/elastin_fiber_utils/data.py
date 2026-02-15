import numpy as np
import pandas as pd
from . import lazy_loader
from . import pulling_data

class LmpData:
    """
    This holds a lammps data file while providing methods to manipulate it
    Methods:
        read_file(self, file: str)
        load_data(self, file: str)
        block_handler(self, gen_func, block_name: str)
        atom_block_handler(self, gen_func)
        bond_block_handler(self, gen_func)
        write_lammps_file(self, file_name: str)
    """
    def __init__(self, data_name: str):
        """
        Initializes the LmpData object with the given data_name.

        Parameters
        ----------
        data_name : str
            path of the lammps data file

        Attributes
        ----------
        n_atoms : int
            number of atoms in the data file
        n_bonds : int
            number of bonds in the data file
        boundaries : dict
            dictionary of boundaries for x, y, and z
        blocks : list
            list of blocks in the data file
        handlers : dict
            dictionary of handlers for each block
        """
        self.n_atoms: int = 0
        self.n_bonds: int = 0
        self.atoms: np.ndarray|None = None
        self.bonds: np.ndarray|None = None
        self.boundaries = {'x': [],
                           'y': [],
                           'z': []}
        self.blocks = ['Atoms', 'Bonds']
        self.handlers = {'Atoms': self.atom_block_handler,
                         'Bonds': self.bond_block_handler}
        self.load_data(data_name)

    def read_file(self, file: str):
        with open(file, 'r') as f:
            for raw_line in f:
                if raw_line.strip():
                    yield raw_line.split()

    def load_data(self, file: str):
        reader = self.read_file(file)

        atom_set = False
        bonds_set = False

        while True:
            line = next(reader)
            if line[1] == "atoms":
                self.n_atoms = int(line[0])
                atom_set = True
            if line[1] == "bonds":
                self.n_bonds = int(line[0])
                bonds_set = True
            if atom_set and bonds_set:
                break
            
        while True:
            line = next(reader)
            if line[2:4] == ["xlo", "xhi"]:
                self.boundaries['x'] = [float(line[0]), float(line[1])]
                line = next(reader)
                self.boundaries['y'] = [float(line[0]), float(line[1])]
                line = next(reader)
                self.boundaries['z'] = [float(line[0]), float(line[1])]
                break

        while True:
            try:
                line = next(reader)
                if line[0] in self.blocks:
                    self.block_handler(reader, line[0])
            except StopIteration:
                break

    def block_handler(self, gen_func, block_name: str):
        self.handlers[block_name](gen_func)

    def atom_block_handler(self, gen_func):
        self.atoms = np.zeros(shape = (self.n_atoms, 5))
        for i in range(self.n_atoms):
            data_line  = next(gen_func)
            self.atoms[i, 0:-1] = data_line[2:6]

    def bond_block_handler(self, gen_func):
        self.bonds = np.zeros((self.n_bonds, 4), dtype=np.int32)
        for i in range(self.n_bonds):
            data_line  = next(gen_func)
            self.bonds[i, 0:3] = data_line[1:]
        self.bonds[:, 1:3] -= 1

    def write_lammps_file(self, file_name):
        if self.atoms is None or self.bonds is None:
            raise ValueError("Atoms or bonds not loaded")
        
        dtype = dict(zip(['id', 'molecule', 'type','x', 'y', 'z'], [0, 1, 2, 3, 4, 5]))
        natoms = self.n_atoms
        ntypes = 2
        atoms = np.zeros((natoms,6))
        atoms[:,dtype.get('id')] = np.arange(1,natoms+1)
        atoms[:,dtype.get('molecule')] = np.ones((natoms,1)).reshape(-1)
        atoms[:, dtype.get('x')] = self.atoms[:, 1]
        atoms[:, dtype.get('y')] = self.atoms[:, 2]
        atoms[:, dtype.get('z')] = 0
        bonds = np.zeros((self.n_bonds, 4), dtype=np.int32)
        bonds[:, 0] = np.arange(1,self.n_bonds+1)
        bonds[:, 1:4] = self.bonds[:, 0:3]
        bonds[:, 2:4] += 1
        outfile = file_name + '.data'

        with open(outfile, "w") as outfile:
            outfile.write("LAMMPS data")
            outfile.write("\n%d atoms\n" % natoms)
            outfile.write("\n%d atom types\n" % ntypes)
            outfile.write("\n%d bonds\n" % self.n_bonds)
            outfile.write("\n%d bond types\n" % 3)

            outfile.write("\n%12.5E %12.5E xlo xhi\n" % (self.boundaries['x'][0], self.boundaries['x'][1]))
            outfile.write("%12.5E %12.5E ylo yhi\n" % (self.boundaries['y'][0], self.boundaries['y'][1]))
            outfile.write("%12.5E %12.5E zlo zhi\n" % (-0.5, 0.5))

            outfile.write("\nMasses\n\n")
            for i in range(ntypes):
                outfile.write("%5d\t%9.3E\n" % (i+1,1))
            outfile.write("\nAtoms # bond\n\n")
            for i in range(natoms):
                    outfile.write("%5d\t%d\t%d\t%10.3f\t%10.3f\t%10.3f\n" % (atoms[i,dtype.get('id')],
                                                                                atoms[i,dtype.get('molecule')],
                                                                                1,
                                                                                atoms[i,dtype.get('x')],
                                                                                atoms[i,dtype.get('y')],
                                                                                atoms[i,dtype.get('z')]))
            outfile.write("\nBonds\n\n")
            for j in range(self.n_bonds):
                outfile.write("%5d\t%5d\t%5d\t%5d\n" % (bonds[j,0], bonds[j,1], bonds[j,2], bonds[j,3]))


class FragmentFiber:
    def __init__(self, lmp_data):

        self.bonds = lmp_data.bonds[:, 1:3]
        self.particles = lmp_data.atoms
        self.n_bonds = lmp_data.n_bonds
        self.n_particles = lmp_data.n_bonds
        self.y_mean = self.particles[:, 2].mean()


    def fragment(self, N_to_remove, decay_lambda=7):
        self.decay_lambda = decay_lambda
        self.bond_adj_map = self.build_adjacency_map()
        self.bonds_weight = self.get_bonds_weights(self.bonds[:, 0], self.bonds[:, 1])
        self.active_bonds = np.ones(self.n_bonds, dtype=bool)
        broken_indices = self.run_dynamic_breakage(N_to_remove)
        return broken_indices

    def get_distance_weights(self, p1, p2):
        y1, y2 = self.particles[p1, 2], self.particles[p2, 2]
        bond_center = (y1 + y2)/2.0
        dist_to_top = self.y_mean - np.abs(self.y_mean - bond_center)
        w_dist = np.exp(-dist_to_top / self.decay_lambda)

        return w_dist
    
    def get_coordinate_weights(self, p1, p2):
        c_total = self.particles[p1, -1] + self.particles[p2, -1]
        w_coord = 1.0 / (c_total**6)

        return w_coord

    def get_bonds_weights(self, p1, p2):
        w_dist = self.get_distance_weights(p1, p2)
        w_coord = self.get_coordinate_weights(p1, p2)
        return w_dist * w_coord

    def build_adjacency_map(self):
        bond_adj = {pid: [] for pid in range(self.n_particles)}
        for idx, (p1, p2) in enumerate(self.bonds):
            self.particles[p1, -1] += 1
            self.particles[p2, -1] += 1
            bond_adj[p1].append(idx)
            bond_adj[p2].append(idx)
        return bond_adj
    
    def run_dynamic_breakage(self, N_to_remove):
        """
        Efficiently breaks N bonds with updates after EVERY single break.
        """
        broken_indices = np.zeros(N_to_remove, dtype=int)

        for _ in range(N_to_remove):
            total_weight = np.sum(self.bonds_weight)
            if total_weight == 0: break 

            probs = self.bonds_weight / total_weight
            chosen_idx = np.random.choice(self.n_bonds, p=probs)

            self.active_bonds[chosen_idx] = False
            self.bonds_weight[chosen_idx] = 0.0
            broken_indices[_] = chosen_idx
            
            p1, p2 = self.bonds[chosen_idx]
            
            self.particles[p1, -1] -= 1
            self.particles[p2, -1] -= 1
            
            affected_bonds = set(self.bond_adj_map[p1] + self.bond_adj_map[p2])
            
            for b_idx in affected_bonds:
                if self.active_bonds[b_idx]: # Only update if it's still alive
                    self.bonds_weight[b_idx] = self.get_bonds_weights(self.bonds[b_idx, 0], self.bonds[b_idx, 1])

        return broken_indices
    

class PrepareData:
    def __init__(self):
        self._columns = ['Step', 'Pxx', 'Pyy', 'Lx', 'Ly', 'Xlo', 'Xhi', 'Ylo', 'Yhi', 
            'PotEng', 'KinEng', 'Temp', 'v_max_bond_length', 'f_111[1]',
            'f_111[2]'] 
        
    @property
    def columns(self):
        return self._columns
    @columns.setter
    def columns(self, columns):
        self._columns = columns

    def load_to_memory(self, isLog: bool, **kwargs) -> pulling_data.pulling_data:
        if isLog:
            self.extract_log_data(kwargs['log_file'], kwargs['data_name'])
        raw_data = self.load_log_data(kwargs['data_name'])
        raw_data_prepared = self.prepare_data(raw_data)
        pulling_numbers = pulling_data.pulling_data(raw_data_prepared)
        pulling_numbers.pre_process(kwargs['pulling_direction'])
        return pulling_numbers

    def extract_log_data(self, log_file: str, data_name: str):
        """
        Extracts the data from the log file
        data name: saved npz file
        log file: lammps log file path
        """
        data = lazy_loader.lazy_loader(data_name, log_file, len(self._columns))
        data.read(log_file)
        data.save()

    def load_log_data(self,data_name: str):
        return np.load(data_name, allow_pickle=True)

    def prepare_data(self, data: np.lib.npyio.NpzFile) -> pd.DataFrame:
        captured_states = len(data)
        states_data = np.zeros((int(data[f'arr_{captured_states-1}'][0]), len(self._columns)))

        i = 0
        for value in data.values():
            if value.shape[0] == 1:
                states_data[i, :] = value
                i += 1

        states_data = pd.DataFrame(states_data, columns=self._columns)

        return states_data

