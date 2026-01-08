import numpy as np

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
        self.atoms: np.ndarray = np.zeros(1)
        self.bonds: np.ndarray = np.zeros(1)
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