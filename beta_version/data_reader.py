import numpy as np
from elastin_fiber_utils import misc_func
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class LmpData:
    def __init__(self, data_name: str):
        self.n_atoms = None
        self.n_bonds = None
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
        self.atoms = np.zeros((self.n_atoms, 5))
        for i in range(self.n_atoms):
            data_line  = next(gen_func)
            self.atoms[i, 0:-1] = data_line[2:6]

    def bond_block_handler(self, gen_func):
        self.bonds = np.zeros((self.n_bonds, 4), dtype=np.int32)
        for i in range(self.n_bonds):
            data_line  = next(gen_func)
            self.bonds[i, 0:3] = data_line[1:]
        self.bonds[:, 1:3] -= 1


def get_color(particle_position, bonds):
    first_row = particle_position[bonds[:, 1], 1:3]
    second_row = particle_position[bonds[:, 2], 1:3]

    diff = second_row - first_row
    lengths = np.sqrt(np.sum(diff**2, axis=1))
    mask = lengths <= 1.5
    bonds = bonds[mask]

    palette = np.empty(4, dtype=object) 
    palette[1] = "#FF5400"
    palette[2] = "#A162E4"
    palette[3] = "#390099"
    colors = palette[bonds[:, 0]]     
    return colors

def get_particle_colors(particles):
    main_palette = np.empty(3, dtype=object) 
    edge_palette = np.empty(3, dtype=object) 
    main_palette[1] = "#FF5400"
    main_palette[2] = "#390099"
    edge_palette[1] = "#e0421f"
    edge_palette[2] = "#5000db"
    edge_colors = edge_palette[particles[:, 0].astype(int)]
    fill_colors = main_palette[particles[:, 0].astype(int)]   
    color_kw = {'color': fill_colors,
                'edgecolor': edge_colors}
    return color_kw


initial_structure = LmpData("12_connected.data")

bonds_info = misc_func.get_position(initial_structure.atoms,initial_structure.bonds[:, 1], initial_structure.bonds[:, 2])
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.set_aspect("equal") 

line_collection = LineCollection(bonds_info, zorder=0,
                                  colors = get_color(initial_structure.atoms, initial_structure.bonds), linewidths=1)
ax.add_collection(line_collection)
ax.axis('off')

ax.scatter(initial_structure.atoms[:, 1], initial_structure.atoms[:, 2], 
           s=10, zorder=1, lw=0.5, **get_particle_colors(initial_structure.atoms))
# plt.savefig("final_structure_rep_12_4.pdf", bbox_inches='tight')
plt.savefig("final_structure_rep_12_2.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()
