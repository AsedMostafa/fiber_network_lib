from ovito.io import import_file, export_file
from ovito.modifiers import ClusterAnalysisModifier
from ovito.modifiers import ExpressionSelectionModifier
from ovito.modifiers import DeleteSelectedModifier
import numpy as np


class OvitoHelper:
    def __init__(self, input_file, output_file, atom_style='angle'):
        self._input_file = input_file
        self._output_file = output_file
        self._atom_style = atom_style
    
    def set_input_file(self, input_file):
        self._input_file = input_file

    def set_output_file(self, output_file):
        self._output_file = output_file

    def load_data(self):
        return import_file(
            self._input_file,
            multiple_frames=False,
            atom_style=self._atom_style)
    
    def export_data(self, data):
        export_file(
            data,
            self._output_file,
            "lammps/data",
            atom_style=self._atom_style,
            ignore_identifiers=True
        )

    def do(self, command):

        pipeline = self.load_data()

        command_list = {'delete_free_particles': self.delete_free_particles,
                        'delete_bonds': self.delete_bonds}

        pipeline = command_list[command](pipeline)
        data = pipeline.compute()

        self.export_data(data)

        
    def delete_free_particles(self, pl):

        cluster = ClusterAnalysisModifier(neighbor_mode=ClusterAnalysisModifier.NeighborMode.Bonding, sort_by_size=True)
        pl.modifiers.append(cluster)
        pl.modifiers.append(ExpressionSelectionModifier(expression = 'Cluster != 1', operate_on='particles'))
        pl.modifiers.append(DeleteSelectedModifier())
        return pl


    def delete_bonds(self, pl):

        pl.modifiers.append(ExpressionSelectionModifier(expression = 'BondType == 1', operate_on='bonds'))
        pl.modifiers.append(DeleteSelectedModifier())
        return pl

def yeoh_incompressible(l, c1, c2, c3):
    I1 = l**2 + 1/l**2 + 1
    term = I1 - 3
    stress = 2*(c1+ 2*c2*term + 3*c3*(term)**2)*(l**2 - 1/l**2)
    return stress

def get_position(particles_data, p1, p2):
    first_row = particles_data[p1, 1:3]
    second_row = particles_data[p2, 1:3]

    diff = second_row - first_row
    lengths = np.sqrt(np.sum(diff**2, axis=1))
    mask = lengths <= 1.5

    first_row = first_row[mask]
    second_row = second_row[mask]

    coor = np.stack((first_row, second_row), axis=1)
    return coor