import numpy as np

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