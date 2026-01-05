import numpy as np
import pandas as pd
from . import lazy_loader
from . import pulling_data

COULMNS = ['Step', 'Pxx', 'Pyy', 'Lx', 'Ly', 'Xlo', 'Xhi', 'Ylo', 'Yhi', 
            'PotEng', 'KinEng', 'Temp', 'v_max_bond_length', 'f_111[1]',
            'f_111[2]'] 

def extract_log_data(log_file: str, data_name: str,):
    """
    Extracts the data from the log file
    data name: saved npz file
    log file: lammps log file path
    """
    data = lazy_loader.lazy_loader(data_name, log_file)
    data.read(log_file)
    data.save()

def load_log_data(data_name: str):
    return np.load(data_name, allow_pickle=True)

def prepare_data(data: np.ndarray) -> pd.DataFrame:
    captured_states = len(data)
    states_data = np.zeros((int(data[f'arr_{captured_states-1}'][0]), 15))

    i = 0
    for value in data.values():
        if value.shape[0] == 1:
            states_data[i, :] = value
            i += 1

    states_data = pd.DataFrame(states_data, columns=COULMNS)

    return states_data

def disk_to_memory_workflow(npz_lib_path: str, pulling_direction: str) -> pulling_data.pulling_data:

    raw_data = load_log_data(npz_lib_path)
    raw_data_prepared = prepare_data(raw_data)
    pulling_numbers = pulling_data.pulling_data(raw_data_prepared)
    pulling_numbers.pre_process(pulling_direction)

    return pulling_numbers