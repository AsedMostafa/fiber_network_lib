import numpy as np
import pandas as pd
from . import lazy_loader
from . import pulling_data

class prepareData:
    def __init__(self):
        self._columns = ['Step', 'Pxx', 'Pyy', 'Lx', 'Ly', 'Xlo', 'Xhi', 'Ylo', 'Yhi', 
            'PotEng', 'KinEng', 'Temp', 'v_max_bond_length', 'f_111[1]',
            'f_111[2]'] 
        
    def update_columns(self, columns):
        self._columns = columns

    def get_columns(self):
        return self._columns

    def load_to_memory(self, isLog: bool, **kwargs) -> pulling_data.pulling_data:
        if isLog:
            self.extract_log_data(kwargs['log_file'], kwargs['data_name'])
        raw_data = self.load_log_data(kwargs['data_name'])
        raw_data_prepared = self.prepare_data(raw_data)
        pulling_numbers = pulling_data.pulling_data(raw_data_prepared)
        pulling_numbers.pre_process(kwargs['pulling_direction'])
        return pulling_numbers

    def extract_log_data(self, log_file: str, data_name: str,):
        """
        Extracts the data from the log file
        data name: saved npz file
        log file: lammps log file path
        """
        data = lazy_loader.lazy_loader(data_name, log_file)
        data.read(log_file)
        data.save()

    def load_log_data(self,data_name: str):
        return np.load(data_name, allow_pickle=True)

    def prepare_data(self, data: np.lib.npyio.NpzFile) -> pd.DataFrame:
        captured_states = len(data)
        states_data = np.zeros((int(data[f'arr_{captured_states-1}'][0]), 15))

        i = 0
        for value in data.values():
            if value.shape[0] == 1:
                states_data[i, :] = value
                i += 1

        states_data = pd.DataFrame(states_data, columns=self._columns)

        return states_data

