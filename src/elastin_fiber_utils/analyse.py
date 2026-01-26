import numpy as np
import pandas as pd
from . import dataloader 
import re

class Simulation:
    def __init__(self, meta_data: dict) -> None:
        self._meta_data = meta_data
        self.replica_holder: None|dict = None
    
    def is_converged(self):
        pass

    @property
    def data(self):
        return self.replica_holder
    
    def __getitem__(self, idx):
        if self.replica_holder is None:
            raise ValueError("No data is not loaded yet.")
        return self.replica_holder[idx]
    
    def __str__(self):
        to_be_printed = f'Simulation: {self._meta_data["name"]}\n with {self._meta_data["n_replicas"]} replicas\n and {self._meta_data["n_fragments"]} fragments\n'
        return to_be_printed

    def sort_addresses(self, path: str) -> tuple[int, int]:
        temp_replica = re.search(r'(\d+)(?=\.npz)', path)
        temp_ts = re.search(r'v?(\d+)', path)

        if not temp_replica or not temp_ts:
            raise ValueError(f"Invalid file name: {path}")
        
        return int(temp_ts.group(1)), int(temp_replica.group(1))
    
    def load_data(self):
        loaded_replica = self.load_replica()

    def load_replica(self):
        replica_holder = {ts: Replica() for ts in np.arange(1, self._meta_data['n_replicas'] + 1)}
        sorted_files = sorted(self._meta_data['npz_files'], key=self.sort_addresses)

        for path in sorted_files:
            temp_replica = re.search(r'(\d+)(?=\.npz)', path)
            if not temp_replica:
                raise ValueError(f"Invalid file name: {path}")
            
            frag_value = int(temp_replica.group(1))      
            loaded_data = dataloader.prepareData().load_to_memory(isLog=False, data_name=path, pulling_direction=self._meta_data['pulling_direction'])
            replica_holder[frag_value].append_data(loaded_data)

        return replica_holder


class Replica:
    def __init__(self, data_list: list|None = None):
        if data_list is None:
            self.datas = [] 
        else:
            self.datas = data_list
        self.n_replicas = 0
        self.all_stresses = pd.DataFrame()
        self._max_stress: None|np.ndarray = None

    def __len__(self):
        return self.n_replicas
    
    def __getitem__(self, idx):
        return self.datas[idx]
    
    @property
    def max_stress(self):
        self._max_stress = np.zeros(self.n_replicas)
        for idx, data in enumerate(self.datas):
            self._max_stress[idx] = data.max_stress
        return self._max_stress
    
    @property
    def failure_strains(self):
        self._failure_strains = np.zeros(self.n_replicas)
        for idx, data in enumerate(self.datas):
            self._failure_strains[idx] = data.failure_strain
        return self._failure_strains
    
    @property
    def toughness(self):
        self._toughness = np.zeros(self.n_replicas)
        for idx, data in enumerate(self.datas):
            self._toughness[idx] = data.toughness
        return self._toughness
    
    def append_data(self, data):
        self.datas.append(data)
        self.n_replicas = len(self.datas)

    def get_mean_stress(self):
        for idx, data in enumerate(self.datas):
            self.all_stresses[f'{idx}'] = data.data['stress']
        return self.all_stresses.mean(axis=1)
    
    def get_std(self):
        return self.all_stresses.std(axis=1)
  
    def get_strain(self):
        return self.datas[0].data['strain']
    
    def get_CI(self):
        return
    