import numpy as np
import pandas as pd
from . import data
import re
import os
from collections.abc import Iterator
from pathlib import Path

class Simulation:
    def __init__(self, meta_data: dict) -> None:
        self._meta_data = meta_data
        self.replica_holder: None|dict = None
        self.prepare_metadata()
    def is_converged(self):
        pass

    def get_files(self, root_dir: Path, _pattern: str) -> list[Path]:
        files: Iterator[Path] = root_dir.glob(pattern=_pattern)
        sorted_files = sorted(files, key=self.sort_replicas)
        return sorted_files

    def prepare_metadata(self):
        os.chdir(self._meta_data['root_path'])
        root_directory = Path(self._meta_data['root_path'])
        self._meta_data['npz_files'] = self.get_files(root_directory, self._meta_data['pattern'])
        self._meta_data['n_total'] = 0
        self._details = {}
        for p in self._meta_data['npz_files']:
            if p.parent.name not in self._details:
                self._details[p.parent.name] = 1
            else:
                self._details[p.parent.name] += 1
            self._meta_data['n_total'] += 1
        
        how_many_frags = np.array([i for i in self._details.values()])
        if np.unique(how_many_frags).shape[0] > 1:
            raise ValueError("Not all replicas have the same number of fragments.")

        self._meta_data['n_replicas'] = how_many_frags[0]

    @property
    def data(self):
        return self.replica_holder
    
    def sort_replicas(self, path: Path) -> tuple[str, float]: 
        return path.parent.name, float(path.stem)
    
    def __getitem__(self, idx):
        if self.replica_holder is None:
            raise ValueError("No data is not loaded yet.")
        return self.replica_holder[idx]
    
    def __str__(self):
        to_be_printed = f'Simulation: {self._meta_data["name"]}\n with {self._meta_data["n_replicas"]} replicas\n and {self._meta_data["n_fragments"]} fragments\n'
        return to_be_printed

    def load_replica(self):
        '''
        You can implemenet change in the number of the columns here
        Maybe in the future
        '''


        self.replica_holder = {ts: Replica() for ts in np.arange(1, self._meta_data['n_replicas'] + 1)}
        data_handler = data.PrepareData()
        for path in self._meta_data['npz_files']:
            frag_value = int(float(path.stem))
            if int(path.parent.name) > 16:
                self._meta_data['stress_factor'] = 2
            else:
                self._meta_data['stress_factor'] = 1

            if self._meta_data['stress_factor']:
                loaded_data = data_handler.load_to_memory(isLog=False, data_name=path, 
                                                          pulling_direction=self._meta_data['pulling_direction'], stress_factor=self._meta_data['stress_factor'])
            else:    
                loaded_data = data_handler.load_to_memory(isLog=False, data_name=path, pulling_direction=self._meta_data['pulling_direction'])
            self.replica_holder[frag_value].append_data(loaded_data)


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
    
    @property
    def elasticity_modulus(self):
        self._elasticity_modulus = np.zeros(self.n_replicas)
        for idx, data in enumerate(self.datas):
            self._elasticity_modulus[idx] = data.elasticity_modulus
        return self._elasticity_modulus
    
    @property
    def hyperelastic_params(self):
        self._hyperelastic_params = np.zeros((self.n_replicas, 3))
        for idx, data in enumerate(self.datas):
            self._hyperelastic_params[idx] = data.params['yeoh_incompressible']

        return self._hyperelastic_params
    
    def append_data(self, data):
        self.datas.append(data)
        self.n_replicas = len(self.datas)

    def get_mean_stress(self):
        for idx, data in enumerate(self.datas):
            if data.stress.shape[0] > 2500:
                self.all_stresses[f'{idx}'] = data.stress[:2500]
            else:
                tempelate = np.zeros(2500)
                tempelate[:data.stress.shape[0]] = data.stress
                self.all_stresses[f'{idx}'] = tempelate
        return self.all_stresses.mean(axis=1)
    
    def get_std(self):
        return self.all_stresses.std(axis=1)
  
    def get_strain(self):
        return self.datas[0].strain[:2500]
    
    def get_CI(self):
        return
    