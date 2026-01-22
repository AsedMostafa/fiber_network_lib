import numpy as np
import pandas as pd

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
    