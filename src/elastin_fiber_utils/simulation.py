import numpy as np
from replica import Replica
import elastin_fiber_utils as efu
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
            loaded_data = efu.dataloader.prepareData().load_to_memory(isLog=False, data_name=path, pulling_direction=self._meta_data['pulling_direction'])
            replica_holder[frag_value].append_data(loaded_data)

        return replica_holder




