import numpy as np
from typing import List

# To do --> implement the pytables for better data control
# Maybe refactor later?

class lazy_loader:

    def __init__(self, data_name: str, log_file: str):
        self.log_file = log_file
        self.data_name = data_name
        self.main_commands: List[str] = ["run", "minimize"]
        self.full_data: List[np.array] = []

    def read(self, log_file: str):
        reader = self.read_log_lammps(log_file)
        line = next(reader)
        for line in reader:
            if line[0] in self.main_commands:
                self.block_handler(reader, line)

    def save(self):
        np.savez(self.data_name, *self.full_data)

    def read_log_lammps(self, log_file: str):
        with open(log_file, 'r') as f:
            for raw_line in f:
                if raw_line.strip():
                    yield raw_line.split()

    def run_block_handler(self, gen_function, line_count: int):

        rows_number = line_count+1
        data = np.zeros((rows_number, 15))
        current_line = next(gen_function)

        while current_line[0] != "Step":
            current_line = next(gen_function)

        for data_row in range(rows_number):
            data[data_row, :] = next(gen_function)
        
        return data        

    def minimize_block_handler(self, gen_func):
        min_data = []
        current_line = next(gen_func)

        while current_line[0] != "Step":
            current_line = next(gen_func)

        current_line = next(gen_func)

        while current_line[0] != "Loop":
            if len(current_line) != 15:
                current_line = next(gen_func)
                continue
            min_data.append(current_line)
            current_line = next(gen_func)

        return np.array(min_data)

    def block_handler(self, gen_func: str, current_line: str):
        if current_line[0] == "run":
            self.full_data.append(self.run_block_handler(gen_func, int(current_line[1])))
        else:
            self.full_data.append(self.minimize_block_handler(gen_func))

