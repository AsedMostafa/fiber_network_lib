import numpy as np

class FragmentFiber:
    def __init__(self, bonds, particles):

        self.bonds = bonds
        self.particles = particles
        self.n_bonds = bonds.shape[0]
        self.n_particles = particles.shape[0]
        self.y_mean = self.particles[:, 2].mean()


    def fragment(self, N_to_remove, decay_lambda=7):
        self.decay_lambda = decay_lambda
        self.bond_adj_map = self.build_adjacency_map()
        self.bonds_weight = self.get_bonds_weights(self.bonds[:, 0], self.bonds[:, 1])
        self.active_bonds = np.ones(self.n_bonds, dtype=bool)
        broken_indices = self.run_dynamic_breakage(N_to_remove)
        return broken_indices

    def get_distance_weights(self, p1, p2):
        y1, y2 = self.particles[p1, 2], self.particles[p2, 2]
        bond_center = (y1 + y2)/2.0
        dist_to_top = self.y_mean - np.abs(self.y_mean - bond_center)
        w_dist = np.exp(-dist_to_top / self.decay_lambda)

        return w_dist
    
    def get_coordinate_weights(self, p1, p2):
        c_total = self.particles[p1, -1] + self.particles[p2, -1]
        w_coord = 1.0 / (c_total**6)

        return w_coord

    def get_bonds_weights(self, p1, p2):
        w_dist = self.get_distance_weights(p1, p2)
        w_coord = self.get_coordinate_weights(p1, p2)
        return w_dist * w_coord

    def build_adjacency_map(self):
        bond_adj = {pid: [] for pid in range(self.n_particles)}
        for idx, (p1, p2) in enumerate(self.bonds):
            self.particles[p1, -1] += 1
            self.particles[p2, -1] += 1
            bond_adj[p1].append(idx)
            bond_adj[p2].append(idx)
        return bond_adj
    
    def run_dynamic_breakage(self, N_to_remove):
        """
        Efficiently breaks N bonds with updates after EVERY single break.
        """
        broken_indices = np.zeros(N_to_remove, dtype=int)

        for _ in range(N_to_remove):
            total_weight = np.sum(self.bonds_weight)
            if total_weight == 0: break 

            probs = self.bonds_weight / total_weight
            chosen_idx = np.random.choice(self.n_bonds, p=probs)

            self.active_bonds[chosen_idx] = False
            self.bonds_weight[chosen_idx] = 0.0
            broken_indices[_] = chosen_idx
            
            p1, p2 = self.bonds[chosen_idx]
            
            self.particles[p1, -1] -= 1
            self.particles[p2, -1] -= 1
            
            affected_bonds = set(self.bond_adj_map[p1] + self.bond_adj_map[p2])
            
            for b_idx in affected_bonds:
                if self.active_bonds[b_idx]: # Only update if it's still alive
                    self.bonds_weight[b_idx] = self.get_bonds_weights(self.bonds[b_idx, 0], self.bonds[b_idx, 1])

        return broken_indices