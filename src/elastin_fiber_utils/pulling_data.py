class pulling_data:
    def __init__(self, data):
        self.data = data
        self.break_strain_index = None    
        self.elasticity_modulus = None
        self.toughness = None
        self.max_stress = None
        self.failure_strain = None

    def make_stress_strain(self, p_dir: str):
        pulling_direction_map = {'x': ['Lx', 'Pxx'], 'y': ['Ly', 'Pyy']}
        length = pulling_direction_map[p_dir][0]
        pressure = pulling_direction_map[p_dir][1]
        self.data['strain'] = (self.data[length] - self.data[length][0]) / self.data[length][0] 
        # self.data['strain'] = np.linspace(0, 1.5, 1500)       
      
        self.data['stress'] = -1 * self.data[pressure] * 101325 * 1e-9
        self.data.rename(columns={'f_111[2]': 'broken_bonds'}, inplace=True)

    def pre_process(self, pulling_direction: str):
        self.make_stress_strain(pulling_direction)
        self.break_strain_index = None    
        self.elasticity_modulus = self.get_elastisity(self.data)
        self.toughness = self.get_toughness(self.data)
        self.max_stress = self.data['stress'].max()
        self.failure_strain = self.data['strain'][self.data['stress'].idxmax()]

    def get_toughness(self, df):
        max_stress = df['stress'].max()
        idxss = df.index[df['stress'] < max_stress / 2]
        try:
            first_break = idxss[np.where(np.diff(idxss) > 1)[0][0]]
        except IndexError:
            return np.trapz(df['stress'], x=df['strain'])
        neg_idx = idxss[first_break+1:]
        if not neg_idx.empty:
            cutoff = neg_idx[0]   # take the first negative index
            df = df.loc[:cutoff-1]
            self.break_strain_index = cutoff
        return np.trapz(df['stress'], x=df['strain'])

    def get_elastisity(self, df):
        idxss = df.index[df['strain'] < 0.7]
        df = df.loc[idxss]
        slope = (df['strain']*df['stress']).sum() / (df['strain'] * df['strain']).sum()
        return slope
