import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder



class GSE:

    sep_tumor = {
        'GSE57872': lambda i: i.split('_')[0]
    }

    def __init__(self, name='GSE57872'):
        self.name = name
        self.n_genes = None
        self.n_cells = None

    def load(self):
        self.data = pd.read_csv('data/' + name + '.txt', sep='\t', index_col=0).T
        self.data_scaled = MinMaxScaler().fit_transform(data.values)
        self.cell_labels = data.index
        self.tumor_labels = data.index.map(GSE.sep_tumor)
        self.n_cells = data.shape[0]
        self.n_genes = data.shape[1]
        return self
