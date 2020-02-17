import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class GSE:

    sep_tumor = {
        'GSE57872': lambda i: i.split('_')[0]
    }

    def __init__(self, name='GSE57872'):
        self.name = name
        self.n_genes = None
        self.n_cells = None
        self.load()

    def load(self):
        self.data = pd.read_csv('data/' + self.name + '.txt', sep='\t', index_col=0).T
        self.data_scaled = MinMaxScaler().fit_transform(self.data.values)
        self.cell_labels = self.data.index
        self.tumor_labels = LabelEncoder().fit_transform(self.data.index.map(GSE.sep_tumor[self.name]))
        self.n_cells = self.data.shape[0]
        self.n_genes = self.data.shape[1]
        self.split()
        return self

    def split(self):
        x = self.data_scaled
        y = self.tumor_labels
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @property
    def train(self):
        return self.x_train, self.y_train

    @property
    def test(self):
        return self.x_test, self.y_test

    
