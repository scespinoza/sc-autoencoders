import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def get_gse84465_metadata(i, get='Cell_type'):
    plate_id, well = i.split('.')
    metadata = pd.read_csv('data/GSE84465_metadata.txt')
    info = metadata[(metadata['plate_id'] == int(plate_id)) & (metadata['well'] == well)]
    return info[get].iloc[0]

class GSE:


    

    sep_tumor = {
        'GSE57872': lambda i: i.split('_')[0],
        'GSE70630': lambda i: i.split('_')[0],
        'GSE89567': lambda i: i.split('_')[0],
        'GSE102130': lambda i: i.split('-')[0],

        # this dataset contains only one tumor
        'GSE132172_GliNS2': lambda i: i.split('_')[-1][0],
        'GSE84465': lambda i: get_gse84465_metadata(i),
        'GSE103224': lambda i: i.split('_')[0],
        'GSE131928_10x': lambda i: i.split('_')[0],
        'GSE131928_SmartSeq2': lambda i: i.split('-')[0]
        

    }

    def __init__(self, name='GSE57872'):
        self.name = name
        self.n_genes = None
        self.n_cells = None
        self.load()

    def load(self):
        if self.name == 'GSE103224':
            self.data = pd.read_csv('data/' + self.name + '.txt', sep='\t', index_col=1).drop("0", axis=1).T
        else:
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

    
if __name__ == '__main__':
    dataset = GSE('GSE84465')
    print(dataset.data.head())
    print(dataset.tumor_labels)