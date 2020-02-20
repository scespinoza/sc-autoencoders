import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def get_metadata(dataset, column='Cell_type'):
    metadata = pd.read_csv('data/{}_metadata.txt'.format(dataset))
    return metadata[column]

class GSE:


    

    get_classes = {
        'GSE57872': lambda i: i.split('_')[0],
        'GSE70630': lambda i: i.split('_')[0],
        'GSE89567': lambda i: i.split('_')[0],
        'GSE102130': lambda i: i.split('-')[0],

        # this dataset contains only one tumor
        'GSE132172_GliNS2': lambda i: i.split('_')[-1][0],
        'GSE84465': get_gse84465_metadata,
        'GSE103224': lambda i: i.split('_')[0],
        'GSE131928_10x': lambda i: i.split('_')[0],
        'GSE131928_SmartSeq2': lambda i: i.split('-')[0]
        

    }

    def __init__(self, name='GSE57872', class_name=None):
        self.name = name
        self.n_genes = None
        self.n_cells = None
        self.class_name = class_name
        self.load()

    def load(self):
        if self.name == 'GSE103224':
            self.data = pd.read_csv('data/' + self.name + '.txt', sep='\t', index_col=1).drop("0", axis=1).T
        else:
            self.data = pd.read_csv('data/' + self.name + '.txt', sep='\t', index_col=0).T
        self.data_scaled = MinMaxScaler().fit_transform(self.data.values)
        self.cell_labels = self.data.index

        if self.name in ['GSE84465', 'GSE84465']:
            self.class_labels = LabelEncoder().fit_transform(GSE.get_classes[self.name](self.name, self.class_name))
        else:
            self.class_labels = LabelEncoder().fit_transform(self.data.index.map(GSE.get_classes[self.name]))
            
        self.n_cells = self.data.shape[0]
        self.n_genes = self.data.shape[1]
        self.split()
        return self

    def split(self):
        x = self.data_scaled
        y = self.class_labels
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