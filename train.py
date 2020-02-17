from 
from preprocess import GSE
from models import AutoEncoder

if __name__ == '__main__':
    dataset = GSE(name='GSE57872')
    model = AutoEncoder(original_dim=dataset.n_genes)