import os
import numpy as np
import pandas as pd

from functools import reduce
from sklearn.preprocessing import MinMaxScaler

gene_counts = pd.read_csv('gene_counts.txt', index_col=0)
selected_genes = list(gene_counts[gene_counts['Count'] == 8].index)

excluded_datasets = ['GSE84465.txt', 'GSE132172_CB660.txt', 'GSE103224.txt']

def filter_df(df, selected_genes):
    df = df.loc[selected_genes]
    print('N° genes: {}, N° Cells: {}'.format(*df.shape))
    return df

def merge_datasets(dfs):
    full_data = reduce(lambda left, right: left.merge(right, right_index=True, left_index=True, how='inner'), dfs)
    return full_data

dfs = []

for filename in os.listdir('clean_data'):
    if (not 'GSE' in filename) or (filename in excluded_datasets):
        continue
    filepath = os.path.join('clean_data', filename)
    print('-' * 40)
    print(filename)
    if 'GSE103224' in filename:
        df = pd.read_csv(filepath, sep='\t', index_col=[0, 1]).groupby(level=1).mean()
        df = np.log2(df + 1)
        df = filter_df(df, selected_genes)
        df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df.values.T).T, index=df.index, columns=df.columns)
        
    else:
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        df = filter_df(df, selected_genes)
        columns = df.columns.map(lambda col_name: filename.split('.')[0] + '-' + col_name)
        df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df.values.T).T, index=df.index, columns=columns)

    dfs.append(df_scaled)
print('-' * 40)
print('-' * 40)
print('Merging Data...')
full_df = merge_datasets(dfs)
full_df.to_csv('clean_data/all_data.txt', sep='\t',)

    
