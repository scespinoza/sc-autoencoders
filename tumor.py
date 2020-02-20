import pandas as pd
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from models import * 

if __name__ == '__main__':
    data = pd.read_csv('data/mgh26.txt', sep='\t', index_col=0)
    n_cells, n_genes = data.shape
    name='stacked_mgh26'

    model = AutoEncoder(original_dim=n_genes, name=name)

    data_scaled = MinMaxScaler().fit_transform(data.values)
    early_stopping = callbacks.EarlyStopping(patience=50)
    model_checkpoint = callbacks.ModelCheckpoint('weights/' + name + '_trained.h5',
                                                 save_best_only=True,
                                                 save_weights_only=True)

    metadata = pd.read_csv('data/GSE57872_metadata.txt')
    classes = LabelEncoder().fit_transform(metadata[metadata['patient_id'] == 'MGH26']['subtype'])
    
    plot_latent = PlotLatentSpace(model, data_scaled, classes, interval=20)

    
    model.compile(optimizer=optimizers.Adam(0.0001), loss='binary_crossentropy')
    model.fit(data_scaled, data_scaled, epochs=1000, validation_data=(data_scaled, data_scaled),
             callbacks=[early_stopping, plot_latent, model_checkpoint])