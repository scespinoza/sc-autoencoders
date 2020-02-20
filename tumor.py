import pandas as pd
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from models import * 

if __name__ == '__main__':
    data = pd.read_csv('data/mgh26.txt', sep='\t', index_col=0)
    n_cells, n_genes = data.shape
    name='stacked_mgh26'

    early_stopping = callbacks.EarlyStopping(patience=args.patience)
    model_checkpoint = callbacks.ModelCheckpoint('weights/' + name + '_trained.h5',
                                                 save_best_only=True,
                                                 save_weights_only=True)
    
    plot_latent = PlotLatentSpace(model, dataset.data_scaled, dataset.class_labels, interval=args.interval)

    model = AutoEncoder(original_dim=n_genes, name=name)
    model.compile(optimizer=optimizers.Adam(0.0001), loss='binary_crossentropy')
    model.fit(data.values, data.values, epochs=1000, validation_data=(data.values, data.values),
             callbacks=[early_stopping, plot_latent, model_checkpoint])