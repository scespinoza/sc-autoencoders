import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import umap

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.losses import mse
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers import scikit_learn
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.activations import softplus
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import he_normal

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, output_notebook, show


class DenseTied(layers.Layer):

    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        
        self.biases = self.add_weight(name='bias', initializer='zeros', shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.kernel, transpose_b=True)
        return self.activation(z + self.biases)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense': self.dense,
            'activation': self.activation
        })


class AutoEncoderManager:
    DIM_RED = {
        'tSNE': TSNE,
        #'UMAP': umap.UMAP
    }

    def __init__(self):
        self.encoder = None
        self.decoder = None

    def reduce(self, X, method='tSNE'):

        if isinstance(self, AutoEncoder):
            encoded = self.encoder(X)

            if method in AutoEncoder.DIM_RED.keys():
                reducer = AutoEncoder.DIM_RED[method](random_state=42)
                reduced = reducer.fit_transform(encoded)
            else:
                print('Method not implemented')
        
        elif isinstance(self, VariationalAutoEncoder):
            _, _, reduced = self.encoder(X)
        return reduced

    def save_model(self):
        filepath = 'models/' + self.name
        model_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        full_path = filepath + '-' + model_date
        os.mkdir(full_path)
        history = self.history.history        
        self.save_weights(full_path + '/weights.h5')
        pd.DataFrame(history).to_csv(full_path + '/history.log', index=False)
        return full_path

    def plot_history(self, ax=None):
        ax = ax or plt.gca()

        if self.restored:
            history = self.saved_history
            ax.plot(history['loss'], label='training loss')
            ax.plot(history['val_loss'], label='validation loss')
            plt.legend()
        else:
            history = self.history
            ax.plot(history.history['loss'], label='training loss')
            ax.plot(history.history['val_loss'], label='validation loss')
            plt.legend()
        
        ax.set_title('Training History, Model: ' + self.name)

    def plot_reconstructions(self, X, figsize=(12, 4)):
        original = X
        reconstructions = self.predict(X)

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        sns.heatmap(original, ax=ax[0])
        ax[0].set_title('Original Data')
        sns.heatmap(reconstructions, ax=ax[1])
        ax[1].set_title('Reconstructed Data')

    def cluster(self, X, method='tSNE', cluster_range=(2, 9), cell_names=None):
        reduced = self.reduce(X, method=method)        
        cluster_scores = {}

        for n_clusters in range(*cluster_range):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(reduced)
            cluster_scores[n_clusters] = silhouette_score(reduced, cluster_labels)
            print('{} clusters: {}'.format(n_clusters, cluster_scores[n_clusters]))
            
        cluster_scores = pd.Series(cluster_scores)
        optimal_clusters = cluster_scores.idxmax()

        print('Optimal Clusters: {}'.format(optimal_clusters))

        cluster_labels = KMeans(n_clusters=optimal_clusters).fit_predict(reduced)

        return cluster_labels, reduced

    def plot_clusters(self, X, method='tSNE', cell_names=None):

        if isinstance(self, VariationalAutoEncoder):
            method = 'VAE'

        cluster_labels, reduced = self.cluster(X, method=method)
        columns = [method + '1', method + '2']
        if not cell_names is None:
            data = np.c_[cell_names, reduced]
            columns.insert(0, 'cell')            
        else:
            data = reduced
        
        df = pd.DataFrame(data, columns=columns)
        df['cluster'] = cluster_labels
        plot_dimensionality_reduction(df, x=method + '1', y=method + '2', title='{} Clustering, model: {}'.format(method, self.name))

class AutoEncoder(tf.keras.Sequential, AutoEncoderManager):
    
    def __init__(self, 
                 input_dim=2048,
                 hidden_dim=[128],
                 latent_dim=16, 
                 initializer='he_normal', 
                 regularizer=l2(1e-6), 
                 learning_rate=0.01,
                 va=False,
                 name='autoencoder'):

        encoder_layers = []
        encoder_layers_dense = []
        decoder_layers = []
        
        # encoder
        for i, hidden_n in enumerate(hidden_dim):
            
            if i == 0:
                # specify input dimension for the first layer
                encoder_layers += [layers.Dense(hidden_n,
                                         activation=None,
                                         kernel_initializer=initializer, 
                                         kernel_regularizer=regularizer, 
                                         input_shape=(input_dim,),
                                         name='input_layer'),
                                    layers.LeakyReLU(name='input_relu')]    
            else:
                encoder_layers_dense += [layers.Dense(hidden_n,
                                         activation=None, 
                                         kernel_initializer=initializer,
                                         kernel_regularizer=regularizer,
                                         name='encoder_dense_{}'.format(i))]
                encoder_layers += [encoder_layers_dense[-1], layers.LeakyReLU(name='encoder_relu_{}'.format(i))]

        
        
        
        encoder_layers += [layers.Dense(latent_dim, 
                           activation=tf.keras.activations.linear,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer)]
        
        
        # decoder

        for i, hidden_n in enumerate(hidden_dim[::-1]):
            decoder_layers += [layers.Dense(hidden_n,
                                     activation=None,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='decoder_dense_{}'.format(i + 1)), LeakyReLU(name='decoder_relu_{}'.format(i))]

        decoder_layers += [layers.Dense(input_dim,
                                 activation='sigmoid', 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer,
                                 name='output')]

        super().__init__(encoder_layers + decoder_layers, name='AE-' + name)

        self.encoder = tf.keras.Sequential(encoder_layers, name='encoder')         
        self.decoder = tf.keras.Sequential(decoder_layers, name='decoder')
        self.decoder.build(input_shape=(None, latent_dim))

        self.restored = False
        
    @classmethod
    def build_autoencoder(cls,
                          input_dim=2048, 
                          hidden_dim=[128], 
                          latent_dim=16, 
                          initializer='he_normal', 
                          regularizer=l2(1e-6),
                          optimizer='adam',
                          learning_rate=0.01,
                          name='autoencoder'):

        tf.keras.backend.clear_session()
        model =  cls(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    initializer=initializer,
                    regularizer=regularizer,
                    learning_rate=learning_rate,
                    name=name)

        optimizer = optimizers.get(optimizer)(learning_rate)
        loss = losses.get(loss)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
        return model

   
    
    @classmethod
    def load_model(cls, filepath, load_data=False, input_dim=2048):
        name = filepath.split('/')[-1]
        print(name)
        model = cls(name=name, input_dim=input_dim)
        model.load_weights(filepath + '/weights.h5')
        model.saved_history = pd.read_csv(filepath + '/history.log')
        model.restored = True
        if load_data:
            data = pd.read_csv(filepath + '/data.csv', index_col=0)
            return model, data
        else:
            return model


class VariationalLayer(layers.Layer):
    
    def __init__(self, units):
        
        super(VariationalLayer, self).__init__()
        self.z_mean_layer = layers.Dense(units, activation=None)
        self.z_logvar_layer = layers.Dense(units, activation=None)
        
    def call(self, inputs):
        z_mean = self.z_mean_layer(inputs)
        z_logvar = self.z_logvar_layer(inputs)
        #self.kl_loss([z_mean, z_logvar])
        return z_mean, z_logvar
        
        
class SamplingLayer(layers.Layer):
    
    def __init__(self):
        super(SamplingLayer, self).__init__()
        
    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean), mean=0., stddev=0.1)
        return (z_mean + tf.exp(z_logvar * 0.5) * eps) * 0.1
    

class ZeroInflatedLayer(layers.Layer):
    
    def __init__(self, tau=0.5):
        super(ZeroInflatedLayer, self).__init__()
        self.tau = tau

    def call(self, inputs):
        p = tf.exp(- inputs ** 2)
        q = 1 - p
        g0 = ZeroInflatedLayer.gumbel_sampling(shape=tf.shape(inputs))
        g1 = ZeroInflatedLayer.gumbel_sampling(shape=tf.shape(inputs))
        exp_p = tf.exp(tf.math.log(p + 1e-20) + g0 / self.tau)
        exp_q = tf.exp(tf.math.log(q + 1e-20) + g1 / self.tau)
        s = exp_p / (exp_p + exp_q)
        return s * inputs

    @classmethod
    def gumbel_sampling(cls, shape=None):
        eps = 1e-10
        return -tf.math.log(-tf.math.log(tf.random.uniform(shape=shape) + eps) + eps)

class Encoder(layers.Layer):

    def __init__(self, input_dim, latent_dim=2, name='encoder'):

        super(Encoder, self).__init__(name=name)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.dropout = layers.Dropout(0.5)
        self.dense1 = layers.Dense(512, activation='linear', kernel_regularizer=l1(0.01))
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(32, activation='relu')
        self.variational = VariationalLayer(latent_dim)
        self.sampling = SamplingLayer()

    def call(self, inputs):

        x = self.dropout(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        z_mean, z_logvar = self.variational(x)
        z = self.sampling([z_mean, z_logvar])

        return z_mean, z_logvar, z


class Decoder(layers.Layer):

    def __init__(self, input_dim, latent_dim=2, tau=0.5, name='decoder'):
        super(Decoder, self).__init__(name=name)
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(512, activation='relu')
        self.dense4 = layers.Dense(input_dim, activation='sigmoid')
        self.zi = ZeroInflatedLayer(tau=tau)
    
    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.zi(x)
        return x

class VariationalAutoEncoder(tf.keras.models.Model, AutoEncoderManager):

    def __init__(self,
                input_dim=2048,
                latent_dim=2,
                tau=0.5,
                name='model'):
        
        super(VariationalAutoEncoder, self).__init__(name='VAE-' + name)
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim, tau=tau)
        self.restored = False


    def call(self, inputs):
        z_mean, z_logvar, z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        self.compute_loss([inputs, reconstructions, z_mean, z_logvar])
        return reconstructions

    def compute_loss(self, inputs):
        x, reconstructions, z_mean, z_logvar = inputs
        reconstruction_loss = self.input_dim * mse(x, reconstructions)
        kl_loss = -0.5 * tf.reduce_mean(z_logvar - tf.square(z_mean) - tf.exp(z_logvar) + 1)
        self.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
    
    @classmethod
    def vae_loss(cls):
        def dummy_loss(y_true, y_pred):
            return 0.
        return dummy_loss

    @classmethod
    def build_vae(cls, input_dim=2048, latent_dim=2, tau=0.5, optimizer=Adam, learning_rate=0.001):
        model = cls(input_dim=input_dim,
                    latent_dim=latent_dim,
                    tau=0.5)

        model.compile(optimizer=Adam(learning_rate), loss=VariationalAutoEncoder.vae_loss())
        return model

    @classmethod
    def load_model(cls, filepath, load_data=False, input_dim=2048, latent_dim=2):
        name = filepath.split('/')[-1]
        print(name)
        model = cls(name=name, input_dim=input_dim, latent_dim=2)
        model.load_weights(filepath + '/weights.h5')
        model.saved_history = pd.read_csv(filepath + '/history.log')
        model.restored = True
        if load_data:
            data = pd.read_csv(filepath + '/data.csv', index_col=0)
            return model, data
        else:
            return model


def plot_dimensionality_reduction(df, x, y, title=""):

    
    radius = max(df[x].max() - df[x].min(), df[y].max() - df[y].min()) * 0.005
    TOOLTIPS=[
        ('cell_name', '@cell')
    ]

    output_notebook()

    cmap = plt.cm.get_cmap('rainbow')
    if 'tumor_ids' in df.columns:
        colors = ["#%02x%02x%02x" % tuple((np.array(cmap(n)[:3]) * 255).astype(int)) for n in (df['tumor_ids'] / len(df['tumor_ids'].unique()))]
        df['colors'] = colors
    elif 'cluster' in df.columns:
        colors = ["#%02x%02x%02x" % tuple((np.array(cmap(n)[:3]) * 255).astype(int)) for n in (df['cluster'] / len(df['cluster'].unique()))]
        df['colors'] = colors
    else:
        df['colors'] = 'blue'
        
    source = ColumnDataSource(df)
    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

    p = figure(plot_height=9 * 50, plot_width=16 * 50, tools=TOOLS, tooltips=TOOLTIPS, title=title)
    p.circle(x=x,
             y=y,
             source=source,
             fill_color='colors',
             radius=radius, 
             line_color=None, 
             fill_alpha=0.4,
             hover_color='yellow')

    show(p, notebook_handle=True)

def random_projection(data, n_dim=2048):
    """
    Select random features (genes) from a scRNAseq gene expression matrix.

    Paramters
    ---------
    data: array, shape(n_cells, n_genes)
        Gene expression matrix.
    n_dim: int 
        Number of genes to select.

    Returns
    -------
    data_projected: array, shape(n_cells, n_dim)
        Randomly projected expression matrix.
    """
    m, n = data.shape
    selected = [True] * n_dim + [False] * (n - n_dim)

    np.random.shuffle(selected)
    data_projected = data.loc[:, selected]
    return data_projected

if __name__ == '__main__':
    data = pd.read_csv('clean_data/GSE57872.txt', sep='\t', index_col=0).T

    tumor_ids = data.index.map(lambda i: i.split('_')[0])
    colors = LabelEncoder().fit_transform(tumor_ids)
    X = MinMaxScaler().fit_transform(data)
    X_proj = random_projection(X)
    X_train, X_test = train_test_split(X_proj)

    print('Train Set = {}'.format(X_train.shape[0]))
    print('Test Set = {}'.format(X_test.shape[0]))

    early_stopping = EarlyStopping(patience=10)

    model = Autoencoder.build_autoencoder(learning_rate=0.0001)
    model.fit(X_train, X_train, validation_data=(X_test, X_test), callbacks=[early_stopping], epochs=1)

    history = model.history

   
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()
    
    y_tsne = model.reduce(X_proj)
    plt.figure()
    plt.scatter(y_tsne[:, 0], y_tsne[:, 1], c=colors, cmap='rainbow')
    cb = plt.colorbar()
    cb.ax.set_title('Tumor ID')
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.title('tSNE dimensionality reduction')
    plt.show()

    plt.figure()
    y_umap = model.reduce(X_proj, method='UMAP')
    plt.figure()
    plt.scatter(y_umap[:, 0], y_umap[:, 1], c=colors, cmap='rainbow')
    cb = plt.colorbar()
    cb.ax.set_title('Tumor ID')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP dimensionality reduction')

    plt.show()