import os
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import GSE
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, output_notebook, show
from bokeh.models import CustomJS, Button
from bokeh.layouts import column
from bokeh.palettes import all_palettes


class ZILayer(layers.Layer):

    """
    TensorFlow 2 implementation of a Zero Inflated layer with gumbel sampling.
    See: https://doi.org/10.1016/j.gpb.2018.08.003 for reference

    Parameters
    ----------
    tau: float. 
        Initial tau for annealing procedure.
    Attributes
    ----------
    tau0: float
        Initial tau for annealing procedure.
    tau: float
        Current tau.
    tau_min: float
        Minimum tau for annealing procedure.

    """

    def __init__(self, tau=1., name='zi'):
        super(ZILayer, self).__init__(name=name)
        self.tau0 = tau
        self.tau = tau
        self.tau_min = 0.5

    def call(self, x):
        p = tf.exp(- x ** 2)
        q = 1 - p
        g0 = ZILayer.gumbel(shape=tf.shape(x))
        g1 = ZILayer.gumbel(shape=tf.shape(x))
        exp_p = tf.exp(tf.math.log(p + 1e-20) + g0 / self.tau)
        exp_q = tf.exp(tf.math.log(q + 1e-20) + g1 / self.tau)
        s = exp_p / (exp_p + exp_q)
        return s * x
    
    @classmethod
    def gumbel(cls, shape=None):
        """
        Sample from a gumbel distribution.
        
        Parameters
        ----------
        shape: tuple.
            Shape of the sampling vector.
        
        Returns
        -------
        gumbel_sample: array-like of shape 'shape'
            Gumbel samples.
        """
        eps=1e-20
        return -tf.math.log(-tf.math.log(tf.random.uniform(shape=shape) + eps) + eps)

class Encoder(layers.Layer):

    """
    TensorFlow 2 implementation of an Encoder suitable for AE, VAE and VaDE.

    Parameters
    ----------
    original_dim: int.
        Number of input dimensions.
    latent_dim: int.
        Number of dimensions of the latent space.
    name: str.
        Layer name.

    Attributes
    ----------
    h1: Layer.
        First hidden layer.
    h2: Layer.
        Second hidden layer.
    mu_dense: Layer.
        Layer for mu estimation
    logvar_dense: Layer.
        Layer for log(Var(z)) estimation
    """

    def __init__(self, original_dim=5491, latent_dim=10, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.h1 = layers.Dense(512, activation='relu')
        self.h2 = layers.Dense(128, activation='relu')
        self.mu_dense = layers.Dense(latent_dim, activation='linear')
        self.logvar_dense = layers.Dense(latent_dim, activation='linear', kernel_initializer='zeros')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        mu = self.mu_dense(x)
        logvar = self.logvar_dense(x)
        return mu, logvar

class Decoder(layers.Layer):

    """
    TensorFlow 2 implementation of an Encoder suitable for AE, VAE and VaDE.

    Parameters
    ----------
    original_dim: int.
        Number of input dimensions.
    latent_dim: int.
        Number of dimensions of the latent space.
    name: str.
        Layer name.

    Attributes
    ----------
    h1: Layer.
        First hidden layer.
    h2: Layer.
        Second hidden layer.
    h3: Layer.
        Second hidden layer.
    output: Layer.
        Output layer.
    """

    def __init__(self, original_dim=5491, latent_dim=10, name='Decoder'):
        super(Decoder, self).__init__(name=name)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.h1 = layers.Dense(latent_dim, activation='relu')
        self.h2 = layers.Dense(128, activation='relu')
        self.h3 = layers.Dense(512, activation='relu')
        self.outputs = layers.Dense(original_dim, activation='sigmoid')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        return self.outputs(x)

class SamplingLayer(layers.Layer):

    """
    TensorFlow 2 implementation of a sampling layer for VAE and VaDE.
    It implements the reparametrization trick z = mu + exp(logvar / 2) * eps
    """

    def __init__(self, name='Sampling'):
        super(SamplingLayer, self).__init__(name=name)

    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

class AutoEncoder(tf.keras.Model):

    """AutoEncoder
    TensorFlow 2 implementation of a stacked autoencoder.

    Parameters
    ----------
    original_dim: int, defaults to 5491.
        Number of input dimensions.
    latent_dim: int, defaults to 10.
        Number of latent dimensions.
    name: str, optional
        Model name.

    Attributes
    ----------
    encoder: Encoder,
        Encoder part of the AutoEncoder.
    decoder: Decoder,
        Decoder part of the AutoEncoder.
    original_dim: int
        Number of input dimensions.
    latent_dim: int
        Number of latent dimensions.
        
    """

    def __init__(self, original_dim=5491, latent_dim=10, name='AutoEncoder'):
        super(AutoEncoder, self).__init__(name=name)
        self.encoder = Encoder(original_dim=original_dim, latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim, latent_dim=latent_dim)
        self.original_dim = original_dim
        self.latent_dim = latent_dim

    def call(self, x):
        # This will throw a warning when training the stacked autoencoder. The call is implemented 
        # this way in order to reuse the code for the VAE an VaDE. The warning should not be an 
        # important issue when training the stacked autoencoder.
        z, _ = self.encoder(x)
        return self.decode(z)

    
    def encode(self, x):
        """
        Encode the input x.

        Parameters
        ----------
        x: array-like of shape (None, input_dim)
            Original data

        Returns
        -------
        mu, logvar: tuple of Tensors each of shape (None, latent_dim)
            Mean and Logvar for the z distribution. In the case of the stacked autoencoder,
            use only the Mean Tensor.
        """
        mu, logvar = self.encoder(x)
        return mu, logvar

    
    def decode(self, z):
        """
        Decode the latent vector z.

        Parameters
        ----------
        z: array-like of shape (None, latent_dim)

        Returns
        -------
        x_hat: Tensor of shape (None, original_dim)
        """
        return self.decoder(z)

    def reconstruction_loss(self, x, x_hat):
        """
        Reconstruction loss.
        
        Parameters
        ----------
        x: Tensor or array-like of shape (None, original_dim).
            Original Data.
        x_hat: Tensor or array-like of shape (None, original_dim).

        Returns
        -------
        reconstruction_loss: float.
            Reconstruction loss.

        """
        return self.original_dim * losses.binary_crossentropy(x, x_hat)

class VAE(AutoEncoder):
    """
    TensorFlow 2 implementation of a Variational AutoEncoder (VAE).

    Parameters
    ----------

    original_dim: int, defaults to 5491.
        Number of input dimensions.
    latent_dim: int, defaults to 10.
        Number of latent dimensions.
    name: str.
        Model Name.

    Attributes
    ----------
    encoder: Encoder,
        Encoder part of the AutoEncoder.
    decoder: Decoder,
        Decoder part of the AutoEncoder.
    original_dim: int
        Number of input dimensions.
    latent_dim: int
        Number of latent dimensions.
    sampling: SamplingLayer,
        Layer that contains the sampling procedure (reparametrization trick) of VAE.
    
    """

    def __init__(self, original_dim=5491, latent_dim=10, name='VAE'):
        super(VAE, self).__init__(original_dim=original_dim,
                                                     latent_dim=latent_dim,
                                                     name=name)
        self.sampling = SamplingLayer()

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling([mu, logvar])
        x_hat = self.decoder(z)
        kl_loss = self.vae_loss([x, mu, logvar, x_hat])
        self.add_loss(kl_loss)
        return x_hat

    
    def encode(self, x):
        """
        Encode the x vector.

        Parameters
        ----------
        x: Tensor or array-like of shape (None, input_dim)
            Original Data.
        
        Returns
        -------
        z: Tensor of shape (None, latent_dim)
            Latent encodings of x.
        """
        mu, logvar = self.encoder(x)
        return self.sampling([mu, logvar])
    
    
    def decode(self, z):
        """
        Decode the latent vector z.

        Parameters
        ----------
        z: Tensor or array-like of shape (None, latent_dim)
            Latent vector.
        Returns
        -------
        x_hat: Tensor of shape (None, original_dim)
            Reconstructions of the latent vector z.
        """
        return self.decoder(z)


    def vae_loss(self, inputs):
        """
        Variational AutoEncoder loss.

        Parameters
        ----------
        inputs: tuple of tensors (x, mu, logvar, x_hat).

        Returns
        vae_loss: float.
            Reconstruction loss + KL loss.
        """
        x, mu, logvar, x_hat = inputs
        reconstruction_loss = self.original_dim * losses.binary_crossentropy(x, x_hat)
        kl_loss = -0.5 * tf.reduce_sum((1 + logvar - tf.square(mu) - tf.exp(logvar)), axis=-1)
        return tf.reduce_mean(reconstruction_loss + kl_loss)

class VaDE(tf.keras.Model):
    """
    TensorFlow 2 implementation of a Variational Deep Embedding(VaDE).
    Based on: https://github.com/mperezcarrasco/Pytorch-VaDE

    Parameters
    ----------

    original_dim: int, defaults to 5491.
        Number of input dimensions.
    latent_dim: int, defaults to 10.
        Number of latent dimensions.
    n_components: int, defaults to 6.
        Number of components for the GMM.
    pretrain: int, defaults to 0.
        Number of epochs of pretraining. If 0, it assumes 
        that there is a file with already pretrained weights.
    pretrain_lr: float, defaults to 1e-4.
        Learning rate for the pretraining of the stacked autoencoder.
    k: float, defaults to 1.
        Weight for log(p(z|c)) in the loss function.
    search_k: bool, default to False.
        If true it search for the optimal number of components of the GMM
        in the stacked autoencoder latent space using a silhouette coefficient.
    name: str.
        Model Name.

    Attributes
    ----------
    original_dim: int
        Number of input dimensions.
    latent_dim: int
        Number of latent dimensions.
    n_components: int.
        Number of components of the GMM.
    gmm: GaussianMixture
        Instance of sklearn's GaussianMixture estimator.
    autoencoder: AutoEncoder,
        Base stacked autoencoder.
    pretrain: int, defaults to 0.
        Number of epochs of pretraining. If 0, it assumes 
        that there is a file with already pretrained weights.
    pretrain_lr: float, defaults to 1e-4.
        Learning rate for the pretraining of the stacked autoencoder.
    pi_prior: tf.Variable, of shape (n_components,)
        Values of prior p(c).
    mu_prior: tf.Variable of shape (n_components, latent_dim)
        Mean values of each GMM component in the latent space.
    logvar_prior: tf.Variable of shape (n_components, latent_dim)
        Log of the var values of each GMM component in the latent space.
    sampling: SamplingLayer,
        Layer that contains the sampling procedure (reparametrization trick) of VAE.
    k: float, defaults to 1.
        Weight for log(p(z|c)) in the loss function.
    search_k: bool, default to False.
        If true it search for the optimal number of components of the GMM
        in the stacked autoencoder latent space using a silhouette coefficient.
    """

    def __init__(self,
                 original_dim=5491,
                 latent_dim=10,
                 n_components=6,
                 pretrain=0,
                 pretrain_lr=0.0001,
                 k = 1,
                 search_k=False,
                 name='VariationalDeepEmbedding'):

        super(VaDE, self).__init__(name=name)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.gmm = None
        self.autoencoder = AutoEncoder(original_dim=original_dim, latent_dim=latent_dim)
        self.pretrain = pretrain
        self.pi_prior = tf.Variable(tf.ones(n_components) / tf.reduce_sum(tf.ones(n_components) ))
        self.mu_prior = tf.Variable(tf.zeros([n_components, latent_dim]))
        self.logvar_prior = tf.Variable(tf.ones([n_components, latent_dim]))
        self.sampling = SamplingLayer()
        self.pretrain_lr = pretrain_lr
        self.k = k
        self.search_k = search_k
        if not pretrain:
            try:
                self.load_pretrained()
            except OSError:
                print("Weights for {} not found.".format(self.name))
                print("Enabling pretrain")
                self.pretrain = 30

    def call(self, x):
        mu, logvar = self.autoencoder.encoder(x)
        z = self.sampling([mu, logvar])
        x_hat = self.autoencoder.decoder(z)
        kl_loss = self.vade_loss([x, mu, logvar, z, x_hat])
        self.add_loss(kl_loss)
        return x_hat

    @tf.function
    def encode(self, x):
        """
        Encode the x vector.

        Parameters
        ----------
        x: Tensor or array-like of shape (None, input_dim)
            Original Data.
        
        Returns
        -------
        z: Tensor of shape (None, latent_dim)
            Latent encodings of x.
        """
        mu, logvar = self.autoencoder.encoder(x)
        return self.sampling([mu, logvar])

    def predict_cluster(self, x):
        """
        Predict cluster with the GMM model.

        Parameters
        ----------
        x: Tensor or array-like of shape (None, input_dim)
            Original data.
        Returns
        -------
        classes: Tensor of shape (None,)
            Predicted classes.
        """
        z, _ = self.encode(x)
        gamma = self.compute_gamma(z)
        return tf.argmax(gamma, axis=1)

    def reconstruction_loss(self, x, x_hat):
        loss = self.original_dim * losses.binary_crossentropy(x, x_hat)
        return loss

    
    def vade_loss(self, inputs):
        """
        Variational AutoEncoder loss.

        Parameters
        ----------
        inputs: tuple of tensors (x, mu, logvar, z, x_hat).

        Returns
        vae_loss: float.
            Reconstruction loss + KL loss.
        """
        x, mu, logvar, z, x_hat = inputs
        p_c = self.pi_prior
        gamma = self.compute_gamma(z)
        log_p_x_z = self.original_dim * losses.binary_crossentropy(x, x_hat)
        h = tf.expand_dims(tf.exp(logvar), axis=1) + tf.pow(tf.expand_dims(mu, axis=1) - self.mu_prior, 2)
        h = tf.reduce_sum(self.logvar_prior + h / tf.exp(self.logvar_prior), axis=2)
        log_p_z_given_c = 0.5 * tf.reduce_sum(gamma * h, axis=1)
        log_p_c = tf.reduce_sum(gamma * tf.math.log(p_c + 1e-30), axis=1)
        log_q_c_given_x =  tf.reduce_sum(gamma * tf.math.log(gamma + 1e-30), axis=1)
        log_q_z_given_x = 0.5 * tf.reduce_sum(1 + logvar, axis=1)
        loss = tf.reduce_mean(log_p_x_z + self.k * log_p_z_given_c - log_p_c + log_q_c_given_x  - log_q_z_given_x)
        
        return loss

    def compute_gamma(self, z):
        """
        Compute q(c|x).

        Parameters
        ----------
        z: Tensor or array-like of shape (None, latent_dim)
            Latent vector.
        
        Returns
        -------
        gamma: Tensor of shape (None, n_components).
            q(c|x) probs for z latent vector.
        """
        p_c = self.pi_prior
        #print(p_c)
        h = ((tf.expand_dims(z, axis=1) - self.mu_prior)  ** 2) /  tf.exp(self.logvar_prior)
        h += self.logvar_prior
        h += tf.math.log(np.pi * 2)
        p_z_c = tf.exp(tf.expand_dims(tf.math.log(p_c + 1e-10), axis=0) - 0.5 * tf.reduce_sum(h, axis=2)) + 1e-30
        
        return p_z_c / tf.reduce_sum(p_z_c, axis=1, keepdims=True)

    def fit(self, X, y, **kwargs):
        """
        Implements pretrain and train routines for VaDE.

        Parameter
        ---------
        X: vector-like of shape (None, original_dim)
            Input.
        y: vector-like of shape (None, original_dim)
            Target.
        **kwargs: kwargs to pass to parent's fit function.

        Returns
        -------
        history: dict,
            History of loss and other metrics for each epoch.
        """
        if self.pretrain:
            self.autoencoder.compile(optimizer=optimizers.Adam(self.pretrain_lr), loss='binary_crossentropy')
            self.autoencoder.fit(X, X, epochs=self.pretrain)
            self.autoencoder.save_weights('weights/' + self.name + '_pretrained.h5')
            self.pretrain = False
        
        z = self.autoencoder.encode(X)
        print(self.n_components)
        self.fit_gmm(z.numpy())
        
        print('Training VaDE')
        
        history = super(VaDE, self).fit(X, y, **kwargs)
        return history
            

    def load_pretrained(self):
        """
        Load pretrained weights of the stacked autoencoder.
        """
        self.autoencoder.build(input_shape=(None, self.original_dim))
        self.autoencoder.load_weights('weights/' + self.name + '_pretrained.h5')

    def fit_gmm(self, X):
        """
        Fit Gaussian Mixture Model on the stacked autoencoder latent space.
        
        Parameters
        ----------
        X: Tensor or array-like of shape (None, original_dim)
            Data to fit GMM.
        """
        if self.search_k:
            print('Searching K')
            self.n_components = self.select_k(X)
        print("Fitting GMM with {} components.".format(self.n_components))
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag')
        self.gmm.fit(X)
        self.pi_prior = tf.Variable(self.gmm.weights_, dtype=tf.float32)
        self.mu_prior = tf.Variable(self.gmm.means_, dtype=tf.float32)
        self.logvar_prior = tf.Variable(np.log(self.gmm.covariances_), dtype=tf.float32)

    def select_k(self, X, klims=(2, 20)):
        """
        Search for optimal number of components for the GMM on the stacked autoencoder
        latent space.
                
        Parameters
        ----------
        X: Tensor or array-like of shape (None, original_dim)
            Data to fit GMM.
        klims: tuple of len 2.
            Min and max number of components to test.

        Returns
        -------
        best_k: best number of components according to silhouette score.
        """
        scores = {}
        for k in range(*klims):
            gmm = GaussianMixture(n_components=k, covariance_type='diag')
            labels = gmm.fit_predict(X)
            scores[k] = silhouette_score(X, labels)
            print('k = {}, score = {:.2f}'.format(k, scores[k]))
        return max(scores, key=scores.get)
            
class ZIAutoEncoder(AutoEncoder):

    """
    TensorFlow 2 implementation of a stacked autoencoder with Zero-Inflation layer.
    See: https://doi.org/10.1016/j.gpb.2018.08.003 for reference

    Parameters
    ----------
    dropout: float,
        Dropout rate for first hidden layer.
    tau: float,
        Initial temperature for the annealing procedure of the ZI Layer.
    *args: *args to pass to parent constructor.
    **kwargs: **kwargs to pass to parent constructor.
    """
    

    def __init__(self, dropout=0.5, tau=0.5, *args, **kwargs):

        super(ZIAutoEncoder, self).__init__( *args, **kwargs)
        self.dropout = layers.Dropout(dropout)
        self.zi = ZILayer(tau=tau)

    def call(self, x):
        z, _ = self.encode(x)
        return self.decode(z)
    
    
    def encode(self, x):
        x = self.dropout(x)
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return self.zi(x)

class ZIVAE(VAE):
    """
    TensorFlow 2 implementation of a VAE with Zero-Inflation layer.
    See: https://doi.org/10.1016/j.gpb.2018.08.003 for reference

    Parameters
    ----------
    dropout: float,
        Dropout rate for first hidden layer.
    tau: float,
        Initial temperature for the annealing procedure of the ZI Layer.
    *args: *args to pass to parent constructor.
    **kwargs: **kwargs to pass to parent constructor.
    """

    def __init__(self, dropout=0.5, tau=0.5, *args, **kwargs):

        super(ZIVAE, self).__init__( *args, **kwargs)
        self.autoencoder = ZIAutoEncoder(dropout=dropout, tau=tau, 
                                        original_dim=kwargs['original_dim'],
                                        latent_dim=kwargs['latent_dim'])

    def call(self, x):
        mu, logvar = self.autoencoder.encode(x)
        z = self.sampling([mu, logvar])
        x_hat = self.autoencoder.decoder(z)
        kl_loss = self.vae_loss([x, mu, logvar, x_hat])
        self.add_loss(kl_loss)
        return x_hat

class ZIVaDE(VaDE):
    """
    TensorFlow 2 implementation of a VaDE with Zero-Inflation layer.
    See: https://doi.org/10.1016/j.gpb.2018.08.003 for reference

    Parameters
    ----------
    dropout: float,
        Dropout rate for first hidden layer.
    tau: float,
        Initial temperature for the annealing procedure of the ZI Layer.
    *args: *args to pass to parent constructor.
    **kwargs: **kwargs to pass to parent constructor.
    """
    def __init__(self, dropout=0.5, tau=0.5, *args, **kwargs):

        super(ZIVaDE, self).__init__( *args, **kwargs)
        self.autoencoder = ZIAutoEncoder(dropout=dropout, tau=tau, 
                                        original_dim=kwargs['original_dim'],
                                        latent_dim=kwargs['latent_dim'])
        if not self.pretrain:
            try:
                self.load_pretrained()
            except OSError:
                print("Weights for {} not found.".format(self.name))
                print("Enabling pretrain")
                self.pretrain = 30

    def call(self, x):
        mu, logvar = self.autoencoder.encode(x)
        z = self.sampling([mu, logvar])
        x_hat = self.autoencoder.decoder(z)
        kl_loss = self.vade_loss([x, mu, logvar, z, x_hat])
        self.add_loss(kl_loss)
        return x_hat

class WarmUpCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to modify the weight of log(p(z|c)) in the VaDE loss function.
    EXPERIMENTAL
    """

    def __init__(self, k=0, b=0.001):
        self.k = k
        self.b = b

    def on_epoch_end(self, epoch, logs=None):
        if self.model.k < 1:
            self.model.k = self.k + epoch * self.b

class TauAnnealing(tf.keras.callbacks.Callback):

    """
    Keras callback to perform the tau annealing procedure on the ZI Layer.
    See: https://doi.org/10.1016/j.gpb.2018.08.003 for reference

    """

    def __init__(self, gamma=3e-4):
        self.gamma = gamma

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(self.model, ZIAutoEncoder):
            zi = self.model.zi
        else:
            zi = self.model.autoencoder.zi
        if epoch % 100 == 0:
            tau0 = zi.tau0
            tau = zi.tau
            tau_min = zi.tau_min
            new_tau = min(tau0 * np.exp(-self.gamma * epoch), tau_min)
            zi.tau = new_tau

class PlotLatentSpace(tf.keras.callbacks.Callback):
    """
    Keras callback to plot latent space using TSNE during training.

    Parameters
    ----------
    X: array-like of shape (None, original_dim).
        Data to plot on latent space.
    c: array-like of ints and shape (original_dim,).
        Encodings to color points on latent space.
    interval: int.
        Interval of trianing epochs to plot latent space.
    random_state: int.
        Random state for TSNE

    Attributes
    ----------
    X: array-like of shape (None, original_dim).
        Data to plot on latent space.
    c: array-like of ints and shape (original_dim,).
        Encodings to color points on latent space.
    interval: int.
        Interval of trianing epochs to plot latent space.
    random_state: int.
        Random state for TSNE
    """

    def __init__(self, X, c=None, interval=20, random_state=42):
        self.X = X
        self.c = c
        self.interval = interval
        self.random_state = random_state

    def plot(self, epoch, loss=None):
        """
        Plot latent space.

        Parameters
        ----------
        epoch: int,
            Current epoch.
        loss: dict,
            Loss dictionary.
        """
        loss = loss or 0.
        z = self.model.encode(self.X)
        if len(z) == 2:
            z = z[0]

        if isinstance(self.model, VaDE):
            z = np.concatenate([z, self.model.mu_prior.numpy()], axis=0)
            z_tsne = TSNE(random_state=self.random_state).fit_transform(z)

            cluster_means = z_tsne[-self.model.n_components:]
            z_tsne = z_tsne[:-self.model.n_components]
        else:
            z_tsne = TSNE(random_state=self.random_state).fit_transform(z)


        if isinstance(self.model, VaDE):
            fig, ax = plt.subplots(1, 2, figsize=(16, 9))
            title = 'epoch = {}, loss = {:.2f}'.format(epoch, loss)
            ax[0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=self.c, cmap='rainbow', alpha=0.6)
            ax[0].set_title('tumor')
            predicted_cluster = self.model.predict_cluster(self.X)
            ax[1].scatter(z_tsne[:, 0], z_tsne[:, 1], c=predicted_cluster, cmap='rainbow', alpha=0.6, s=5)
            ax[1].scatter(cluster_means[:, 0], cluster_means[:, 1], c='black', s=30, alpha=0.6)
            ax[1].set_title('predicted_cluster')
            fig.suptitle(title)
            fig.savefig('figures/' + self.model.name + "/epoch_{}.png".format(epoch))
            
            plt.close(fig)
        else:
            fig, ax = plt.subplots()

            title = 'epoch = {}, loss = {:.2f}'.format(epoch, loss)
            ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=self.c, cmap='rainbow', alpha=0.6, s=5)
            ax.set_title(title)
            fig.savefig('figures/' + self.model.name + "/epoch_{}.png".format(epoch))
            plt.close(fig)

    def on_train_begin(self, logs=None):
        try:
            os.mkdir('figures/' + self.model.name)
        except FileExistsError:
            pass

    def on_train_end(self, logs=None):
        try:
            self.model.load_weights('weights/' + self.model.name + '_trained.h5')      
            self.plot('last')
        except OSError:
            pass

    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.interval == 0:
            self.plot(epoch, logs['loss'])
          
class PrintLossAndAccuracy(tf.keras.callbacks.Callback):

    """
    Keras Callback to plot loss and custom accuracy in VaDE training.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict_cluster(self.x)
        y_true = self.y
        acc, w = self.compute_accuracy(y_true, y_pred)
        logs['test_metric'] = acc
        loss = logs['loss']
        val_loss = logs['val_loss']
        print()
        print('Epoch: {}, loss: {:.2f}, val_loss: {:.2f}, Acc: {:.2f}'.format(epoch, loss, val_loss, acc))
        #print('pi: ', self.model.pi_prior.numpy())
        z =self.model.encode(self.x[:1])
        gamma = self.model.compute_gamma(z).numpy()

        #print('gamma: ', gamma)
        #log_p_z_given_c = -0.5 * tf.reduce_sum(((self.model.mu_prior - z) ** 2) / tf.exp(self.model.logvar_prior), axis=1)
        #print('log p(c)p(z|c): ', log_p_z_given_c)
        #print('gamma: ', gamma)


    def compute_accuracy(self, y_true, y_pred):
        D = max(max(y_pred), max(y_true))+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in zip(*ind)])*1.0/len(y_pred)*100, w


models_dict = {
    'stacked': AutoEncoder,
    'vae': VAE,
    'vade': VaDE,
    'zi_stacked': ZIAutoEncoder,
    'zi_vae': ZIVAE,
    'zi_vade': ZIVaDE
}

def load_weights(dataset, model, class_name='', n_classes=0):
    """
    Helper function to load trained model.

    Parameters
    ----------
    dataset: string.
        A GSE dataset name.
    model: str one of {'stacked', 'vae', 'vade', 'zi_stacked', 'zi_vae', 'zi_vade'}
        String describing model to load.

        'stacked': Stacked AutoEncoder
        'vae': Variational AutoEncoder
        'vade': Variational Deep Embedding
        'zi_stacked': Zero-Inflated Stacked AutoEncoder
        'zi_vae': Zero-Inflated Variational AutoEncoder
        'zi_vade': Zero-Inflated Variational Deep Embedding
    class_name: str,
        Class name. Only for datasets GSE57872 and GSE84465.
    n_classes: int
        Number of classes of VaDE.

    Returns
    -------
    dataset: GSE object,
        Data for trained model.
    model: tf.keras.models.Model,
        Trained model.
    """
    name = dataset + '_' + class_name + '_' + model
    weights_filename = name + '_trained.h5'
    dataset = GSE(name=dataset, class_name=class_name)
    
    model = models_dict[model](original_dim=dataset.n_genes, name=name)
    model(dataset.data_scaled)
    model.load_weights('weights/' + weights_filename)
    return dataset, model

def plot_latent(dataset, model, cell_names=None, suffix='', ax=None, c=None):
    """
    Helper function to plot latent space from a dataset and model.

    Parameters
    ----------
    dataset: GSE object.
        Dataset to plot.
    model: tf.keras.Model,
        Model to generate latent space.
    cell_names: list of strings.
        Labels of cells to plot.
    suffix: str
        Suffix of output file name.
    c: array-like of ints, optional
        Encoding to colour points in latent space.

    Returns
    -------
    ax: axis
    """

    z = model.encode(dataset.data_scaled)

    if isinstance(z, tuple):
        z = z[0]

    if c is None:
        c = dataset.class_labels

    z_tsne = TSNE(random_state=42).fit_transform(z)


    TOOLTIPS=[
        ('cell_names', '@cell_names')
    ]
    cmap = plt.cm.get_cmap('rainbow')
    
    colors = [
        "#%02x%02x%02x" % tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i in (255 // max(c)) * c
    ]   

    output_file('bokeh_plots/' + model.name + suffix + ".html", title=model.name + ' - Latent Space', mode="cdn")

    output_notebook()

    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

    if cell_names is None:
        cell_names = list(dataset.cell_labels)

    source = ColumnDataSource(dict(
        x=list(z_tsne[:, 0]),
        y=list(z_tsne[:, 1]),
        colors=colors,
        cell_names=cell_names))

    savebutton = Button(label="Save", button_type="success")
    savebutton.callback = CustomJS(args=dict(source_data=source), code="""
        var inds = source_data.selected['1d'].indices;
        var data = source_data.data;
        var out = "";
        for (i = 0; i < inds.length; i++) {
            out += data['cell_names'][inds[i]] + ',\n';
        }
        var file = new Blob([out], {type: 'text/plain'});
        var elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(file);
        elem.download = 'selected-data.txt';
        document.body.appendChild(elem);
        elem.click();
        document.body.removeChild(elem);
        """)

    p = figure(plot_height=9 * 50, plot_width=16 * 50, tools=TOOLS, tooltips=TOOLTIPS)
    p.circle('x', 'y', fill_color='colors', fill_alpha=0.6, line_color=None, size=8, source=source)
    plot = column(p, savebutton)
    show(plot)
    

def plot_reconstructions(dataset, model, figsize=(16, 20)):

    """
    Helper function to plot reconstructions.

    Parameters
    ----------
    dataset: GSE object.
        Dataset to plot.
    model: tf.keras.Model,
        Model to generate latent space

    Returns
    -------
    ax: axis
    """

    x_hat = model.predict(dataset.data_scaled)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(dataset.data_scaled, ax=ax[0])
    sns.heatmap(x_hat, ax=ax[1])

    ax[0].set_title('Original Data')
    ax[1].set_title('Reconstructed Data')

    return ax
    