import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt


class Encoder(layers.Layer):

    def __init__(self, original_dim=5491, latent_dim=10, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.h1 = layers.Dense(2048, activation='relu')
        self.h2 = layers.Dense(512, activation='relu')
        self.mu_dense = layers.Dense(latent_dim, activation='linear')
        self.logvar_dense = layers.Dense(latent_dim, activation='linear', kernel_initializer='zeros')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        mu = self.mu_dense(x)
        logvar = self.logvar_dense(x)
        return mu, logvar

class Decoder(layers.Layer):

    def __init__(self, original_dim=5491, latent_dim=10, name='Decoder'):
        super(Decoder, self).__init__(name=name)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.h1 = layers.Dense(latent_dim, activation='relu')
        self.h2 = layers.Dense(518, activation='relu')
        self.h3 = layers.Dense(2048, activation='relu')
        self.outputs = layers.Dense(original_dim, activation='sigmoid')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        return self.outputs(x)

class SamplingLayer(layers.Layer):

    def __init__(self, name='Sampling'):
        super(SamplingLayer, self).__init__(name=name)

    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps



class AutoEncoder(tf.keras.Model):

    def __init__(self, original_dim=5491, latent_dim=10, name='AutoEncoder'):
        super(AutoEncoder, self).__init__(name=name)
        self.encoder = Encoder(original_dim=original_dim, latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim, latent_dim=latent_dim)
        self.original_dim = original_dim
        self.latent_dim = latent_dim

    def call(self, x):
        x, _ = self.encoder(x)
        return self.decoder(x)

    def encode(self, x):
        z, _ = self.encoder(x)
        return z.numpy()

    def reconstruction_loss(self, x, x_hat):
        return self.original_dim * losses.binary_crossentropy(x, x_hat)


class VariationalAutoEncoder(AutoEncoder):

    def __init__(self, original_dim=5491, latent_dim=10, name='VariationalAutoEncoder'):
        super(VariationalAutoEncoder, self).__init__(original_dim=original_dim,
                                                     latent_dim=latent_dim,
                                                     name=name)
        self.sampling = SamplingLayer()

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling([mu, logvar])
        kl_loss = self.vae_loss([mu, logvar])
        self.add_loss(kl_loss)
        return self.decoder(z)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.sampling([mu, logvar])

    def vae_loss(self, inputs):
        mu, logvar = inputs
        loss = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        return tf.reduce_mean(loss)


class VariationalDeepEmbedding(tf.keras.Model):

    def __init__(self,
                 original_dim=5491,
                 latent_dim=10,
                 n_components=6,
                 pretrain=True,
                 batch_size=32,
                 name='VariationalDeepEmbedding'):

        super(VariationalDeepEmbedding, self).__init__(name=name)
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
        self.batch_size = batch_size
        if not pretrain:
            try:
                self.load_pretrained()
            except OSError:
                print("Weights for {} not found.".format(self.name))
                print("Enabling pretrain")
                self.pretrain = True

    def call(self, x):
        mu, logvar = self.autoencoder.encoder(x)
        z = self.sampling([mu, logvar])
        x_hat = self.autoencoder.decoder(z)
        kl_loss = self.vade_loss([mu, logvar, z])
        self.add_loss(kl_loss)
        return x_hat

    def encode(self, x):
        mu, logvar = self.autoencoder.encoder(x)
        return self.sampling([mu, logvar])

    def predict_cluster(self, x):
        z = self.encode(x)
        gamma = self.compute_gamma(z)
        return tf.argmax(gamma, axis=1)

    def reconstruction_loss(self, x, x_hat):
        loss = self.original_dim * losses.binary_crossentropy(x, x_hat)
        return loss

    
    def vade_loss(self, inputs):
        mu, logvar, z = inputs
        p_c = self.pi_prior
        gamma = self.compute_gamma(z)
        h = tf.expand_dims(tf.exp(logvar), axis=1) + tf.pow(tf.expand_dims(mu, axis=1) - self.mu_prior, 2)
        h = tf.reduce_sum(self.logvar_prior + h / tf.exp(self.logvar_prior), axis=2)
        log_p_z_given_c = 0.5 * tf.reduce_sum(gamma * h, axis=1)
        log_p_c = tf.reduce_sum(gamma * tf.math.log(p_c + 1e-10), axis=1)
        log_q_c_given_x = tf.reduce_sum(gamma * tf.math.log(gamma + 1e-10), axis=1)
        log_q_z_given_x = 0.5 * tf.reduce_sum(1 + logvar, axis=1)
        loss = tf.reduce_mean(log_p_z_given_c - log_p_c + log_q_c_given_x  - log_q_z_given_x)
        
        return loss

    def compute_gamma(self, z):

        p_c = self.pi_prior
        #print(p_c)
        h = (tf.expand_dims(z, axis=1) - self.mu_prior)  ** 2 /  tf.exp(self.logvar_prior)
        h += self.logvar_prior
        h += tf.math.log(np.pi * 2)
        p_z_c = tf.exp(tf.expand_dims(tf.math.log(p_c + 1e-10), axis=0) - 0.5 * tf.reduce_sum(h, axis=2)) + 1e-10
        
        return p_z_c / tf.reduce_sum(p_z_c, axis=1, keepdims=True)

    def fit(self, X, y, **kwargs):
        if self.pretrain:
            self.autoencoder.compile(optimizer=optimizers.Adam(0.0001), loss='binary_crossentropy')
            self.autoencoder.fit(X, X, epochs=self.pretrain)
            self.autoencoder.save_weights('weights/' + self.name + '_pretrained.h5')
            self.pretrain = False
        print("Fitting GMM")
        z, _ = self.autoencoder.encoder(X)
        self.fit_gmm(z.numpy())
        
        print('Training VaDE')
        history = super(VariationalDeepEmbedding, self).fit(X, y, **kwargs)
        return history
            

    def load_pretrained(self):
        self.autoencoder.build(input_shape=(None, self.original_dim))
        self.autoencoder.load_weights('weights/' + self.name + '_pretrained.h5')

    def fit_gmm(self, X):
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag')
        self.gmm.fit(X)
        self.pi_prior.assign(self.gmm.weights_)
        self.mu_prior.assign(self.gmm.means_)
        self.logvar_prior.assign(np.log(self.gmm.covariances_))

class PlotLatentSpace(tf.keras.callbacks.Callback):

    def __init__(self, model, X, c=None, interval=20):
        self.X = X
        self.c = c
        self.model = model
        self.interval = interval

    def plot(self, epoch, loss=None):
        z = self.model.encode(self.X)
        z = np.concatenate([z, self.model.mu_prior.numpy()], axis=0)
        z_tsne = TSNE().fit_transform(z)

        cluster_means = z_tsne[-6:]
        z_tsne = z_tsne[:-6]

        if isinstance(self.model, VariationalDeepEmbedding):
            fig, ax = plt.subplots(1, 2, figsize=(16, 9))
            title = 'epoch = {}, loss = {:.2f}'.format(epoch, loss)
            ax[0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=self.c, cmap='rainbow', alpha=0.6)
            ax[0].set_title('tumor')
            predicted_cluster = self.model.predict_cluster(self.X)
            ax[1].scatter(z_tsne[:, 0], z_tsne[:, 1], c=predicted_cluster, cmap='rainbow', alpha=0.6)
            ax[1].scatter(cluster_means[:, 0], cluster_means[:, 1], c='black', s=30)
            ax[1].set_title('predicted_cluster')
            fig.suptitle(title)
            fig.savefig('figures/' + self.model.name + "/epoch_{}.png".format(epoch))
            
            plt.close(fig)
        else:
            fig, ax = plt.subplots()

            title = 'epoch = {}, loss = {:.2f}'.format(epoch, loss)
            ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=self.c, cmap='rainbow', alpha=0.6)
            ax.set_title(title)
            fig.savefig('figures/' + self.model.name + "/epoch_{}.png".format(epoch))
            plt.close(fig)

    def on_train_begin(self, logs=None):
        try:
            os.mkdir('figures/' + self.model.name)
        except FileExistsError:
            pass

    def on_train_end(self, logs=None):
        loss = logs or {'loss': 0}        
        self.plot('last', loss['loss'])

    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.interval == 0:
            self.plot(epoch, logs['loss'])

    
            
class PrintLossAndAccuracy(tf.keras.callbacks.Callback):

    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y
        print('init acc')

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict_cluster(self.x)
        y_true = self.y
        acc, w = self.compute_accuracy(y_true, y_pred)
        logs['test_metric'] = acc
        loss = logs['loss']
        val_loss = logs['val_loss']
        print('Epoch: {}, loss: {:.2f}, val_loss: {:.2f}, Acc: {:.2f}'.format(epoch, loss, val_loss, acc))
        print('pi: ', self.model.pi_prior.numpy())
        z =self.model.encode(self.x[:1])
        gamma = self.model.compute_gamma(z).numpy()

        #print('gamma: ', gamma)
        log_p_z_given_c = -0.5 * tf.reduce_sum(tf.math.log(2 * np.pi) + self.model.logvar_prior + \
                                 ((self.model.mu_prior - z) ** 2) / tf.exp(self.model.logvar_prior), axis=1)
        print('log p(z|c): ', log_p_z_given_c.numpy())
        print('gamma: ', gamma)


    def compute_accuracy(self, y_true, y_pred):
        D = max(max(y_pred), max(y_true))+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in zip(*ind)])*1.0/len(y_pred)*100, w

