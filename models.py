import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

from sklearn.mixture import GaussianMixture


class Encoder(layers.Layer):

    def __init__(self, original_dim=5491, latent_dim=10, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.h1 = layers.Dense(2048, activation='relu')
        self.h2 = layers.Dense(512, activation='relu')
        self.mu_dense = layers.Dense(latent_dim, activation='linear')
        self.logvar_dense = layers.Dense(latent_dim, activation='linear')

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
        eps = tf.random.normal(shape=tf.shape(mean))
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

    def reconstruction_loss(self, x, x_hat):
        return self.original_dim * losses.binary_crossentropy(x, x_hat)


class VariationalAutoEncoder(AutoEncoder):

    def __init__(self, original_dim=5491, latent_dim=10, name='VariationalAutoEncoder'):
        super(VariationalAutoEncoder, self).__init__(origina_dim=original_dim,
                                                     latent_dim=latent_dim,
                                                     name=name)
        self.sampling = SamplingLayer()

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling([mu, logvar])
        kl_loss = self.vae_loss([mu, logvar])
        self.add_loss(kl_loss)
        return self.decoder(z)

    def vae_loss(self, inputs):
        mu, logvar = inputs
        loss = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        return tf.reduce_mean(loss)


class VariationalDeepEmbedding(VariationalAutoEncoder):

    def __init__(self,
                 original_dim=5491,
                 latent_dim=10,
                 n_components=5,
                 name='VariationalDeepEmbedding'):

        super(VariationalDeepEmbedding, self).__init__(origina_dim=original_dim,
                                                        latent_dim=latent_dim,
                                                        name=name)
        self.n_components = n_components
        self.gmm = None

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling([mu, logvar])
        kl_loss = self.vade_loss([mu, logvar, z])
        self.add_loss(kl_loss)
        return self.decoder(z)

    def vade_loss(self, inputs):
        mu, logvar, z = inputs
        loss = 0
        return loss

    def fit(self, X, *args):
        if self.gmm:
            super(VariationalDeepEmbedding, self).fit(*args)
        else:
            print('Fitting GMM')
            self.fit_gmm(X)
            self.fit(X, X, *args)

    def fit_gmm(self, X):
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag')
        self.gmm.fit(X)
        self.pi_prior = self.gmm.weights_
        self.mu_prior = self.gmm.means_
        self.logvar_prior = self.gmm.covariances_





