import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


class Encoder(tf.keras.Model):

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

class Decoder(tf.keras.Model):

    def __init__(self, original_dim=5491, latent_dim=10, name='Decoder'):
        super(Decoder, self).__init__(name=name)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.h1 = layers.Dense(latent_dim, activation='relu')
        self.h2 = layers.Dense(518, activation='relu')
        self.h3 = layers.Dense(2048, activation='relu')
        self.outputs = layers.Dense(original_dim, activation='relu')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        return self.outputs(x)


class AutoEncoder(tf.keras.Model):

    def __init__(self, original_dim=5491, latent_dim=10, name='AutoEncoder'):
        super(AutoEncoder, self).__init__(name=name)
        self.encoder = Encoder(original_dim=original_dim, latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim, latent_dim=latent_dim)

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, x):
        return self.decoder(x)

    def call(self, x):
        x = self.encode(x)
        return self.decode(x)

    def compute_loss(self, x, x_hat):
        return self.original_dim * losses.binary_crossentropy(x, x_hat)
