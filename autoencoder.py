import tensorflow as tf
from keras import layers
from keras import Model


class Autoencoder(Model):
    def __init__(self, latent_dim, dim_obj_1, dim_obj_2):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dim_obj_1 = dim_obj_1
        self.dim_obj_2 = dim_obj_2
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(self.latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(self.dim_obj_1 * self.dim_obj_2, activation='sigmoid'),
            layers.Reshape((self.dim_obj_1, self.dim_obj_2))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
