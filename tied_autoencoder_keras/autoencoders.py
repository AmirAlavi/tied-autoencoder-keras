from keras import backend as K
from keras.layers import Dense

from sparsely_connected_keras import Sparse


class DenseLayerAutoencoder(Dense):
    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        if self.use_bias:
            self.bias2 = self.add_weight(shape=(input_shape[-1],),
                                         initializer=self.bias_initializer,
                                         name='bias2',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias2 = None
        super().build(input_shape)

    def call(self, inputs):
        latent = K.dot(inputs, self.kernel)
        if self.use_bias:
            latent = K.bias_add(latent, self.bias)
        if self.activation is not None:
            latent = self.activation(latent)
        reconstruction = K.transpose(self.kernel)
        reconstruction = K.dot(latent, reconstruction)
        if self.use_bias:
            reconstruction = K.bias_add(reconstruction, self.bias2)
        if self.activation is not None:
            reconstruction = self.activation(reconstruction)
        return reconstruction

    def encode(self, inputs):
        latent = K.dot(inputs, self.kernel)
        if self.use_bias:
            latent = K.bias_add(latent, self.bias)
        if self.activation is not None:
            latent = self.activation(latent)
        return latent

class SparseLayerAutoencoder(Sparse):
    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        if self.use_bias:
            # bias2 is for after the latent layer
            self.bias2 = self.bias = self.add_weight(
                shape=(
                    self.adjacency_mat.shape[0],
                ),
                initializer=self.bias_initializer,
                name='bias2',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias2 = None
        super().build(input_shape)

    def call(self, inputs):
        latent = self.kernel * self.adjacency_tensor
        latent = K.dot(inputs, latent)
        if self.use_bias:
            latent = K.bias_add(latent, self.bias)
        if self.activation is not None:
            latent = self.activation(latent)
        reconstruction = K.transpose(
            self.kernel) * K.transpose(self.adjacency_tensor)
        reconstruction = K.dot(latent, reconstruction)
        if self.use_bias:
            reconstruction = K.bias_add(reconstruction, self.bias2)
        if self.activation is not None:
            reconstruction = self.activation(reconstruction)
        return reconstruction

    def encode(self, inputs):
        latent = self.kernel * self.adjacency_tensor
        latent = K.dot(inputs, latent)
        if self.use_bias:
            latent = K.bias_add(latent, self.bias)
        if self.activation is not None:
            latent = self.activation(latent)
        return latent
