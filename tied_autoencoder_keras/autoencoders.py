from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense
from sparsely_connected_keras import Sparse


class DenseLayerAutoencoder(Dense):
    def __init__(self, layer_sizes, *args, **kwargs):
        self.layer_sizes = layer_sizes
        self.kernels = []
        self.biases = []
        self.biases2 = []
        super().__init__(units=1, *args, **kwargs)  # 'units' not used

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        for i in range(len(self.layer_sizes)):

            self.kernels.append(
                self.add_weight(
                    shape=(
                        input_dim,
                        self.layer_sizes[i]),
                    initializer=self.kernel_initializer,
                    name='ae_kernel_{}'.format(i),
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(
                        shape=(
                            self.layer_sizes[i],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias_{}'.format(i),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))
            input_dim = self.layer_sizes[i]

        if self.use_bias:
            for n, i in enumerate(range(len(self.layer_sizes)-2, -1, -1)):
                self.biases2.append(
                    self.add_weight(
                        shape=(
                            self.layer_sizes[i],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias2_{}'.format(n),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))
            self.biases2.append(self.add_weight(
                        shape=(
                            input_shape[-1],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias2_{}'.format(len(self.layer_sizes)),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))

        self.built = True

    def call(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        latent = inputs
        for i in range(len(self.layer_sizes)):
            latent = K.dot(latent, self.kernels[i])
            if self.use_bias:
                latent = K.bias_add(latent, self.biases[i])
            if self.activation is not None:
                latent = self.activation(latent)
        return latent

    def decode(self, latent):
        recon = latent
        for i in range(len(self.layer_sizes)):
            recon = K.dot(recon, K.transpose(self.kernels[len(self.layer_sizes) - i - 1]))
            if self.use_bias:
                recon = K.bias_add(recon, self.biases2[i])
            if self.activation is not None:
                recon = self.activation(recon)
        return recon

    def get_config(self):
        config = {
            'layer_sizes': self.layer_sizes
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
