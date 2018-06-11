import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model, load_model

from tied_autoencoder_keras import DenseLayerAutoencoder

X = np.random.randn(500, 1000)
X_corrupt = X + 0.1 * np.random.normal(loc=0, scale=1, size=X.shape)

inputs = Input(shape=(1000,))
x = DenseLayerAutoencoder([100, 50, 20], activation='tanh')(inputs)

model = Model(inputs=inputs, outputs=x)
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(X_corrupt, X, epochs=10)

# Use it to embed some data
embedded = model.layers[1].encode(model.layers[0].input)

#get_activations = embedded
get_activations = K.function([model.layers[0].input], [embedded])
X_transformed = get_activations([X])[0]
print(X_transformed)
print(X_transformed.shape)
assert(X_transformed.shape[1] == 20)

model.save('dense_autoencoder_test.h5')

del model

model = load_model('dense_autoencoder_test.h5', custom_objects={'DenseLayerAutoencoder': DenseLayerAutoencoder})
print(model.summary())

