import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model, load_model, Sequential

from tied_autoencoder_keras import DenseLayerAutoencoder

X = np.random.randn(500, 1000)
X_corrupt = X + 0.1 * np.random.normal(loc=0, scale=1, size=X.shape)
print("X_original shape: {}".format(X.shape))

inputs = Input(shape=(1000,))
x = DenseLayerAutoencoder([100, 50, 20], activation='tanh', dropout=0.20)(inputs)

model = Model(inputs=inputs, outputs=x)
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(X_corrupt, X, epochs=10)

# Use it to embed some data
sample_in = Input(shape=model.layers[0].input_shape[1:], name='sample_input')
embedded = Lambda(lambda x: model.layers[1].encode(x),
                  output_shape=(20,), name='encode')(sample_in)
embedded._uses_learning_phase = True # Dropout ops use learning phase
embedder = Model(sample_in, embedded)
print(embedder.summary())

X_transformed = embedder.predict(X)
print(X_transformed)
print("X_encoded shape: {}".format(X_transformed.shape))
assert(X_transformed.shape[1] == 20)

model.save('dense_autoencoder_test.h5')

del model

model = load_model('dense_autoencoder_test.h5', custom_objects={'DenseLayerAutoencoder': DenseLayerAutoencoder})
print(model.summary())

