# tied-autoencoder-keras
Autoencoder layers (with tied encode and decode weights) for Keras

Tutorial: [https://amiralavi.net/blog/2018/08/25/tied-autoencoders](https://amiralavi.net/blog/2018/08/25/tied-autoencoders)

`DenseLayerAutoencoder` is a derived class of the Keras built-in Dense class.

To use `DenseLayerAutoencoder`, you call its constructor in exactly the same way as you would for Dense, but instead of passing in a `units` argument, you pass in a `layer_sizes` argument which is just a python list of the number of units that you want in each of your encoder layers (it assumes that your autoencoder will have a symmetric architecture, so you only specify the encoder sizes). An example:

```
inputs = Input(shape=(1000,))
x = DenseLayerAutoencoder([100, 50, 20], activation='tanh')(inputs)

model = Model(inputs=inputs, outputs=x)
print(model.summary())
```

`DenseLayerAutoencoder` also provides a `encode` and `decode` function (which are both called by the `call` function).
