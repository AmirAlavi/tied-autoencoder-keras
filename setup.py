from setuptools import setup, find_packages

setup(
    name='tied_autoencoder_keras',
    version='0.4.0',
    description='Autoencoder layers (with tied encode and decode weights) for Keras',
    author='Amir Alavi',
    url='https://github.com/AmirAlavi/tied-autoencoder-keras',
    license='GPLv3',
    packages=find_packages(),
    install_requires=[
        'keras>=2',
        'sparsely-connected-keras'],
    python_requires='>=3')
