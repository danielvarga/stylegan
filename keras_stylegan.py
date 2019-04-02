# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
import tensorflow as tf


def load_stylegan_networks():
        # Initialize TensorFlow.
        tflib.init_tf()

        # Load pre-trained network.
        url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
            # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
            # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
            # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        return _G, _D, Gs


class StyleGANLayer(Layer):
    def __init__(self, Gs):
        self.network = Gs
        self.latent_dim = 512
        self.output_dim = (1024, 1024, 3)

        self.input_tensor = self.network.input_templates[0]
        self.output_tensor = self.network.output_templates[0]

        super(StyleGANLayer, self).__init__()

    def call(self, x):
        images = self.network.get_output_for(x, x[:, :0])
        images = K.permute_dimensions(images, [0, 2, 3, 1])
        return images
        #drange=[-1, 1]
        #scale = 255 / (drange[1] - drange[0])
        #images = images * scale + (0.5 - drange[0] * scale)
        #return tf.saturate_cast(images, tf.uint8)

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + list(self.output_dim))


def main():
    G, D, Gs = load_stylegan_networks()
    sglayer = StyleGANLayer(Gs)

    inputs =  Input(shape=(sglayer.latent_dim, ))
    outputs = sglayer(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    # Pick latent vector.
    rnd = np.random.RandomState(600)
    latents = rnd.randn(1, sglayer.latent_dim)

    images = model.predict(latents)
    print(images.shape, images.dtype)
    images = ((np.clip(images, -1, 1) + 1) * 255 / 2).astype(np.int8)
    print(images.shape, images.dtype)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, 'example.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
