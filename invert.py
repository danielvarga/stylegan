from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, AveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np

import keras_stylegan
import vis

def loadModel(filePrefix):
    jsonFile = filePrefix + ".json"
    weightFile = filePrefix + ".h5"
    jFile = open(jsonFile, 'r')
    loaded_model_json = jFile.read()
    jFile.close()
    mod = model_from_json(loaded_model_json)
    mod.load_weights(weightFile)
    print("Loaded model from files {}, {}".format(jsonFile, weightFile))
    return mod


def saveModel(mod, filePrefix):
    weightFile = filePrefix + ".h5"
    mod.save_weights(weightFile)
    jsonFile = filePrefix + ".json"
    with open(filePrefix + ".json", "w") as json_file:
        json_file.write(mod.to_json())
    print("Saved model to files {}, {}".format(jsonFile, weightFile))


def save_imgs(generator, latent_dim, filename):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(filename)
        plt.close()


def save_grid(gen_imgs, r, c, filename):
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(filename)
        plt.close()


def save_synth_recons(generator, hourglass, latent_dim, prefix):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)
        recons_imgs = hourglass.predict(gen_imgs)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        recons_imgs = 0.5* recons_imgs + 0.5
        save_grid(gen_imgs,    r, c, prefix+"-gen.png")
        save_grid(recons_imgs, r, c, prefix+"-gen-recons.png")


def save_real_recons(hourglass, inp, prefix):
        r, c = 5, 5
        gen_imgs = inp[:r * c]
        recons_imgs = hourglass.predict(gen_imgs)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        recons_imgs = 0.5* recons_imgs + 0.5
        save_grid(gen_imgs,    r, c, prefix+"-real.png")
        save_grid(recons_imgs, r, c, prefix+"-real-recons.png")


def build_inverter(img_shape, latent_dim):
    from keras.layers import LeakyReLU, Conv2D, AveragePooling2D

    resolution = 64
    resolution_log2 = int(np.log2(resolution))
    fmap_base = 8192
    fmap_decay = 1.0
    fmap_max = 512
    leak = 0.2
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    gain = np.sqrt(2)

    assert resolution == img_shape[0] == img_shape[1]
    assert latent_dim == 512
    images_in = Input(shape=img_shape)


    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        return LeakyReLU(leak)(Conv2D(filters=nf(res-1), kernel_size=1, padding="same", input_shape=img_shape)(x))

    def block(x, res): # res = 2..resolution_log2
        if res >= 3: # 8x8 and up
            x = LeakyReLU(leak)(Conv2D(filters=nf(res-1), kernel_size=3, padding="same")(x))
            x = Conv2D(filters=nf(res-2), kernel_size=3, padding="same")(x)
            x = LeakyReLU(leak)(AveragePooling2D()(x)) # blur omitted
        else: # 4x4
            x = LeakyReLU(leak)(Conv2D(filters=nf(res-1), kernel_size=3, padding="same")(x))
            x = Dense(nf(res-2))(Flatten()(x))
        return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    x = fromrgb(images_in, resolution_log2)
    for res in range(resolution_log2, 2, -1):
        x = block(x, res)
    z = block(x, 2)
    return Model(images_in, z)


import keras.utils

class GaussianDataGenerator(keras.utils.Sequence):
    def __init__(self, latent_dim, epoch_size, batch_size):
        self.latent_dim = latent_dim
        self.epoch_size = epoch_size
        self.batch_size = batch_size

    def __len__(self):
        return self.epoch_size // self.batch_size

    def __getitem__(self, index):
        z = np.random.normal(size=(self.batch_size, self.latent_dim))
        return z, z # autoencoder!

    def on_epoch_end(self):
        pass

from keras.callbacks import Callback

class VisualizationCallback(Callback):
    def __init__(self, generator, inverter, images):
        self.generator = generator
        self.inverter = inverter
        self.images = images
        super(VisualizationCallback, self).__init__()

    def on_epoch_end(self, epoch, logs):
        dumb_inputs = Input(shape=(64, 64, 3))
        hourglass_output = AveragePooling2D(pool_size=(16, 16))(self.generator(self.inverter(dumb_inputs)))
        hourglass = Model(dumb_inputs, hourglass_output)
        vis.display_reconstructed(hourglass, self.images, "images/%03d-heldout_synth_recons_64" % (epoch+1))


class ModelSaveCallback(Callback):
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % 10 == 0:
            saveModel(self.model, "%s-%03d" % (self.name, epoch + 1))

def main():
    G, D, Gs = keras_stylegan.load_stylegan_networks()

    latent_dim = 512
    generator = keras_stylegan.StyleGANLayer(Gs)
    # images = generator.predict(inp)
    # save_imgs(generator, latent_dim, "model-d%d.png" % latent_dim)

    print("we dumb it down to 64x64.")
    inverter = build_inverter((64, 64, 3), latent_dim)
    inputs = Input(shape=(512, ))
    dumbed_output = AveragePooling2D(pool_size=(16, 16))(generator(inputs))
    dumbed_generator = Model(inputs, dumbed_output)

    data_from_cache = True
    if data_from_cache:
        print("loading data...")
        latents = np.load("latent.npy")
        images = np.load("generated.npy")
        print("data loaded")
        n = len(latents)
    else:
        n = 30000
        latents = np.random.normal(size=(n, latent_dim))
        np.save("latent.npy", latents)
        print("creating images...")
        images = dumbed_generator.predict(latents)
        print("images created")
        np.save("generated.npy", images)

    inverter_from_cache = False
    if inverter_from_cache:
        # if we take the saves/inverter.h5 with the
        # saved/generated.npy that it was trained on,
        # we get very good reconstruction, see
        # https://old.renyi.hu/~daniel/tmp/inverting-is-hard/stylegan/overfit.png
        inverter = loadModel("inverter")
        dumb_inputs = Input(shape=(64, 64, 3))
        hourglass_output = dumbed_generator(inverter(dumb_inputs))
        hourglass = Model(dumb_inputs, hourglass_output)
        vis.display_reconstructed(hourglass, images, "overfit")
        return
    else:
        generate_on_the_fly = True
        if generate_on_the_fly:
            print("training the inverter with data generated on-the-fly.")

            latent_generator = GaussianDataGenerator(latent_dim=latent_dim, epoch_size=20000, batch_size=32)
            vis_callback = VisualizationCallback(generator, inverter, images)
            save_callback = ModelSaveCallback(inverter, "snapshots/inverter-snapshot")

            barrel = Sequential([dumbed_generator, inverter])
            barrel.compile(optimizer=Adam(lr=0.0001), loss='mse')
            
            barrel.fit_generator(generator=latent_generator, epochs=1200, callbacks=[vis_callback, save_callback])
            saveModel(inverter, "inverter")
            return
        else:
            print("training the inverter with precomputed data.")
            inverter.compile(optimizer=Adam(lr=0.0001), loss='mse')
            inverter.fit(images, latents, batch_size=32, shuffle=True, epochs=100)
            saveModel(inverter, "inverter")
            return

    epochs = 3
    which = "barrel"
    if which == "barrel":
        barrel.compile(optimizer=Adam(lr=0.0001), loss='mse')
        z_train = np.random.normal(size=(256000, latent_dim))
        z_test  = np.random.normal(size=(  160, latent_dim))
        barrel.fit(z_train, z_train,
                epochs=epochs,
                batch_size=32,
                shuffle=True,
                validation_data=(z_test, z_test))
    else:
        assert which == "hourglass"
        hourglass.compile(optimizer='adam', loss='mse')

        hourglass.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

    name = which + "-inverter"
    saveModel(inverter, name)

    save_synth_recons(generator, hourglass, latent_dim, name)
    
    print("save_real_recons badly missing")
    # save_real_recons(hourglass, x_test, name)

    n = 50
    a = np.mgrid[-2:+2:(n*1j), -2:+2:(n*1j)].reshape(2,-1)
    grid_points = np.vstack((a, np.zeros((latent_dim-2, a.shape[-1])))).T
    mapped_grid_points = barrel.predict(grid_points)
    colors = [(float(i%n)/n, float(i-i%n)/n/n, 0.0) for i in range(n*n)]
    plt.scatter(mapped_grid_points[:, 0], mapped_grid_points[:, 1], c=colors)
    plt.savefig(name + "-latent.png")
    plt.close()


if __name__ == '__main__':
    main()
