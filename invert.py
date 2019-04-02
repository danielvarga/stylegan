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


def save_real_recons(generator, hourglass, inp, prefix):
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

    resolution = 32
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


def main():
    G, D, Gs = keras_stylegan.load_stylegan_networks()

    latent_dim = 512
    generator = keras_stylegan.StyleGANLayer(Gs)
    inp = np.random.normal(size=(128, latent_dim))
    # images = generator.predict(inp)
    # save_imgs(generator, latent_dim, "model-d%d.png" % latent_dim)

    print("we dumb it down to 32x32.")
    inverter = build_inverter((32, 32, 3), latent_dim)
    inputs = Input(shape=(512, ))
    barrel_output = inverter(AveragePooling2D(pool_size=(32, 32))(generator(inputs)))
    barrel = Model(inputs, barrel_output)

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
    # save_real_recons(generator, hourglass, x_test, name)

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
