from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np

from keras_stylegan import StyleGANLayer


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
        model = Sequential()

        # dummy
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(Flatten())
        model.add(Dense(latent_dim, activation='linear'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)



        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(latent_dim, activation='linear'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)


def main():
    latent_dim = 512
    generator = StyleGANLayer()
    generator.trainable = False
    generator = Sequential([generator])
    inp = np.random.normal(size=(128, latent_dim))
    images = generator.predict(inp)
    save_imgs(generator, latent_dim, "model-d%d.png" % latent_dim)
    
    inverter = build_inverter((1024, 1024, 3), latent_dim)
    print(">>>", inverter.predict(images).shape)
    barrel = Sequential([generator, inverter])
    hourglass = Sequential([inverter, generator])
    print(barrel.predict(inp).shape)

    epochs = 10
    which = "barrel"
    if which == "barrel":
        barrel.compile(optimizer=Adam(lr=0.002), loss='mse')
        z_train = np.random.normal(size=(25600, latent_dim))
        z_test  = np.random.normal(size=(  160, latent_dim))
        barrel.fit(z_train, z_train,
                epochs=epochs,
                batch_size=16,
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
