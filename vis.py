import numpy as np
from PIL import Image


def mergeSets(arrays):
    size = arrays[0].shape[0]
    result = []
    for i in range(size):
        for array in arrays:
            assert array.shape[0] == size, "Incorrect length {} in the {}th array".format(array.shape[0], i)
            result.append(array[i])
    return np.array(result)


# assumes (-1, +1) range, normalizes to (0, 1)
def plotImages(data, n_x, n_y, name, text=None):
    data = data[:n_x * n_y].copy()
    data = np.clip((data + 1) / 2, 0, 1)

    (height, width, channel) = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    if channel == 1:
        mode = "L"
        data = data[:,:,:,0]
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    else:
        mode = "RGB"
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1, channel), dtype='uint8')
    for idx in range(n):
        x = idx % n_x
        y = idx // n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)
    fileName = name + ".png"
    print("Creating file " + fileName)
    if text is not None:
        img.text(10, 10, text)
    img.save(fileName)


# assumes (-1, +1) range, passes it to plotImages unchanged
def display_reconstructed(hourglass, images, name):
    images = images[:50].copy()
    recons = hourglass.predict(images)
    mergedSet = mergeSets([images, recons])
    plotImages(mergedSet, 10, 10, name)
