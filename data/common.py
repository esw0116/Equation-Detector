import numpy as np
from scipy import misc


def preprocess(img):
    if img.ndim == 3:
        img[:, :, 0] *= 65.738
        img[:, :, 1] *= 129.507
        img[:, :, 2] *= 25.064
        temp = img.sum(axis=2) / 256
        img = temp

    max = img.max()
    return max - img

def normalize_img(img):
    return img/255

def rand_place(img, h=96, w=480):
    y, x = img.shape
    max_upscale = min(h/y, w/x)
    min_upscale = 0.5
    scale = np.random.rand()*(max_upscale - min_upscale) + min_upscale
    after_scale_x = int(scale * x)
    after_scale_y = int(scale * y)
    img_scaled = misc.imresize(img, size=(after_scale_y, after_scale_x), interp='nearest')

    randx = np.random.randint(0, w - after_scale_x)
    randy = np.random.randint(0, h - after_scale_y)

    palette = np.zeros((h, w))
    palette[randy:randy+after_scale_y, randx:randx+after_scale_x] = img_scaled

    return palette

def exp_rand_place(img, h=96, w=480):
    y, x = img.shape
    randx = np.random.randint(0, w - x)
    randy = np.random.randint(0, h - y)

    palette = np.zeros((h, w))
    palette[randy:randy+y, randx:randx+x] = img
    
    return palette
    