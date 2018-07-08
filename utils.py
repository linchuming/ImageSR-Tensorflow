import numpy as np
from scipy import misc
import tensorflow as tf

def psnr(im1, im2):
    """ im1 and im2 value must be between 0 and 255"""
    im1 = np.float64(im1)
    im2 = np.float64(im2)
    rmse = np.sqrt(np.mean(np.square(im1[:] - im2[:])))
    psnr = 20 * np.log10(255 / rmse)
    return psnr


def img_to_uint8(img):
    if np.max(img) <= 1.0:
        img *= 255.0
    img = np.clip(img, 0, 255)
    return np.round(img).astype(np.uint8)


rgb_to_ycbcr = np.array([[65.481, 128.553, 24.966],
                         [-37.797, -74.203, 112.0],
                         [112.0, -93.786, -18.214]])

ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)


def rgb2ycbcr(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = np.dot(img, rgb_to_ycbcr.T) / 255.0
    img = img + np.array([16, 128, 128])
    return img


def ycbcr2rgb(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = img - np.array([16, 128, 128])
    img = np.dot(img, ycbcr_to_rgb.T) * 255.0
    return img


def tf_resize_image(imgs, scale):
    def resize_image(imgs, scale):
        b = imgs.shape[0]
        c = imgs.shape[-1]
        res = []
        for i in range(b):
            img = imgs[i]
            tar_img = []
            for j in range(c):
                tar_img.append(misc.imresize(img[:, :, j], scale / 1.0, 'bicubic', mode='F'))
            img = np.stack(tar_img, -1)
            res.append(img)

        return np.stack(res)
    return tf.py_func(lambda x: resize_image(x, scale), [imgs], tf.float32)
