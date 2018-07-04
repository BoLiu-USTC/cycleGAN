# From https://github.com/LynnHo/CycleGAN-Tensorflow-Simple/blob/master/utils.py

import copy
import os.path
import numpy as np
import scipy.misc
import tensorflow as tf


class ItemPool(object):

    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """ in_items is a list of item"""
        if self.max_num == 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

def load_checkpoint(checkpoint_dir, sess, saver):
    print(" [*] Loading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        saver.restore(sess, ckpt_path)
        print(" [*] Loading successful!")
        return ckpt_path
    else:
        print(" [*] No suitable checkpoint!")
        return None

def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)
    `images` is in shape of N * H * W(* C=1 or 3)
    """

    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))
