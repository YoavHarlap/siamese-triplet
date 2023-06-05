from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import glob, os
import random
from scipy.stats import bernoulli
from scipy import ndimage


def load_mrcs_from_path_to_arr(path):
    images = []
    images_filenames = []
    os.chdir(path)
    for file in glob.glob("*.mrc"):
        # print(file)
        mrc = mrcfile.open(file, 'r')
        mrc_imgs = np.array(mrc.data)
        mrc.close()
        # print(mrc_imgs.shape)
        # print(mrc_imgs.ndim)

        if (mrc_imgs.ndim == 2):  # was just 1 pic in mrc file
            mrc_imgs = [mrc_imgs]

        for i in range(len(mrc_imgs)):
            images.append(mrc_imgs[i])
            images_filenames.append(file)
    return images


def make_imgs_arr_from_labels(labels, good_imgs, outliers_imgs):
    imgs = []
    k = 0
    p = 0
    for i in range(len(labels)):
        if (labels[i] != 1):
            labels[i] = -1
            imgs.append(outliers_imgs[k])
            k = k + 1
        else:
            imgs.append(good_imgs[p])
            p = p + 1

    imgs = np.array(imgs)
    return imgs, labels


def make_cryo_imgs_arr(len_arr):
    good_imgs_path = "/data/yoavharlap/eman_particles/good"
    outliers_imgs_path = "/data/yoavharlap/eman_particles/outliers"
    good_imgs = load_mrcs_from_path_to_arr(good_imgs_path)
    outliers_imgs = load_mrcs_from_path_to_arr(outliers_imgs_path)

    random.shuffle(good_imgs)
    random.shuffle(outliers_imgs)

    print(len(good_imgs))
    print(len(outliers_imgs))
    p = 2 / 3
    random_labels = bernoulli.rvs(p, size=len_arr)
    imgs, true_labels = make_imgs_arr_from_labels(random_labels, good_imgs, outliers_imgs)
    return imgs,true_labels

