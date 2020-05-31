from math import ceil

import cv2
import keras
# import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def read(path, desired_size, augment_data=0, plot=False):
    """Will be used in DataGenerator
    Loads image, crops and resizes. With optional image data augmentation.
    We assume that the image has been centered.
    Input:
    ----------------------------------
    desired_size     :  desired size for the image (tuple)
    augment_data     :  nb of data augmented samples (int)
    """
    new_width, new_height, _ = desired_size
    print(path)
    img = cv2.imread(path)
    rows, cols, _ = img.shape
    if rows < cols:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
    res = cv2.resize(img, dsize=desired_size[:2], interpolation=cv2.INTER_CUBIC)
    samples = np.expand_dims(res, 0)
    if augment_data > 0:
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=90, width_shift_range=[-100, 100])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        for i in range(augment_data):
            batch = it.next()  # generate batch of images
            image = batch[0]
            samples = np.vstack((samples, np.expand_dims(image, 0)))
#             if plot:
#                 plt.subplot(330 + 1 + i)
#                 plt.imshow(batch[0].astype('uint8'))
#             # img = np.stack((res,)*3, axis=-1)
#     if plot:
#         plt.show()
#    print('samples size in read:', samples.shape)
    return samples


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        list_IDs,
        img_labels,
        img_dir,
        batch_size=1,
        img_size=(512, 512, 3),
        n_classes=4,
        train=True,
        n_augment=9,
        shuffle=True,
    ):

        self.list_IDs = list_IDs
        self.indices = np.arange(len(self.list_IDs))
        self.img_labels = (
            img_labels
        )  ### contains col1: names of images for loading + col2(!exits fr test) for labels
        self.n_classes = n_classes  ### nb of classes
        self.n_augment = n_augment  ### nb of additional data samples
        self.batch_size = batch_size
        self.img_size = img_size  ###  desired image size: (width, height, n_channels)
        self.img_dir = img_dir
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        if self.train:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        #print("Self image size", self.img_size)
        X = np.empty((self.batch_size * (self.n_augment + 1), *self.img_size))

        if self.train:  # training phase
            Y = np.zeros(
                (self.batch_size * (self.n_augment + 1), self.n_classes),
                dtype=np.float32,
            )

            for i, ID in enumerate(list_IDs_temp):
                test = read(
                    self.img_dir + self.img_labels['ID'].loc[ID] + ".jpg",
                    self.img_size,
                    augment_data=self.n_augment,
                    plot=False,
                )
                #print("Dim data gen", test.shape)
                X[i : (i + self.n_augment + 1),] = test
                ### Convert label  into one hot vector
                Y[
                    i : (i + self.n_augment + 1), int(self.img_labels['label'].loc[ID])
                ] = 1

            return X, Y

        else:  # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = read(
                    self.img_dir + self.img_labels['ID'].loc[ID] + ".jpg",
                    self.img_size,
                    augment_data=0,
                    plot=False,
                )

            return X
