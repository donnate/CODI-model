from typing import Any

import keras
import numpy as np
from tensorflow.keras.applications import imagenet_utils

from base_model import BaseModel
from image_processing.data_generator import DataGenerator, read
from image_processing.image_model_utils import (
    weighted_log_loss,
    weighted_log_loss_metric,
)

# Can probably do this better, for plotting validation loss
import os
import pandas as pd
import matplotlib.pyplot as plt

preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


TEST_IMAGES_DIR = 'data/'
TRAIN_IMAGES_DIR = 'data/'


class PredictionCheckpoint(keras.callbacks.Callback):
    def __init__(
        self,
        test_df,
        valid_df,
        n_classes=4,
        test_images_dir=TEST_IMAGES_DIR,
        valid_images_dir=TRAIN_IMAGES_DIR,
        batch_size=32,
        input_size=(224, 224, 3),
        output_dir='output/test/'
    ):
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_classes = n_classes
        self.output_dir = output_dir

    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []

    def on_epoch_end(self, batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(
                    self.test_df.index,
                    self.test_df,
                    batch_size=self.batch_size,
                    img_size=self.input_size,
                    img_dir=self.test_images_dir,
                    n_classes=self.n_classes,
                    train=False,
                    n_augment=0,
                    shuffle=True,
                ),
                verbose=2,
            )[: len(self.test_df)]
        )

        self.valid_predictions.append(
            self.model.predict_generator(
                DataGenerator(
                    self.valid_df.index,
                    self.valid_df,
                    batch_size=self.batch_size,
                    img_size=self.input_size,
                    img_dir=self.valid_images_dir,
                    n_classes=self.n_classes,
                    train=False,
                    n_augment=0,
                    shuffle=True,
                ),
                verbose=2,
            )[: len(self.valid_df)]
        )
        valid_labels = np.zeros((self.valid_df.shape[0], self.n_classes))
        valid_labels[np.arange(self.valid_df.shape[0]), self.valid_df['label']] = 1
        print('valid_labels', valid_labels)
        print('pred_labels', self.valid_predictions)
        print(
            "validation loss: %.4f"
            % weighted_log_loss_metric(
                valid_labels, np.average(self.valid_predictions, axis=0)
            )
        )

        # plot the validation loss
        # Check for previous results
        print("\n\n\nHERE\n\n")
        if os.path.exists(f"{self.output_dir}validation_loss.csv"):
            # subsequent epochs
            print("exists")
            results_df = pd.read_csv(f"{self.output_dir}validation_loss.csv")
            print(results_df)
            new_df = pd.DataFrame({'Epoch': [results_df.iloc[len(results_df.index)-1]['Epoch']+1,], 'Validation Loss': [
                weighted_log_loss_metric(
                    valid_labels, np.average(self.valid_predictions, axis=0)
                ),]})
            results_df = results_df.append(new_df)
        else:
            # First epoch
            print("new")
            results_df = pd.DataFrame({'Epoch': [1,], 'Validation Loss': [
                weighted_log_loss_metric(
                    valid_labels, np.average(self.valid_predictions, axis=0)
                ),]})
        print(results_df)
        print(f"{self.output_dir}validation_loss.csv")
        results_df.to_csv(f"{self.output_dir}validation_loss.csv", index=False)

        fig, ax = plt.subplots()
        ax.scatter(results_df['Epoch'], results_df['Validation Loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_ylim(0, ax.get_ylim()[1])
        plt.savefig(f"{self.output_dir}validation_loss.png")
        plt.close()


        # here you could save the predictions with np.save()


class ImageModel(BaseModel):
    def __init__(
        self,
        engine,
        input_dims,
        batch_size=5,
        num_epochs=4,
        n_classes=4,
        learning_rate=1e-3,
        n_augment=9,
        decay_rate=1.0,
        decay_steps=1,
        weights=WEIGHTS_PATH_NO_TOP,
        loss=weighted_log_loss,
        verbose=1,
        test_images_dir=TEST_IMAGES_DIR,
        output_dir='output/test/'
    ):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.n_augment = n_augment
        self.weights = weights
        self.verbose = verbose
        self.loss = loss
        self.test_images_dir = test_images_dir
        self.output_dir = output_dir
        self._build()

    def _build(self):
        self.engine.trainable = True

        engine = self.engine(
            include_top=False,
            weights=self.weights,
            input_shape=(*self.input_dims[:2], 3),
            backend=keras.backend,
            layers=keras.layers,
            models=keras.models,
            utils=keras.utils,
        )
        #TODO: DO we want this to be False?
        for layer in engine.layers:
            #    if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
            #      set_trainable = True
            #    if set_trainable:
            #      layer.trainable = False
            #    else:
            layer.trainable = False
        x = keras.layers.GlobalAveragePooling2D(name='max_pool')(engine.output)
        out = keras.layers.Dense(
            self.n_classes, activation="sigmoid", name='dense_output'
        )(x)

        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        #TODO: loss function has been changed needs to be investigated.
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'],
        )

        # Save results in a specified dir
        print(f"\nSaving here: {self.output_dir}")
        if not os.path.exists('./output'):
            os.mkdir('./output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    #TODO: Lets confirm this is parsed correctly from the notebook
    def fit(self, train_df, valid_df, test_df):

        # callbacks
        pred_history = PredictionCheckpoint(
            test_df,
            valid_df,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            input_size=self.input_dims,
            test_images_dir=self.test_images_dir,
            output_dir=self.output_dir,
        )
        # TODO: Do we want multiprocessing on training?
        self.model.fit_generator(
            DataGenerator(
                list_IDs=train_df.index,
                img_labels=train_df,
                batch_size=self.batch_size,
                img_size=self.input_dims,
                img_dir=TRAIN_IMAGES_DIR,
                n_classes=self.n_classes,
                train=True,
                n_augment=self.n_augment,
                shuffle=True,
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            # use_multiprocessing=True,
            # workers=4#,
            #TODO: Confirm this is meant to be `pred_history` defined above, notebook has
            # this as `history` a return from a previous run of `fit_and_predict`
            callbacks=[pred_history],
        )

        return pred_history

    def predict(self, image_name, path2image=TRAIN_IMAGES_DIR):
        #### Predict one image at a time
        X = read(path2image + image_name, self.input_dims, 0, plot=False)

        res = self.model.predict(X, batch_size=1)
        return res

    def train(self, train_data: Any):
        """
        Code to train model here. This is probably just the iterative online training and not the initial supervised
        training

        :param train_data: training data, should this be a generator?
        :return:
        """
        pass

    def load_training_data_from_s3(self, s3_path: str, *args, **kwargs) -> Any:
        """
        Get training data from s3. Return format should be compatible with train
        :param s3_path: (str) of s3 location to load training data from
        :return:
        """
        pass

    def save(self, path: str) -> None:
        """
        Just save model weights and dont save full model. Should be easier to load
        :param path: (str)
        :return: (None)
        """
        self.model.save_weights(path)

    def save_full_model(self, path: str) -> None:
        """
        Save full keras model
        :param path: (str)
        :return: (None)
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load model weights from local path into self.model
        :param path: (str)
        :return: (None)
        """
        self.model.load_weights(path)
