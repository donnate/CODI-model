#!/usr/bin/env python3

# ---------------------------------
# Set seeds to ensure reproducibility
# ---------------------------------

# From: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# with API updated with lots of ".compat.v1."
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.random.set_seed(1234)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# ---------------------------------
# end setting seeds
# ---------------------------------



import pandas as pd
from keras_applications.resnet import ResNet50
from sklearn.model_selection import ShuffleSplit
import os

from image_processing.image_model import ImageModel

import argparse

def _parse_args():
    parser = argparse.ArgumentParser()

    # Test data directory as an example
    parser.add_argument('--test-data-dir', type=str, default='data/')
    parser.add_argument('--output-dir', type=str, default='output/test/')
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--n-epochs', type=int, default=12)

    return parser.parse_known_args()


def read_testset(filename="data/classification_labels.txt"):
    '''
    Data in data folder
    '''
    df = pd.read_csv(filename, sep=" ", header=None)
    df.columns = ["ID", "label"]
    return df


def read_trainset(filename="data/classification_labels.txt"):
    df = pd.read_csv(filename, sep=" ", header=None)
    df.columns = ["ID", "label"]
    return df

args, unknown = _parse_args()
print(f"Args: {args}")
assert(args.output_dir[-1] == '/'), f"Need to complete path with '/' for directory structure. Provided path = {args.output_dir}"
assert(args.learning_rate < 1e-2 and args.learning_rate > 1e-6), f"Force reasonable learning rates where learning_rate < 1e-2 and learning_rate > 1e-6. You provided {args.learning_rate}"
assert(args.n_epochs < 500), f"Just to keep this reasonable, leep n_epochs < 500, you have {args.n_epochs}"

# Get test and train data
test_df = read_testset()
df = read_trainset()

# train set (80%) and validation set (20%)
ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)

# lets go for the first fold only
#TODO: Decide how to train on the full data set
train_idx, valid_idx = next(ss)

# obtain model
model = ImageModel(
    engine=ResNet50,
    input_dims=(224, 224, 3),
    batch_size=3,
    learning_rate=args.learning_rate,
    n_augment=0,
    num_epochs=args.n_epochs,
    decay_rate=0.8,
    decay_steps=1,
    weights="imagenet",
    verbose=2,
    test_images_dir=args.test_data_dir,
    output_dir=args.output_dir
)

# obtain test + validation predictions (history.test_predictions, history.valid_predictions)
history = model.fit(df.iloc[train_idx], df.iloc[valid_idx], test_df)

# Not sure what we actually want to output here, the initial code didn't work so this produces a csv of submissions with predictions for labels
out_df = test_df.join(
    pd.DataFrame(
        np.average(
            history.test_predictions,
            axis=0,
            weights=[2 ** i for i in range(len(history.test_predictions))],
        )
    )
)
# test_df = test_df.stest_df.aptack().reset_index()
#
# test_df.insert(
#     loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis']
# )
#
# test_df = test_df.drop(["Image", "Diagnosis"], axis=1)



out_df.to_csv(f'{args.output_dir}submission.csv', index=False)
model.save_full_model(f'{args.output_dir}model_out.h5')
