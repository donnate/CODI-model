""" Incremental Training of Image Featuriser"""

import argparse
import json
import os

from keras.applications.resnet import ResNet50

from image_processing.image_model import ImageModel


def _parse_args():
    """
    Parse args for training job
    :return:
    """

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--latest_model_path', type=str)
    parser.add_argument(
        '--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR')
    )
    parser.add_argument(
        '--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING')
    )
    parser.add_argument(
        '--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS'))
    )
    parser.add_argument(
        '--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST')
    )
    parser.add_argument(
        '--bucket_name', type=str
    )
    parser.add_argument(
        '--aws_secret_access_key', type=str
    )
    parser.add_argument(
        '--aws_access_key_id', type=str
    )
    parser.add_argument(
        '--region_name', type=str
    )

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()

    s3_kwargs = {
        'aws_access_key_id': args.aws_access_key_id,
        'aws_secret_access_key': args.aws_secret_access_key,
        'bucket_name': args.bucket_name,
        'region_name': args.region_name,
    }

    # Create the Estimator
    model = ImageModel(
        engine=ResNet50,
        input_dims=(512, 512, 3),
        batch_size=3,
        learning_rate=1e-5,
        n_classes=4,
        num_epochs=1,
        decay_rate=0.8,
        decay_steps=1,
        weights='imagenet',
        loss="mean_squared_error",
        verbose=2,
    )

    # Load training data from S3
    #TODO: Do we want to pipe this in to the model training step?
    train_data = model.load_training_data_from_s3(args.train, **s3_kwargs)

    # Load previous model state
    #TODO: This is loading in model weights atm, could we just load the seriealised model?
    model.load_model_from_s3(args.latest_model_path, **s3_kwargs)

    # Train the model here from passed in training data
    model.fit(train_data)

    # This saves model for easy reloading
    model.save_model_to_s3(args.latest_model_path, **s3_kwargs)

    # Save model in compressed form to be consumed by client
    model.save_(f"{args.model_output_dir}/model_out")
