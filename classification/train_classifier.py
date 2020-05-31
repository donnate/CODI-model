""" Incremental Training of Image Featuriser"""

import argparse
import json
import os

from classification.vanilla_classifier import VanillaClassifier


def _parse_args():
    """
    Parse args for training job
    :return:
    """

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--latest_model_path', type=str)
    parser.add_argument(
        '--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR')
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
    model = VanillaClassifier()

    # Load training data from S3
#     train_data = model.load_training_data_from_s3(args.train, **s3_kwargs)

    # Load previous model state
    model.load_model_from_s3(args.latest_model_path, **s3_kwargs)

    # Train the model here from passed in training data
#     model.fit(train_data)

    print(args.model_dir)
    print(args)
    print(os.path.join(args.model_dir, 'model.pickle'))
    
    model.save(os.path.join(args.model_dir, 'model.pickle'))
        
    # This saves model for easy reloading
    model.save_model_to_s3(args.latest_model_path, **s3_kwargs)
