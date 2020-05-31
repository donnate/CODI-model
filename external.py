import os
from typing import Any

import boto3
from botocore.config import Config


def _pop_arg(kwargs: dict, key: str, env_default_key: str) -> Any:
    """
    :param kwargs: (dict)
    :param key: (str)
    :param env_default_key: (str)
        if key is not in kwargs or if popped val is None
        then use os.environ.get(env_default_key)
    """
    if key in kwargs:
        popped_val = kwargs.pop(key)
    else:
        popped_val = None

    return popped_val or os.environ.get(env_default_key)


def get_bucket(**s3_kwargs) -> "boto3.s3.Bucket":
    """
    Get s3 bucket
    s3_kwargs can be passed through or taken from env vars

    :param s3_kwargs: to pass into constructor
    :return: (boto3.s3.bucket) Bucket object
    """
    resource = boto3.resource(
        's3',
        aws_access_key_id=_pop_arg(s3_kwargs, 'aws_access_key_id', 'AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=_pop_arg(
            s3_kwargs, 'aws_secret_access_key', 'AWS_SECRET_ACCESS_KEY'
        ),
        region_name=_pop_arg(s3_kwargs, 'region_name', 'AWS_DEFAULT_REGION'),
        config=Config(signature_version='s3v4'),
    )
    bucket = resource.Bucket(_pop_arg(s3_kwargs, 'bucket_name', 'DEFAULT_S3_BUCKET'))

    return bucket
