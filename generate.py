import json
import os
import logging
from functools import reduce
from typing import Tuple, Dict, List, Any

import pandas as pd

from classification.vanilla_classifier import VanillaClassifier as Model

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)


def camel_to_snake(camel: str) -> str:
    """
    Convert input Camel Case to snake case for prediction
    :param camel: (str) input camel case string
    :return: (str) output snake case string
    """
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, camel).lower()


def input_fn(
    request_body: str, request_content_type: str = JSON_CONTENT_TYPE
) -> List[dict]:
    """
    Standard input_fn for hosted model. Takes request body and content type directly from request
    Parse input json as records containing structure {'image': image_data, 'questionnaire':
    {questionnaire_key:val}},

    :param request_body: (str) json of records containing ['questionnaire', 'image']
    :param request_content_type: (str) content_type passed, should be 'application/json'
    :return: (list(dict)) of {'image': str, 'questionnaire': dict}
    """
    logger.info('Parsing input data')
    if request_content_type == JSON_CONTENT_TYPE:
        # Read the raw input json data
        parsed_data = json.loads(request_body)

        clean_data = [
            {camel_to_snake(key): val for key, val in datum.items()}
            for datum in parsed_data
        ]

        logger.info(f'Parsed input data: {parsed_data}')

        return clean_data
    else:
        raise ValueError(f"{request_content_type} not supported by script!")


def output_fn(
    prediction_output: List[Dict[str, Any]], accept: str = JSON_CONTENT_TYPE
) -> Tuple[str, str]:
    """
    Standard output_fn for hosted model
    Takes output from predict_fn and converts to Json

    :param prediction_output: ([{}]) Raw inputs augmented by Predictions
    :param accept: (str) Content type
    :return: (str, str) Jsonified results, content type
    """
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')


def model_fn(model_key: str) -> Model:
    """
    Standard model_fn for hosted model
    Loads trained model from model_dir

    :param model_key: (str) s3 key of stored model.pickle file
    :return: (Model) Model loaded with state from fit model file
    """
    logger.info('loading model')
    model = Model()
    model.load(f"{model_key}/model.pickle")
    logger.info('loaded model')

    return model


def predict_fn(input_data: List[dict], model: Model) -> List[dict]:
    """
    Basic predict_fn for model hosting. Predicts from input_data and model.

    Takes:
     input_data from input_fn
     model from model_fn

    :param input_data: (list(dict)) of {
        'image': raw_image_data,
        'questionnaire': dict of key:value
    }
    :param model: (Model)
    :return: ([{}]) Predictions as input object
    """
    input_df = pd.DataFrame(input_data)

    input_df = input_df.reindex(columns=model.list_variables).fillna(0)

    predictions = model.predict_proba(input_df)

    ret = []
    for input_object, prediction in zip(input_data, predictions):
        raw_data = {"id": input_object["id"], "prediction": list(prediction)}
        ret.append(raw_data)

    return ret
