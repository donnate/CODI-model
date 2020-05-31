import numpy as np
from keras import backend as K


def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """

    class_weights = np.array([1.0, 2.0, 2.0, 2.0])

    eps = K.epsilon()

    y_pred = K.clip(y_pred, eps, 1.0 - eps)

    out = -(
        y_true * K.log(y_pred) * class_weights
        + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights
    )

    return K.mean(out, axis=-1)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [1.0, 2.0, 2.0, 2.0]

    epsilon = 1e-7

    preds = np.clip(preds, epsilon, 1 - epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return -loss_samples.mean()
