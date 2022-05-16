from sklearn.metrics import mean_absolute_percentage_error
import numpy as np


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    nan_indices = np.isnan(y_pred)
    pos_inf_indices = np.isposinf(y_pred)
    neg_inf_indices = np.isneginf(y_pred)

    drop_indices = nan_indices | pos_inf_indices | neg_inf_indices

    return mean_absolute_percentage_error(
        y_true=y_true[~drop_indices],
        y_pred=y_pred[~drop_indices],
    )
