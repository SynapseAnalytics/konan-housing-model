from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from utils.metrics import mae, mape


def run_testing(
    regressor: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    _y_pred_test = _predict(
        model=regressor,
        X=X_test,
    )
    _mape_test = mape(
        y_true=y_test,
        y_pred=_y_pred_test,
    )
    _mae_test = mae(
        y_true=y_test,
        y_pred=_y_pred_test,
    )

    return _mape_test, _mae_test


def run_training(
    regressor_name: str,
    regressor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    print(datetime.now(), f'Starting to fit {regressor_name}')

    _rmse_cv = _cross_validate(
        regressor=regressor,
        X_train=X_train,
        y_train=y_train,
    )
    print(f"Model {regressor_name} score: {np.mean(_rmse_cv)} ({np.std(_rmse_cv)})")

    _fit_model = _train(
        regressor=regressor,
        X_train=X_train,
        y_train=y_train,
    )

    _y_pred_train = _predict(
        model=_fit_model,
        X=X_train,
    )

    _mape_train = mape(
        y_true=y_train,
        y_pred=_y_pred_train,
    )
    _mae_train = mae(
        y_true=y_train,
        y_pred=_y_pred_train,
    )
    print(f"Model {regressor_name} training MAPE: {_mape_train}")
    print(f"Model {regressor_name} training MAE: {_mae_train}")

    return _fit_model, _mape_train,_mae_train


def _cross_validate(
    regressor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    rmse = np.sqrt(
        -cross_val_score(
            regressor,
            np.array(X_train),
            np.array(np.log1p(y_train)),
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
        )
    )
    return (rmse)


def _train(
    regressor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    _fit_model = regressor.fit(
        np.array(X_train),
        np.array(np.log1p(y_train)),
    )
    return _fit_model


def _predict(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    _raw_output = model.predict(X)
    _denormalized_output = np.expm1(_raw_output)
    return _denormalized_output
