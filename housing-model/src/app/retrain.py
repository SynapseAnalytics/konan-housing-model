import joblib
import json
import sys
import yaml

from sklearn.model_selection import train_test_split
import pandas as pd

from konan_sdk.konan_service.serializers import KonanServicePredefinedMetricName

from utils.encoding import one_hot_encode, ordinal_encode
from utils.pipeline import run_testing, run_training
from utils.regressors import get_xgboost

RETRAINING_DIR_PATH = "/retraining"
METRICS_FILE_PATH = f"{RETRAINING_DIR_PATH}/metrics.json"
ARTIFACTS_DIR_PATH = f"{RETRAINING_DIR_PATH}/artifacts"
DATA_DIR_PATH = f"{RETRAINING_DIR_PATH}/data"
TRAINING_DATA_FILE_PATH = f"{DATA_DIR_PATH}/training.csv"
SERVING_DATA_FILE_PATH = f"{DATA_DIR_PATH}/serving.csv"


def retrain():
    # ------------------------------------------------------------------- #
    # Read the files

    training_data = None
    serving_data = None
    try:
        training_data = pd.read_csv(TRAINING_DATA_FILE_PATH)
    except Exception:
        pass
    try:
        serving_data = pd.read_csv(SERVING_DATA_FILE_PATH)
    except Exception:
        pass

    if training_data is None and serving_data is None:
        sys.exit('Unable to read both training and serving data')

    data = []
    if training_data is not None:
        data.append(training_data.reset_index(drop=True))
    if serving_data is not None:
        data.append(serving_data.reset_index(drop=True))

    # ------------------------------------------------------------------- #
    # Combine and split all data
    df = pd.concat(
        data,
        axis=0,
    )
    train, test = train_test_split(
        df,
        test_size=0.15,
        random_state=68,
    )
    with open(f'{ARTIFACTS_DIR_PATH}/metadata.yml') as file:
        metadata = yaml.safe_load(file)

    # ------------------------------------------------------------------- #
    y_train: pd.Series = train.SalePrice
    y_test: pd.Series = test.SalePrice

    X_train = train.drop(
        ['SalePrice'],
        axis=1,
    )
    X_test = test.drop(
        ['SalePrice'],
        axis=1,
    )

    # ------------------------------------------------------------------- #
    # One-Hot-Encode Categorial Features
    X_train, one_hot_encoder = one_hot_encode(
        df=X_train,
        columns=metadata['oneHotEncoding'],
        encoder=None,
    )
    X_test, _ = one_hot_encode(
        df=X_test,
        columns=metadata['oneHotEncoding'],
        encoder=one_hot_encoder,
    )
    # print("Train Features after OneHotEncoding set size:", X_train.shape)
    # print("Test Features after OneHotEncoding set size:", X_test.shape)

    # ------------------------------------------------------------------- #
    # Label-Encode Ordinal Features
    X_train, ordinal_encoder = ordinal_encode(
        df=X_train,
        columns=list(metadata['ordinalEncoding'].keys()),
        categories=list(metadata['ordinalEncoding'].values()),
        encoder=None,
    )
    X_test, _ = ordinal_encode(
        df=X_test,
        columns=list(metadata['ordinalEncoding'].keys()),
        categories=list(metadata['ordinalEncoding'].values()),
        encoder=ordinal_encoder,
    )
    # print("Train Features after OrindalEncoding set size:", X_train.shape)
    # print("Test Features after OrindalEncoding set size:", X_test.shape)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # ------------------------------------------------------------------- #
    # Train and test the model
    regressor, mape_train, mae_train = run_training(
        regressor_name="xgboost",
        regressor=get_xgboost(),
        X_train=X_train,
        y_train=y_train,
    )
    mape_test, mae_test = run_testing(
        regressor=regressor,
        X_test=X_test,
        y_test=y_test,
    )

    # ------------------------------------------------------------------- #
    # Save model artifacts
    joblib.dump(
        regressor,
        f'{ARTIFACTS_DIR_PATH}/model.pkl',
    )
    joblib.dump(
        one_hot_encoder,
        f'{ARTIFACTS_DIR_PATH}/one_hot_encoder.pkl',
    )
    joblib.dump(
        ordinal_encoder,
        f'{ARTIFACTS_DIR_PATH}/ordinal_encoder.pkl',
    )

    # ------------------------------------------------------------------- #
    # Save retraining metrics
    retraining_metrics = {
        'split': {
            'train': 0.85,
            'test': 0.15,
        },
        'evaluation': {
            'train': [
                {
                    'metric_name': 'mean_absolute_percentage_error',
                    'metric_value': mape_train,
                },
                {
                    'metric_name': KonanServicePredefinedMetricName.mae,
                    'metric_value': mae_train,
                },
            ],
            'test': [
                {
                    'metric_name': 'mean_absolute_error',
                    'metric_value': mape_test,
                },
                {
                    'metric_name': KonanServicePredefinedMetricName.mae,
                    'metric_value': mae_test,
                },
            ],
        },
    }

    with open(METRICS_FILE_PATH, 'w') as file:
        json.dump(retraining_metrics, file)


if __name__ == '__main__':
    retrain()
