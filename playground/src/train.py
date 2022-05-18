import os
import joblib
import pandas as pd

from file_handling import (
    read_metadata,
    read_data,
)
from utils.encoding import one_hot_encode, ordinal_encode
from utils.pipeline import run_training
from utils.regressors import (
    get_svr,
    get_xgboost,
)


REGRESSORS_MAPPING = {
    'svr': get_svr,
    'xgboost': get_xgboost,
}
REGRESSOR_NAME = os.getenv('KONAN_MODEL_REGRESSOR_NAME')


# ------------------------------------------------------------------- #
# Read the files
train, _ = read_data()
metadata = read_metadata()
# print("Train set size:", train.shape)


# ------------------------------------------------------------------- #
# Get log of target
# We use the numpy function log1p which  applies log(1+x) to all elements of the column
y_train: pd.Series = train.SalePrice


# ------------------------------------------------------------------- #
X_train = train.copy().drop(
    ['SalePrice'],
    axis=1,
)


# ------------------------------------------------------------------- #
# One-Hot-Encode Categorical Features
X_train, one_hot_encoder = one_hot_encode(
    df=X_train,
    columns=metadata['oneHotEncoding'],
    encoder=None,
)
# print("Train Features after OneHotEncoding set size:", X_train.shape)


# ------------------------------------------------------------------- #
# Label-Encode Ordinal Features
X_train, ordinal_encoder = ordinal_encode(
    df=X_train,
    columns=list(metadata['ordinalEncoding'].keys()),
    categories=list(metadata['ordinalEncoding'].values()),
    encoder=None,
)
# print("Train Features after OrdinalEncoding set size:", X_train.shape)

X_train = X_train.fillna(0)


# ------------------------------------------------------------------- #
# Train and test the model

fit_regressor, _, _ = run_training(
    regressor_name=REGRESSOR_NAME,
    regressor=REGRESSORS_MAPPING.get(REGRESSOR_NAME, None)(),
    X_train=X_train,
    y_train=y_train,
)


# ------------------------------------------------------------------- #
# Save model artifacts

joblib.dump(
    fit_regressor,
    f'artifacts/{REGRESSOR_NAME}/model.pkl',
)
joblib.dump(
    one_hot_encoder,
    f'artifacts/{REGRESSOR_NAME}/one_hot_encoder.pkl',
)
joblib.dump(
    ordinal_encoder,
    f'artifacts/{REGRESSOR_NAME}/ordinal_encoder.pkl',
)
