from typing import List, Optional, Tuple

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd


def one_hot_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    encoder: OneHotEncoder = None,
) -> Tuple[pd.DataFrame, OneHotEncoder]:

    _df = df.copy()
    _columns = columns if columns is not None else _df.select_dtypes('object').columns.to_list()

    _df_categorical = _df[_columns]

    _encoder: OneHotEncoder = encoder or OneHotEncoder(
        dtype=np.int64,
        handle_unknown='ignore',
        min_frequency=0.05,
        max_categories=20,
    ).fit(_df_categorical)

    _df_categorical_transformed = _encoder.transform(_df_categorical).toarray()
    _columns_transformed = _encoder.get_feature_names_out(_columns)

    _df_transformed = pd.concat(
        [
            _df.drop(_columns, axis=1).reset_index(),
            pd.DataFrame(
                _df_categorical_transformed,
                columns=_columns_transformed,
            ).astype(int).reset_index(),
        ],
        axis=1,
    )

    return _df_transformed, _encoder


def ordinal_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    categories: Optional[List[List[str]]],
    encoder: OrdinalEncoder = None,
) -> Tuple[pd.DataFrame, OrdinalEncoder]:

    _df = df.copy()
    _columns = columns if columns is not None else _df.select_dtypes('object').columns.to_list()
    _df_categorical = _df[_columns]

    _encoder: OrdinalEncoder = encoder or OrdinalEncoder(
        categories=categories if categories is not None else 'auto',
        dtype=np.int64,
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    ).fit(_df_categorical)

    _df[_columns] = _encoder.transform(_df_categorical)

    return _df, _encoder
