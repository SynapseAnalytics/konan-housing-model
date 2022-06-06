import pandas as pd


def preprocess_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer the AMES dataset features according to https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1

    Args:
        features (pd.DataFrame): raw features

    Returns:
        pd.DataFrame: engineered features
    """
    _df = features.copy()

    _df['Functional'] = _df['Functional'].fillna('Typ')
    _df['KitchenQual'] = _df['KitchenQual'].fillna("TA")

    _df['CentralAir'] = _df['CentralAir'] == 'Y'

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in _df.columns:
        if _df[i].dtype in numeric_dtypes:
            numerics.append(i)
    _df.update(_df[numerics].fillna(0))

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics2 = []
    for i in _df.columns:
        if _df[i].dtype in numeric_dtypes:
            numerics2.append(i)

    _df['TotalSF'] = _df['TotalBsmtSF'] + _df['1stFlrSF'] + _df['2ndFlrSF']
    _df['Bathrooms'] = (_df['FullBath'] + (0.5 * _df['HalfBath']) + _df['BsmtFullBath'] + (0.5 * _df['BsmtHalfBath']))
    _df['TotalPorchSF'] = _df['OpenPorchSF'] + _df['3SsnPorch'] + _df['EnclosedPorch'] + _df['ScreenPorch'] + _df['WoodDeckSF']

    # simplified _df
    _df['Pool'] = _df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    _df['SecondFloor'] = _df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    _df['Garage'] = _df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    _df['Basement'] = _df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    _df['Fireplace'] = _df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    _df = _df.drop(
        columns=[
            'PoolArea',
            '2ndFlrSF',
            'GarageArea',
            'Fireplaces',
            'TotalBsmtSF',
            'FullBath',
            'HalfBath',
            'BsmtFullBath',
            'BsmtHalfBath',
            '1stFlrSF',
            'OpenPorchSF',
            '3SsnPorch',
            'EnclosedPorch',
            'ScreenPorch',
            'WoodDeckSF',
            '1stFlrSF',
            '2ndFlrSF',
        ],
    )

    return _df


def filter_features(
    df: pd.DataFrame,
):
    _df = df.copy().drop(
        columns=[
            'MSSubClass',
            'MSZoning',
            'Street',
            'Alley',
            'LotShape',
            'LandContour',
            'Utilities',
            'LotConfig',
            'LandSlope',
            'Neighborhood',
            'Condition1',
            'Condition2',
            'BldgType',
            'HouseStyle',
            'YearBuilt',
            'YearRemodAdd',
            'RoofStyle',
            'RoofMatl',
            'Exterior1st',
            'Exterior2nd',
            'MasVnrType',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinSF1',
            'BsmtFinType2',
            'BsmtFinSF2',
            'BsmtUnfSF',
            'LowQualFinSF',
            'PoolQC',
            'LotFrontage',
            'MasVnrArea',
            'Electrical',
            'GrLivArea',
            'BsmtFullBath',
            'BsmtHalfBath',
            'FullBath',
            'HalfBath',
            'FireplaceQu',
            'GarageType',
            'GarageYrBlt',
            'GarageFinish',
            'GarageQual',
            'GarageCond',
            'WoodDeckSF',
            'OpenPorchSF',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'MoSold',
            'YrSold',
            'SaleType',
            'PoolArea',
            '2ndFlrSF',
            'GarageArea',
            'Fireplaces',
            'TotalBsmtSF',
            '1stFlrSF',
            'SaleCondition',
        ],
    )

    return _df


def rename_features(
    df: pd.DataFrame,
):
    _df = df.copy().rename(
        columns={
            'BedroomAbvGr': 'Bedrooms',
            'KitchenAbvGr': 'Kitchens',
            'TotRmsAbvGrd': 'TotalRooms',
        }
    )
    return _df
