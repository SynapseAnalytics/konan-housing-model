from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def get_xgboost():
    return XGBRegressor(
        learning_rate=0.01,
        n_estimators=2048,
        max_depth=3,
        min_child_weight=0,
        gamma=0,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='reg:squarederror',
        nthread=-1,
        scale_pos_weight=1,
        seed=27,
        reg_alpha=0.00006,
    )


def get_svr():
    return make_pipeline(
        RobustScaler(),
        SVR(
            C= 20,
            epsilon= 0.008,
            gamma=0.0003,
        ),
    )
