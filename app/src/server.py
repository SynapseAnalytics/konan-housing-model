import joblib
import json
import pydantic
import yaml

import numpy as np
import pandas as pd

from konan_sdk.konan_service.models import KonanServiceBaseModel
from konan_sdk.konan_service.services import KonanService
from konan_sdk.konan_service.serializers import (
    KonanServiceBasePredictionRequest, KonanServiceBasePredictionResponse,
    KonanServiceBaseEvaluateRequest, KonanServiceBaseEvaluateResponse,
    KonanServiceEvaluation,
    KonanServicePredefinedMetricName,
)

from housing_enums import (
    DetailedRatingTypes,
    DrivewayTypes,
    FenceQualityTypes,
    FoundationTypes,
    FunctionalTypes,
    HeatingTypes,
    MiscellaneousFeaturesTypes,
    RatingTypes,
)
from utils.encoding import (
    one_hot_encode,
    ordinal_encode,
)
from utils.metrics import (
    mae,
    mape,
)


ARTIFACTS_DIR = '/app/artifacts'


class MyPredictionRequest(KonanServiceBasePredictionRequest):
    """Defines the schema of a prediction request
    Follow the convention of <field_name>: <type_hint>
    Check https://pydantic-docs.helpmanual.io/usage/models/ for more info
    """
    LotArea: pydantic.NonNegativeInt

    OverallQual: DetailedRatingTypes
    OverallCond: DetailedRatingTypes
    ExterQual: RatingTypes
    ExterCond: RatingTypes

    Foundation: FoundationTypes
    BsmtQual: RatingTypes
    BsmtCond: RatingTypes

    Heating: HeatingTypes
    HeatingQC: RatingTypes
    CentralAir: bool

    Bedrooms: pydantic.NonNegativeInt
    Kitchens: pydantic.NonNegativeInt
    KitchenQual: RatingTypes
    TotalRooms: pydantic.NonNegativeInt

    Functional: FunctionalTypes
    GarageCars: pydantic.NonNegativeInt

    PavedDrive: DrivewayTypes
    Fence: FenceQualityTypes = None

    MiscFeature: MiscellaneousFeaturesTypes = None
    MiscVal: pydantic.NonNegativeFloat = 0


class MyPredictionResponse(KonanServiceBasePredictionResponse):
    """Defines the schema of a prediction response
    Follow the convention of <field_name>: <type_hint>
    Check https://pydantic-docs.helpmanual.io/usage/models/ for more info
    """
    SalePrice: float


class MyModel(KonanServiceBaseModel):
    def __init__(self):
        """Add logic to initialize your actual model here

        Maybe load weights, connect to a database, etc ..
        """
        super().__init__()

        self.model = joblib.load(f'{ARTIFACTS_DIR}/model.pkl')
        self.one_hot_encoder = joblib.load(f'{ARTIFACTS_DIR}/one_hot_encoder.pkl')
        self.ordinal_encoder = joblib.load(f'{ARTIFACTS_DIR}/ordinal_encoder.pkl')

        self.metadata = yaml.safe_load(open(f'{ARTIFACTS_DIR}/metadata.yaml'))

    def predict(self, req: MyPredictionRequest) -> MyPredictionResponse:
        """Makes an intelligent prediction

        Args:
            req (MyPredictionRequest): raw request from API

        Returns:
            MyPredictionResponse: this will be the response returned by the API
        """
        df = pd.DataFrame({k: [v] for k, v in json.loads(req.json()).items()})
        df, _ = one_hot_encode(
            df=df,
            columns=self.metadata['oneHotEncoding'],
            encoder=self.one_hot_encoder,
        )
        df, _ = ordinal_encode(
            df=df,
            columns=list(self.metadata['ordinalEncoding'].keys()),
            categories=list(self.metadata['ordinalEncoding'].values()),
            encoder=self.ordinal_encoder,
        )
        df = df.fillna(0)
        print(df)

        # Use your logic to make a prediction
        sale_price = float(np.expm1(self.model.predict(df)))
        # Create a MyPredictionResponse object using kwargs
        sample_prediction = MyPredictionResponse(
            SalePrice=sale_price,
        )

        # Optionally postprocess the prediction here
        return sample_prediction

    def evaluate(self, req: KonanServiceBaseEvaluateRequest) -> KonanServiceBaseEvaluateResponse:
        """Evaluates the model based on passed predictions and their ground truths

        Args:
            req (KonanServiceBaseEvaluateRequest): includes passed predictions and their ground truths

        Returns:
            KonanServiceEvaluateResponse: the evaluation(s) of the model based on some metrics
        """
        # Use your logic to make an evaluation
        # Create a KonanServiceBaseEvaluateResponse object using kwargs
        sample_evaluation = KonanServiceBaseEvaluateResponse(
            # results should be a list of KonanServiceEvaluation objects
            # define each KonanServiceEvaluation object using kwargs
            results=[
                KonanServiceEvaluation(
                    metric_name="mean_absolute_percentage_error",
                    metric_value=mape(
                        y_pred=np.array([x.prediction.SalePrice for x in req.data]),
                        y_true=np.array([x.target.SalePrice for x in req.data]),
                    )
                ),
                KonanServiceEvaluation(
                    metric_name=KonanServicePredefinedMetricName.mae,
                    metric_value=mae(
                        y_pred=np.array([x.prediction.SalePrice for x in req.data]),
                        y_true=np.array([x.target.SalePrice for x in req.data]),
                    )
                ),
            ],
        )
        return sample_evaluation


app = KonanService(MyPredictionRequest, MyPredictionResponse, MyModel)
