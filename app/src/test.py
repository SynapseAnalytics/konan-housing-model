from konan_sdk.konan_service.serializers import KonanServiceBaseEvaluateRequest, KonanServiceBaseFeedback
import pandas as pd

from server import MyPredictionRequest, MyModel, MyPredictionResponse


TEST_DATA_PATH = 'data/final/test.csv'
TEST_DATA_SAMPLE_SIZE = 10


def test_model():
    df = pd.read_csv(
        TEST_DATA_PATH
    ).sample(
        TEST_DATA_SAMPLE_SIZE,
    ).reset_index(
        drop=True,
    )
    df = df.where(
        pd.notnull(
            df,
        ),
        None,
    )

    requests = [
        MyPredictionRequest(
            **d
        ) for d in df.to_dict('records')
    ]
    targets = [
        MyPredictionResponse(
            **d
        ) for d in df.to_dict('records')
    ]

    model = MyModel()
    predictions = [
        model.predict(req)
        for req in requests
    ]
    evaluation = model.evaluate(
        req=KonanServiceBaseEvaluateRequest(
            data=[
                KonanServiceBaseFeedback(
                    prediction=prediction,
                    target=target,
                )
                for (prediction, target) in zip(predictions, targets)
            ],
        ),
    )

    for eval in evaluation.results:
        print(f'{eval.metric_name}: {eval.metric_value}')


if __name__ == '__main__':
    test_model()
