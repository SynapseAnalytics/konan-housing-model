import os
from time import sleep
from typing import List

from konan_sdk.sdk import KonanSDK
from konan_sdk.konan_types import KonanFeedbackSubmission
import pandas as pd


TEST_DATA_PATH = 'data/final/test.csv'
TEST_DATA_SAMPLE_START = 0
TEST_DATA_SAMPLE_SIZE = 100


def simulate_serving():
    df = pd.read_csv(
        TEST_DATA_PATH,
    ).reset_index(
        drop=True,
    )
    df = df.where(
        pd.notnull(
            df,
        ),
        None,
    )[TEST_DATA_SAMPLE_START:TEST_DATA_SAMPLE_START + TEST_DATA_SAMPLE_SIZE]

    X_test = df.drop(
        columns=[
            'SalePrice',
        ]
    )
    y_test = df[['SalePrice']]

    requests = X_test.to_dict(orient='records')
    ground_truths = y_test.to_dict(orient='records')

    sdk = KonanSDK(
        auth_url=os.environ['KONAN_AUTH_URL'],
        api_url=os.environ['KONAN_API_URL'],
        verbose=True,
    )

    sdk.login(
        email=os.environ['KONAN_USERNAME'],
        password=os.environ['KONAN_PASSWORD'],
    )

    prediction_uuids: List[str] = []
    for req in requests:
        prediction_uuid, _ = sdk.predict(
            deployment_uuid=os.environ['KONAN_DEPLOYMENT_UUID'],
            input_data=req,
        )
        prediction_uuids.append(prediction_uuid)

    sleep(30)

    sdk.feedback(
        deployment_uuid=os.environ['KONAN_DEPLOYMENT_UUID'],
        feedbacks=[
            KonanFeedbackSubmission(
                prediction_uuid=p,
                target=g,
            )
            for (p, g) in zip(prediction_uuids, ground_truths)
        ]
    )


if __name__ == '__main__':
    simulate_serving()
