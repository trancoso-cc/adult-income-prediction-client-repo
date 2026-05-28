"""HTTP client that consumes the adultbinaryservice Azure ML web service.

This module owns the consume-side contract of the integration:

  - Where to POST the request   →  `AZURE_ML_SCORING_URI`
  - How to authenticate         →  `Authorization: Bearer <api key>`
  - How to shape the payload    →  `{"input_data": {"columns": [...], "data": [[...]]}}`

The model itself (a scikit-learn binary classifier) lives entirely inside the
Azure ML workspace; this client never loads it locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from adult_income_client.config import (
    AZURE_ML_API_KEY,
    AZURE_ML_SCORING_URI,
    AZURE_ML_SERVICE_NAME,
)

# Order matches the schema the deployed scikit-learn pipeline expects (UCI
# Adult dataset columns, excluding the income label).
ADULT_FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]


@dataclass(frozen=True)
class PredictionResult:
    """Result of a single inference call against the Azure ML web service."""

    prediction: str  # ">50K" or "<=50K"
    raw_response: dict[str, Any]


class AdultIncomeClient:
    """Inference client for the `adultbinaryservice` Azure ML web service.

    Every `predict()` call is a fresh HTTPS POST to the service's scoring
    URI. There is no local fallback — if the service is unavailable,
    predict() raises.
    """

    def __init__(
        self,
        api_key: str | None = None,
        scoring_uri: str = AZURE_ML_SCORING_URI,
        service_name: str = AZURE_ML_SERVICE_NAME,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.scoring_uri = scoring_uri
        self.service_name = service_name
        self.timeout_seconds = timeout_seconds

        resolved_key = api_key or AZURE_ML_API_KEY
        if not resolved_key:
            raise ValueError(
                "AZURE_ML_API_KEY is required (pass api_key=... or set env var)."
            )
        self._api_key = resolved_key

    def predict(self, features: dict[str, Any]) -> PredictionResult:
        """Score one record against the Azure ML web service.

        `features` is a dict whose keys are a subset of `ADULT_FEATURE_COLUMNS`.
        Missing columns are sent as None and let the service's preprocessing
        pipeline impute them.
        """
        row = [features.get(col) for col in ADULT_FEATURE_COLUMNS]
        payload = {
            "input_data": {
                "columns": ADULT_FEATURE_COLUMNS,
                "index": [0],
                "data": [row],
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        response = requests.post(
            self.scoring_uri,
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        # The web service returns a list of labels — one per input row.
        if isinstance(body, list) and body:
            label = str(body[0])
        elif isinstance(body, dict) and "predictions" in body:
            label = str(body["predictions"][0])
        else:
            raise RuntimeError(
                f"Unexpected response shape from {self.scoring_uri}: {body!r}"
            )

        return PredictionResult(prediction=label, raw_response={"response": body})
