"""Runnable example: score one record against the Azure ML web service.

Usage:
    cp .env.example .env       # fill in AZURE_ML_API_KEY
    python -m adult_income_client.example 

Targets the `adultbinaryservice` Azure ML web service.
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

from adult_income_client.client import AdultIncomeClient
from adult_income_client.config import (
    AZURE_ML_MODEL_NAME,
    AZURE_ML_SCORING_URI,
    AZURE_ML_SERVICE_NAME,
)


def main() -> None:
    load_dotenv()

    print(f"Service:     {AZURE_ML_SERVICE_NAME}")
    print(f"Model:       {AZURE_ML_MODEL_NAME}")
    print(f"Scoring URI: {AZURE_ML_SCORING_URI}")
    print()

    client = AdultIncomeClient()
    result = client.predict(
        {
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        }
    )

    print("Prediction:")
    print(json.dumps({"income": result.prediction}, indent=2))


if __name__ == "__main__":
    main()
