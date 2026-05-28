# Adult Income Prediction Client

**Project:** adult-income-prediction
**System:** Adult Income Prediction Client
**Repo role:** Inference client (serving consumer)

## Overview

Thin Python client that calls the **adultbinaryservice** Azure Machine Learning
web service. Given a US Census-style demographic record, it returns a
prediction of whether the individual's annual income is greater than $50,000
(the classic UCI Adult dataset task).

The model itself — `adult_binary_classifier_model_03_10_2025` — is trained
and deployed by the ML platform team in the Azure ML workspace. This
repository contains **only the inference client** — it does not train,
register, or own the model. The model artifact and its training pipeline live
in the Azure ML workspace; this client just consumes the deployed web
service.

Downstream consumers (HR analytics dashboards, candidate pre-screening tools)
call `predict()` from this package and treat the Azure ML scoring URI as the
authoritative scoring service.

## Azure ML web service consumed

This client consumes a single Azure Machine Learning web service backed by
Azure Container Instances:

- **Scoring URI:** `http://712f9a93-9514-4df8-bf24-8306675775f0.eastus2.azurecontainer.io/score`
- **Service name:** `adultbinaryservice`
- **Model:** `adult_binary_classifier_model_03_10_2025` (version 1)
- **Region:** East US 2
- **Workspace API:** `https://eastus2.api.azureml.ms`

Every inference request is POSTed to the **scoring URI** above. Authentication
uses an API key sent in the `Authorization: Bearer <key>` header.

This client is **fully dependent** on that service — there is no local
fallback, no cached model, no alternative provider. If the
`adultbinaryservice` Azure ML web service is unavailable, this client cannot
serve predictions.

## Architecture

```
HR / analytics caller
        │
        │  AdultIncomeClient.predict(features)
        ▼
src/adult_income_client/client.py
        │
        │  POST <scoring URI>
        │  Authorization: Bearer <api key>
        ▼
Azure ML web service  (adultbinaryservice)
        │
        ▼
adult_binary_classifier_model_03_10_2025  (scikit-learn, ACI-hosted)
```

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env  # then fill in AZURE_ML_API_KEY
python -m adult_income_client.example
```

Programmatic use:

```python
from adult_income_client import AdultIncomeClient

clf = AdultIncomeClient()  # reads AZURE_ML_API_KEY from env
result = clf.predict({
    "age": 39,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "race": "White",
    "sex": "Male",
    "hours_per_week": 40,
    "native_country": "United-States",
})
print(result)  # → {"prediction": "<=50K"}
```

## Configuration

The scoring URI, service name, and API key are read from environment
variables. See `.env.example` and `src/adult_income_client/config.py`. The
defaults match the deployed `adultbinaryservice` web service listed above;
you should only override them if pointing at a different web service.

## Why this is a separate repository

The scoring service is owned by the ML platform team and deployed via the
Azure ML workspace. Client applications (this repo, and other downstream
consumers) are intentionally decoupled from training and deployment — they
only carry the consume-side contract: how to format the request, what URL
to call, and how to authenticate.
