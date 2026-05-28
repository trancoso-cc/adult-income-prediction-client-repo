"""Configuration for the Adult Income Prediction Client.

All values below point at the Azure Machine Learning web service this client
consumes. They are constants by default and can be overridden via environment
variables only when pointing the client at a different service (e.g. for
staging or a re-deployed service).
"""

import os

# ---------------------------------------------------------------------------
# Azure Machine Learning web service consumed by this client.
# ---------------------------------------------------------------------------
# The scoring URI below is the `scoringUri` field returned by the Azure ML
# control-plane API for the `adultbinaryservice` web service. Every inference
# call this client makes is a POST to this exact URL.
AZURE_ML_SCORING_URI = os.environ.get(
    "AZURE_ML_SCORING_URI",
    "http://712f9a93-9514-4df8-bf24-8306675775f0.eastus2.azurecontainer.io/score",
)

# Name of the Azure ML web service (control-plane identity).
AZURE_ML_SERVICE_NAME = os.environ.get(
    "AZURE_ML_SERVICE_NAME",
    "adultbinaryservice",
)

# Name of the model registered in the workspace and served by the web
# service above.
AZURE_ML_MODEL_NAME = os.environ.get(
    "AZURE_ML_MODEL_NAME",
    "adult_binary_classifier_model_03_10_2025",
)

# Workspace control-plane API. Used for logging / observability only — the
# runtime data plane (the scoring URI above) is enough to call the model.
AZURE_ML_WORKSPACE_API = os.environ.get(
    "AZURE_ML_WORKSPACE_API",
    "https://eastus2.api.azureml.ms",
)
AZURE_ML_REGION = os.environ.get("AZURE_ML_REGION", "eastus2")

# API key for the web service. The service uses key-based auth, so every
# request must include `Authorization: Bearer <key>`. Read from env so the
# key is not committed.
AZURE_ML_API_KEY = os.environ.get("AZURE_ML_API_KEY", "")
