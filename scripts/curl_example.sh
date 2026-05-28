#!/usr/bin/env bash
#
# Smoke-test the Azure ML web service with a single record using curl.
# Loads AZURE_ML_API_KEY from .env (or falls back to the env var).
#
# Usage:
#   ./scripts/curl_example.sh

set -euo pipefail

if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

if [[ -z "${AZURE_ML_API_KEY:-}" ]]; then
  echo "AZURE_ML_API_KEY is not set. Create .env from .env.example." >&2
  exit 1
fi

SCORING_URI="${AZURE_ML_SCORING_URI:-http://712f9a93-9514-4df8-bf24-8306675775f0.eastus2.azurecontainer.io/score}"

curl -sS -X POST "$SCORING_URI" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_ML_API_KEY" \
  -d '{
    "input_data": {
      "columns": [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country"
      ],
      "index": [0],
      "data": [[
        39,"State-gov",77516,"Bachelors",13,
        "Never-married","Adm-clerical","Not-in-family","White","Male",
        2174,0,40,"United-States"
      ]]
    }
  }'
echo
