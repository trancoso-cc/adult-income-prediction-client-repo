"""Adult Income Prediction Client.

Thin client that consumes the Azure ML `adult-binary-classifier-1` deployment
on the `trancoso-ml-models-aqloo` online endpoint.
"""

from adult_income_client.client import AdultIncomeClient

__all__ = ["AdultIncomeClient"]
