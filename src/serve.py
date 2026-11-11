"""The prediction service the hotels call. Each booking gets a cancellation score and a flag.

Part 5 is about this file. Start from the incident report in SPEC.md, not from here.
"""
# AI Used : rafted by Claude, reviewed and run by me
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import logging

from features import FEATURES
from tracking import MODEL_NAME

log = logging.getLogger(__name__)


class Service:
    def __init__(self, alias="champion"):
        self.alias = alias
        self.client = MlflowClient()
        self.live = None  # (version, model, threshold), always swapped together
        self._follow_alias()  # an unservable version fails at startup, not on the first request

    def _follow_alias(self):
        try:
            mv = self.client.get_model_version_by_alias(MODEL_NAME, self.alias)
        except Exception:
            if self.live is None:
                raise
            log.warning("registry unreachable, still serving v%s", self.live[0])  # keep the last known good version
            return self.live
        if self.live and self.live[0] == mv.version:
            return self.live
        if "threshold" not in mv.tags:
            raise RuntimeError(f"v{mv.version} has no threshold tag; refusing to serve it")
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")  # exactly the version the alias resolved to
        self.live = (mv.version, model, float(mv.tags["threshold"]))
        return self.live

    def predict(self, bookings):
        version, model, threshold = self._follow_alias()  # one lookup per request, so a moved alias takes effect at once
        score = model.predict_proba(bookings[FEATURES])[:, 1]
        return [{"booking_id": int(b), "cancel_probability": float(s), "flag": bool(s >= threshold),
                 "model_version": version}
                for b, s in zip(bookings["booking_id"], score)]