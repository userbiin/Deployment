"""Part 2: score an existing model on the holdout and record the evidence in MLflow.

Usage: python src/evaluate.py --run-id <id> [--threshold 0.5]
       python src/evaluate.py --version <n>   [--threshold 0.5]

See SPEC.md, Part 2, for what the evaluation run must contain.
"""
# AI Used : Code debugging
 
import argparse

import mlflow
from mlflow import MlflowClient

from features import fingerprint, load, period 
from metrics import evaluate
from tracking import EXPERIMENT, MODEL_NAME


def main():
    mlflow.set_experiment(EXPERIMENT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--version")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    run_id, version, threshold = args.run_id, args.version, args.threshold
    
    if bool(run_id) == bool(version): # accept only version or runid 
            ap.error("give exactly one of --run-id or --version")
    
    if version: # if given version
        run_id = MlflowClient().get_model_version(MODEL_NAME, version).run_id
        uri = f"models:/{MODEL_NAME}/{version}"
    else: # given runid
        run_id = run_id
        uri = f"runs:/{run_id}/model"
    
    model = mlflow.sklearn.load_model(uri) # loading model 
    hold = period(load(), "holdout")
    score = model.predict_proba(hold[list(model.feature_names_in_)])[:, 1] 
    # predict_proba(...)[:, 1] : the arr of cancellation proba (ex. [0.21, 0.88, ...])
    with mlflow.start_run(run_name=f"eval v{version}" if version else f"eval-{run_id[:8]}"):
        mlflow.set_tag("a3.part", "2") # set tag 
        mlflow.set_tag("evaluates_run", run_id)
        if args.version:
            mlflow.set_tag("evaluates_version", version)
        mlflow.log_param("threshold", threshold)
        mlflow.log_params({f"data.{k}": v for k, v in fingerprint().items()})
        mlflow.log_metrics(evaluate(hold, score, threshold, prefix="holdout"))


if __name__ == "__main__":
    main()
