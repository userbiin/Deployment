"""Train a cancellation model and record it in MLflow.

Usage (inside the container): python src/train.py

PART 1: this works, and it logs almost nothing. See SPEC.md, Part 1.
"""
# AI Used : Code debugging, searching mlflow API 

import mlflow
import hashlib
import platform
import sklearn
from pathlib import Path 
from sklearn.ensemble import HistGradientBoostingClassifier
from metrics import confusion_figure, evaluate, pr_curve_figure
from sklearn.metrics import roc_auc_score, precision_score

from features import FEATURES, build_pipeline, load, period, PERIODS, fingerprint
from tracking import EXPERIMENT, log_model

def choose_threshold(y, score): # precision = TP/(TP+FP)
    for i in range(30, 96, 5): 
        th = i/100
        if precision_score(y, score >= th, zero_division=0) >= 0.90: 
            return th 

SRC = Path(__file__).resolve().parent

def code_version(): 
    code = b"".join((SRC/f).read_bytes() for f in ("train.py", "features.py")) 
    return hashlib.sha256(code).hexdigest()[:12]


def main():
    mlflow.set_experiment(EXPERIMENT)
    df = load()
    train, val, hold = period(df, "train"), period(df, "validation"), period(df, "holdout")

    model = build_pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0))
    model.fit(train[FEATURES], train.is_canceled)
    
    score = {} # dict. key = period("validation" or "holdout"), value = predicted prob of val, hold
    for p, data in (("validation", val), ("holdout", hold)): # 1st iter : val, 2nd iter: hold 
        score[p] = (data, model.predict_proba(data[FEATURES])[:, 1]) # score = {validation: [0.6, 0.38, 0.7...], holdout: [0.4, 0.7...]}
    
    threshold = choose_threshold(val.is_canceled, score["validation"][1])

    with mlflow.start_run(run_name="hgb"):
        mlflow.set_tag("a3.part", "1") # set tag 
        mlflow.log_param("model.class", "HistGradientBoostingClassifier")
        mlflow.log_params({f"model.{k}": v for k, v in model.named_steps["classifier"].get_params().items()})
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_params({f"period.{p}": f"{start}..{end}" for p, (start, end) in PERIODS.items()})
        mlflow.log_params({f"data.{k}": v for k, v in fingerprint().items()})
        mlflow.log_params({"code.version": code_version(), "lib.sklearn": sklearn.__version__, 
                           "lib.mlflow": mlflow.__version__, "lib.python": platform.python_version(), "threshold": threshold})
        for name, (d, score) in score.items(): 
            mlflow.log_metrics(evaluate(d, score, threshold, prefix=name))
            mlflow.log_figure(pr_curve_figure(d, score, threshold), f"plots/{name}_pr_curve.png")
            mlflow.log_figure(confusion_figure(d, score, threshold), f"plots/{name}_confusion_matrix.png")
        log_model(model, input_example=train[FEATURES].head(5))

if __name__ == "__main__":
    main()
