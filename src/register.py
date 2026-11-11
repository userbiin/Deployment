"""Part 3: register a challenger, gate it against the champion, and record the result.

Usage: python src/register.py --run-id <the training run you want to promote>

What this must do is in SPEC.md, Part 3. In short:
  1. Register the run's model as a new version of the registered model.
  2. Put the run's threshold on the new version as a tag named "threshold".
  3. Find the champion's evaluation run (from Part 2) and read its holdout metrics.
  4. Read the challenger's holdout metrics from its training run.
  5. Call gate.should_promote(champion_metrics, challenger_metrics).
  6. Tag the new version: gate.passed ("true" or "false") and gate.reasons.
  7. Point the @challenger alias at the new version. Do not move @champion here.
"""
# AI Used : drafted by Claude, reviewed, modified and run by me
import argparse
import mlflow
from mlflow import MlflowClient

from gate import should_promote
from tracking import MODEL_NAME, EXPERIMENT


def holdout_metrics(run):
  return {k: v for k, v in run.data.metrics.items() if k.startswith("holdout.")}

def champion_evidence(client):
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    exp_id = client.get_experiment_by_name(EXPERIMENT).experiment_id
    runs = client.search_runs([exp_id], f"tags.`a3.part` = '2' and tags.evaluates_version = '{champion.version}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"No Part 2 evaluation of champion v{champion.version}; run src/evaluate.py first")
    return holdout_metrics(runs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    client = MlflowClient()
    
    run_id = args.run_id
    
    run = client.get_run(run_id)
    threshold = run.data.params.get("threshold")
    if threshold is None:
      raise SystemExit(f"Run {run_id} has no threshold param; train it with src/train.py")
    champion = champion_evidence(client)
    
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    client.set_model_version_tag(MODEL_NAME, version, "threshold", threshold)
    
    decision = should_promote(champion, holdout_metrics(run))
    client.set_model_version_tag(MODEL_NAME, version, "gate.passed", str(decision.promote).lower())
    client.set_model_version_tag(MODEL_NAME, version, "gate.reasons", "; ".join(decision.reasons))
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"v{version} is @challenger. gate passed: {decision.promote}. {'; '.join(decision.reasons)}")

if __name__ == "__main__":
    main()
