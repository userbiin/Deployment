<div align="center">

# Deployment

**Getting a better cancellation model into production without breaking anything**

A hotel group flags bookings that are likely to cancel and asks those guests for a deposit.<br>
I inherited the model with almost no record, built a challenger I could defend, gated it, shadowed it, fixed a broken rollback,<br>
and used MLflow as the record of every step.

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow_3.16-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn_1.8-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker_Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

<img src="https://github.com/user-attachments/assets/c7ff16a5-47d1-4449-9b54-187279b1b536" width="90%" alt="Rollback drill before and after the fix">

<sub>The rollback drill before the fix (top: rolling back changes nothing) and after it (bottom: v1 is back at once)</sub>

</div>

<br>

## Overview

- **Period** : 2026.10 – 2026.11 · Boise State University, CS 459 Machine Learning in Production
- **Type** : individual assignment (Assignment 3: Deployment Patterns)
- **Data** : 119,390 real bookings at a city hotel and a resort hotel, arrivals July 2015 – August 2017
- **Models** : the inherited champion is a logistic regression; the challenger is a `HistGradientBoostingClassifier`
- **Outcome** : the challenger passed the promotion gate, but I recommended **holding** the release (see [Final Decision](#final-decision-hold))

The starter project (data loading, metrics, the history of past runs, tests and the drill) was provided by the instructor.
What I wrote is listed in [What I Built](#what-i-built). The original assignment text is in [`README/SPEC.md`](README/SPEC.md).

<br>

## The Situation

The hotel group's revenue team uses a model to flag bookings that are likely to cancel. A flagged booking gets a deposit request, and the hotel may overbook against it. Both decisions cost money when the flag is wrong: a guest who was going to show up gets hassled for a deposit, or arrives to a full hotel and has to be sent somewhere else. The hotels have been clear: **a flag has to be right at least 80% of the time.**

The model in production is a logistic regression that flags a booking when its cancellation probability is **0.65** or higher. The person who built it has left, and the team's MLflow server holds everything that was ever recorded about it, which turns out to be very little. The story is set in **spring 2017**.

<br>

## What Had to Be Decided

1. What can, and can't, be trusted in the record the previous engineer left?
2. Which past experiments really beat the champion, and which only look like they do?
3. Is the new model good enough to ship, measured against a rule written in code rather than someone's judgment?
4. How does it behave on live traffic, when most outcomes haven't arrived yet?
5. If the release goes wrong, does rolling back actually restore the old behavior?

<br>

## Data and Periods

Every period is split by **arrival date**, because a booking's outcome is only final once its arrival date has passed.

| Period | Arrivals | Used for |
|---|---|---|
| `train` | Jul 2015 – Jun 2016 | Fitting |
| `validation` | Jul 2016 – Dec 2016 | Choosing anything: settings, threshold |
| `holdout` | Jan 2017 – Mar 2017 | Judging the final model and comparing it to the champion |
| `next_quarter` | Apr 2017 – Jun 2017 | Shadow traffic |

<br>

## How I Worked Through It

### 1. What did I inherit?

The champion is version 1 of the registered model `cancellations`, produced by run `logreg-final`. Its run recorded **no parameters, one metric (`auc` = 0.8747), no tags, no dataset and no source**, and the model was logged without an input schema.

| The champion's run | The champion's model artifact |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/644a0dac-6741-461e-a176-f189c63c9beb" alt="Champion run with no parameters"> | <img src="https://github.com/user-attachments/assets/30fece1d-21ba-4e55-925d-e6ffae0d16eb" alt="Champion MLmodel file"> |
| One metric, zero parameters, no dataset, no source | No `signature` in the `MLmodel` file |

- **What I could recover** by loading the artifact: the pipeline (one-hot encoding + scaling + `LogisticRegression`), the input columns and their order, the categories seen in training, and the library versions.
- **What I couldn't:** which data file it was trained on, which period it was trained and scored on, the threshold (it lived only in `config/serve.yaml`), and its precision or recall at that threshold.
- **The date doesn't add up.** The run is dated January 2017, but the holdout outcomes (January–March arrivals) weren't complete until April. Either the date is wrong or the AUC wasn't measured on the holdout.
- **The cost during an incident:** with no training data on record there is nothing to compare today's inputs against, and with no precision on record there is no "normal" to compare today's flags against.

<p align="center"><img src="https://github.com/user-attachments/assets/7e6b76cd-e564-4486-ab8f-f751265a8c24" width="70%" alt="Champion pipeline loaded from the artifact"></p>
<p align="center"><sub>The champion's pipeline, recovered by loading <code>model.skops</code></sub></p>

### 2. Making training rebuildable

I changed `train.py` so a run records everything needed to rebuild the model exactly. A grader rebuilds the model from the logged parameters alone and must get the same holdout ROC-AUC back.

| Logged | Why |
|---|---|
| `model.*` from `get_params()`, including `random_state=0` | Early stopping holds out a random 10% of the training rows; without a fixed seed, two fits on identical data differ |
| `features`, `period.*` | The exact columns, their order, and the training window |
| `data.sha256`, `data.rows`, first and last arrival | Proves which data file was used |
| `code.version` | SHA-256 of `train.py` + `features.py`, the two files that decide how the model is built |
| `lib.sklearn`, `lib.mlflow`, `lib.python` | Library defaults and implementations change between versions |
| Metrics, PR curves and confusion matrices for validation and holdout | Evidence for every later decision |
| The model with an `input_example` | Records the input schema |

**Threshold rule:** the lowest threshold from 0.30 to 0.95 (steps of 0.05) whose **validation** precision is at least 0.90. That gave **0.55**.

| At threshold 0.55 | Validation | Holdout |
|---|---|---|
| Precision | 0.919 | 0.828 |
| Recall | 0.338 | 0.563 |
| Flag rate | 13.6% | 22.7% |
| Cancel rate | 36.9% | 33.4% |

Aiming for 0.90 on validation left a margin of only **0.028** above the 0.80 floor on the holdout: the precision you aim for doesn't carry over unchanged to another period.

<p align="center"><img src="https://github.com/user-attachments/assets/b6e91e97-2904-4176-ad0b-ed30b78ca39e" width="90%" alt="Training run with 84 metrics"></p>

### 3. Comparing past experiments fairly

Five past runs logged an `auc` higher than the champion's. For each, I asked two separate questions: **is its number comparable** (measured on the holdout, split by arrival date, with booking-time features only), and **is its model usable** (never saw holdout bookings and uses only inputs known at booking time)?

| Run | Logged `auc` | Comparable? | Evidence | Model usable? | Holdout ROC-AUC, rescored |
|---|---|---|---|---|---|
| `hgb-0130` | **1.0000** | No · `uses_outcome` | `reservation_status` in `features`: the outcome itself | No | – |
| `hgb-0206` | 0.9541 | No · `random_split` | `split=random`, `evaluated_on=random_20pct` | No: training data may include holdout bookings | – |
| `hgb-0213` | 0.9037 | No · `post_booking_features` | `assigned_room_type`, `booking_changes`, `days_in_waiting_list` | No | – |
| `hgb-0220` | 0.8964 | **Yes** · `ok` | 22 booking-time features, arrival-date split, holdout | Yes | **0.8964** |
| `hgb-0227` | 0.8499 | No · `different_eval_period` | `evaluated_on=validation` | **Yes** | **0.8857** |
| champion (v1) | 0.8747 | – | – | – | **0.8747** |

| `hgb-0130`: a perfect AUC is the warning sign | The champion, rescored on the holdout at 0.65 |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/1924fdbf-8c9c-4a2d-a654-7521518f266e" alt="hgb-0130 run with auc 1"> | <img src="https://github.com/user-attachments/assets/6057c3c7-2ed7-49ad-b491-021c3f5f1b56" alt="Champion evaluation run"> |

- **The highest numbers were the ones that couldn't be compared.** An AUC of 1.0 is what a model scores when the answer is one of its inputs.
- **`hgb-0227` is the case where the two answers differ.** Its logged number came from the wrong period, but the model itself is clean. Measured on the holdout it scores 0.8857, not 0.8499, so it actually beats the champion.
- **The champion's logged 0.8747 matched my holdout rescore to four decimal places.** A different period moves the same model's AUC by 0.01 or more, so the number was almost certainly the holdout AUC, which also confirms that the run's January date is wrong.

### 4. A promotion gate

The hotels' rule became code in `gate.py`, and `register.py` runs it on evidence read from MLflow, never on numbers typed by hand.

| Rule | Check |
|---|---|
| 1. Precision floor | The challenger's precision overall, at the city hotel and at the resort hotel must each be at least 0.80 |
| 2. It has to be better | The challenger's recall must be at least 0.03 higher than the champion's |
| 3. No segment left behind | In every market segment with at least 500 holdout bookings, precision may not drop more than 0.05 |
| 4. Missing evidence is a no | A missing or non-finite metric blocks promotion instead of skipping the check |

`register.py` registers the training run as a new version, tags it with its own `threshold`, reads the champion's evidence from its Part 2 evaluation run, runs the gate, tags `gate.passed` and `gate.reasons`, and points `@challenger` at the new version without touching `@champion`. From here on, **a version's `threshold` tag is the single source of truth for the threshold it runs at.**

<p align="center"><img src="https://github.com/user-attachments/assets/7e0f745e-1282-4dab-aee2-0f0a340a40a3" width="90%" alt="register.py output: v2 is @challenger, gate passed"></p>

**In the hotels' terms, per 1,000 holdout bookings:**

| | Flags | Correct flags | Wrong flags |
|---|---|---|---|
| Champion (v1, threshold 0.65) | 146.3 | 130.0 | 16.3 |
| Challenger (v2, threshold 0.55) | 227.2 | 188.1 | 39.1 |
| **Difference** | +80.9 | **+58** | **+23** |

The challenger catches 58 more cancellations, at the cost of 23 more guests who will actually show up getting a wrong flag.

### 5. Shadow deployment

Pretending today is April 30, 2017, I ran both models side by side on every booking arriving April to June 2017 (17,621 bookings), without either one's flags reaching the hotels.

| Shadow report | Shadow run in MLflow |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/9eebef0c-aba8-4900-a91d-c289866d9d8a" alt="Shadow report JSON"> | <img src="https://github.com/user-attachments/assets/71485c52-0b48-42ca-bd54-b169ae8ce209" alt="Shadow run metrics"> |

- **Where they disagree:** on 14.7% of bookings, and about 90% of that is the challenger **adding** flags rather than overturning the champion's. Disagreement is highest in **Online TA (22.1%)**, the largest segment and the one where the challenger's holdout precision was only 0.681.
- **What the known outcomes say:** on April arrivals, the challenger kept precision above 0.80 (0.854 vs the champion's 0.939) while catching far more cancellations (recall 0.657 vs 0.430), the same pattern as on the holdout.
- **How much to trust it:** outcomes are known for only **31.3%** of the quarter, all from one month. May and June, and Online TA in particular, are exactly where the evidence is thinnest.
- **Both models flag more in spring than on the holdout** (champion 14.6% → 20.5%, challenger 22.7% → 32.2%), so releasing the challenger means hotels see about 1.5 times as many flags as they do today.

### 6. The rollback incident: what I fixed and restored

The team had replayed an incident from last quarter:

> 10:02 the challenger was released · 10:09 hotels report far more flags than usual · 10:14 ops roll back by moving `@champion` to the previous version · 10:30 nothing has changed · 10:45 ops restart the service, and still nothing has changed.

`make drill` reproduced it: after the rollback, both the running and the restarted service still flagged 32.6% of bookings instead of v1's 21.7%, and nothing reported which version was answering.

| Cause | Evidence |
|---|---|
| The service never looked at `@champion` | It loaded a pinned `model_uri` from `config/serve.yaml`, and the release overwrote it with `models:/cancellations/2`, so moving the alias changed nothing |
| It read the model only once, at startup | `Service.__init__` loaded the model and threshold, and `predict` never checked again |
| The release changed three things; the rollback undid one | The release moved the alias **and** rewrote the config's model URI **and** its threshold (0.65 → 0.55). The rollback only moved the alias, so even a restart read v2 and 0.55 again |
| Nothing recorded which version answered | Every line of the drill read `version: not reported` |

**The fix (`serve.py`):**
- On every request, the service asks the registry which version `@champion` points at. It loads a new model only when that version changes, and swaps version, model and threshold together so a request can never mix a new model with an old threshold.
- The threshold comes only from the version's `threshold` tag, and a version without one is refused.
- Every prediction includes `model_version`.
- If the registry is unreachable, a running service keeps serving the last version it confirmed; a new service refuses to start rather than guess.
- `config/serve.yaml` is no longer read.

After the fix, the same running service answers as v1 the moment `@champion` moves back, with no restart (see the image at the top).

**Prevention:** keep each value in exactly one place, make the rollback procedure revert everything the release changed, and rehearse a release and rollback (what `make drill` does) before anything reaches the hotels. At high traffic, I would move the registry lookup off the request path into a background check at a fixed interval, accepting that a rollback then takes effect at the next check.

<br>

## Final Decision: Hold

**The challenger passed the gate, but I recommended holding the release** and deciding again once May's outcomes are in.

**Why it is better:** on the holdout it catches 58 more cancellations per 1,000 bookings while keeping precision above the hotels' 0.80, and April's shadow traffic showed the same pattern.

**Why I held anyway:**
1. **Most outcomes aren't in yet.** Only 31% of the quarter's outcomes were known on April 30, all from April. There is no guarantee the results hold in May and June.
2. **The largest segment misses the standard under both models.** In Online TA, precision is 0.626 (champion) and 0.681 (challenger), and the challenger would raise wrong flags in that segment from 177 to 453. The current policy checks the 0.80 floor only overall and per hotel, so it can't catch this. Whether it should is a business decision.
3. **Hotels would see about 1.5 times as many flags.** Last quarter's incident started with exactly that report.

**Conditions for release:**
1. The revenue lead decides how to handle Online TA: apply the 80% floor per segment, apply a stricter threshold there, or keep the current policy and watch it closely.
2. A re-comparison at the end of May shows the challenger's precision still at or above 0.80, overall and for each hotel.
3. The hotels are told in advance that flags will increase.

**What to watch after release:** the flag rate overall and by segment and the version answering (daily, available immediately); hotel complaints about deposits and walked guests; and precision on bookings whose arrival date has passed (weekly), with a rollback if it stays below 0.80. Early precision only reflects short-lead-time bookings, so it shouldn't be over-read.

<br>

## What I Built

| File | What it does |
|---|---|
| [`src/train.py`](src/train.py) | Trains the challenger and logs everything needed to rebuild it; picks the threshold on validation |
| [`src/evaluate.py`](src/evaluate.py) | Scores any run or registered version on the holdout and logs the evidence as a new run |
| [`answers/part2.yaml`](answers/part2.yaml) | Comparability verdicts for the five past runs |
| [`src/gate.py`](src/gate.py) | `should_promote()`: the hotels' rule as code, one check per rule, failing closed on missing evidence |
| [`src/register.py`](src/register.py) | Registers a challenger, tags its threshold, gates it against the champion's logged evidence, moves `@challenger` |
| [`src/shadow.py`](src/shadow.py) | Loads models by alias, measures disagreement by group, and finds the bookings whose outcome is known on a given date |
| [`src/serve.py`](src/serve.py) | The prediction service, now following `@champion` and reporting its version |

<br>

## Run It

```bash
docker compose build
docker compose up -d mlflow           # MLflow UI at http://localhost:5050
docker compose run --rm app           # shell inside the container

make bootstrap                        # load the team's history
python src/train.py                   # Part 1
python src/evaluate.py --version 1 --threshold 0.65
python src/register.py --run-id <training run id>
python src/shadow.py --as-of 2017-04-30
make drill                            # replay the rollback incident
make test
```

<br>

## Data

`data/hotels.csv`: Nuno Antonio, Ana de Almeida, Luis Nunes, "Hotel booking demand datasets," *Data in Brief* 22 (2019) 41–49, [doi:10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126). Licensed CC BY 4.0.
