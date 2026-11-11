"""Part 3: the promotion policy, written down.

should_promote() takes two dicts of holdout metrics, using the names produced by
metrics.evaluate(..., prefix="holdout"). It must not talk to MLflow: register.py
fetches the metrics and passes them in. That keeps the policy testable on its own.

The policy is specified exactly in SPEC.md, Part 3. Implement it as written.
"""
# AI Used : drafted by Claude, reviewed and run by me

import math
from dataclasses import dataclass, field

from numbers import Real
from metrics import GROUPS

PRECISION_FLOOR = 0.80          # overall and per hotel, challenger only
MIN_RECALL_GAIN = 0.03          # challenger recall minus champion recall, overall
MAX_SEGMENT_PRECISION_DROP = 0.05
MIN_SEGMENT_SIZE = 500          # segments with fewer holdout bookings are not checked
TOLERANCE = 1e-9                # for comparisons that land exactly on a limit

SEGMENTS = [g for g in GROUPS if g.startswith("segment.")]


@dataclass
class Decision:
    promote: bool
    reasons: list = field(default_factory=list)   # why not, one string per failed rule


def _get(metrics, name, missing, who):
    value = metrics.get(name)
    if not isinstance(value, Real) or not math.isfinite(value):
        missing.append(f"{who} {name}")
        return None
    return float(value)

def should_promote(champion: dict, challenger: dict) -> Decision:
    reasons, missing = [], []
    
    # Rule 1: precision floor, overall and per hotel
    for name in ("holdout.precision", "holdout.hotel.city.precision", "holdout.hotel.resort.precision"):
        p = _get(challenger, name, missing, "challenger")
        if p is not None and p < PRECISION_FLOOR - TOLERANCE:
            reasons.append(f"precision floor: {name} is {p:.3f}, below {PRECISION_FLOOR}")
    
    # Rule 2: recall must improve by at least MIN_RECALL_GAIN
        new = _get(challenger, "holdout.recall", missing, "challenger")
        old = _get(champion, "holdout.recall", missing, "champion")
        if new is not None and old is not None and new - old < MIN_RECALL_GAIN - TOLERANCE:
            reasons.append(f"recall gain: {new:.3f} vs champion {old:.3f}, needs +{MIN_RECALL_GAIN}")
    
    # Rule 3: no large segment loses more than MAX_SEGMENT_PRECISION_DROP of precision
    for seg in SEGMENTS:
        n = _get(champion, f"holdout.{seg}.n", missing, "champion")
        if n is None or n < MIN_SEGMENT_SIZE:
            continue
        new = _get(challenger, f"holdout.{seg}.precision", missing, "challenger")
        old = _get(champion, f"holdout.{seg}.precision", missing, "champion")
        if new is not None and old is not None and new < old - MAX_SEGMENT_PRECISION_DROP - TOLERANCE:
            reasons.append(f"{seg}: precision {new:.3f} vs champion {old:.3f}, "
                           f"drops more than {MAX_SEGMENT_PRECISION_DROP}")
    
    # Rule 4: any evidence a rule needed but couldn't read is a no
    if missing:
        reasons.append(f"missing evidence: {', '.join(missing)}")
    
    return Decision(promote=not reasons, reasons=reasons)        
