# Generates synthetic arithmetic with stratified answer-length sampling.
# 1- and 2-digit buckets are enumerated exhaustively (spaces are small);
# 3- and 4-digit buckets use random sampling.
# Each (a, op, b) triple appears in exactly one split — seen sets enforce this.

import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FIELDNAMES = ["equation", "operand_1", "operation", "operand_2", "answer"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BUCKETS: dict[int, tuple[int, int]] = {
    1: (0, 9),
    2: (10, 99),
    3: (100, 999),
    4: (1000, 1998),  # addition only; max = 999+999
}

# Training: target sample counts per digit bucket.
# Addition 1-digit has ≤55 unique pairs; 44 go to train, 11 reserved for val/test.
_ADD_TRAIN: dict[int, int] = {1: 44,    2: 2_000, 3: 9_000, 4: 9_000}
_SUB_TRAIN: dict[int, int] = {1: 3_000, 2: 5_000, 3: 12_000}

# Val and test: explicit targets for small buckets guarantee measurable coverage.
_ADD_VAL:  dict[int, int] = {1: 5,   2: 200, 3: 1_100, 4: 1_200}
_ADD_TEST: dict[int, int] = {1: 6,   2: 200, 3: 1_100, 4: 1_200}
_SUB_VAL:  dict[int, int] = {1: 100, 2: 500, 3: 1_900}
_SUB_TEST: dict[int, int] = {1: 100, 2: 500, 3: 1_900}


# ---------------------------------------------------------------------------
# Row helper
# ---------------------------------------------------------------------------

def _row(a: int, operation: str, b: int) -> dict:
    answer = a + b if operation == "+" else a - b
    return {
        "equation": f"{a}{operation}{b}={answer}",
        "operand_1": a,
        "operation": operation,
        "operand_2": b,
        "answer": answer,
    }


def _digit_count(answer) -> int:
    n = int(answer)
    if n < 10:   return 1
    if n < 100:  return 2
    if n < 1000: return 3
    return 4


# ---------------------------------------------------------------------------
# Full-space enumerators (pure; do NOT modify seen)
# ---------------------------------------------------------------------------

def _all_add(lo: int, hi: int) -> list[dict]:
    """Enumerate every unique addition pair with answer in [lo, hi]."""
    rows: list[dict] = []
    for s in range(lo, hi + 1):
        for a in range(min(s + 1, 1000)):
            b = s - a
            if b < 1000:
                rows.append(_row(a, "+", b))
    return rows


def _all_sub(lo: int, hi: int) -> list[dict]:
    """Enumerate every unique subtraction pair with answer in [lo, hi]."""
    rows: list[dict] = []
    for d in range(lo, hi + 1):
        for b in range(1000):
            a = b + d
            if a >= 1000:
                break
            rows.append(_row(a, "-", b))
    return rows


# ---------------------------------------------------------------------------
# Bucket sampler (random sampling for large spaces; modifies seen)
# ---------------------------------------------------------------------------

def _sample(
    rng: random.Random,
    operation: str,
    lo: int,
    hi: int,
    count: int,
    seen: set,
) -> list[dict]:
    rows: list[dict] = []
    max_tries = count * 60
    for _ in range(max_tries):
        if len(rows) >= count:
            break
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        if operation == "-" and a < b:
            a, b = b, a
        answer = a + b if operation == "+" else a - b
        if not (lo <= answer <= hi):
            continue
        key = (a, operation, b)
        if key not in seen:
            seen.add(key)
            rows.append(_row(a, operation, b))
    return rows


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def _build_split(
    rng: random.Random,
    operation: str,
    targets: dict[int, int],
    seen: set,
) -> list[dict]:
    """
    Build one split with per-bucket targets.

    For digits 1–2: enumerate the full bucket space, filter out already-seen
    keys, shuffle the remainder, take up to target, then mark chosen rows seen.
    For digits 3–4: random sample (hit rates are high enough to be efficient).
    """
    rows: list[dict] = []
    for digits, target in targets.items():
        lo, hi = _BUCKETS[digits]

        if digits <= 2:
            # Enumerate entire space, exclude already-claimed triples.
            if operation == "+":
                pool = _all_add(lo, hi)
            else:
                pool = _all_sub(lo, hi)

            available = [
                r for r in pool
                if (int(r["operand_1"]), operation, int(r["operand_2"])) not in seen
            ]
            rng.shuffle(available)
            chosen = available[:target]
            for r in chosen:
                seen.add((int(r["operand_1"]), operation, int(r["operand_2"])))
            rows.extend(chosen)
        else:
            rows.extend(_sample(rng, operation, lo, hi, target, seen))

    return rows


# ---------------------------------------------------------------------------
# CSV writer and distribution reporter
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _print_dist(label: str, rows: list[dict]) -> None:
    n = len(rows)
    by_op: dict[str, dict[int, int]] = {"+": {}, "-": {}}
    totals: dict[int, int] = {}
    for row in rows:
        d = _digit_count(row["answer"])
        op = row["operation"]
        by_op[op][d] = by_op[op].get(d, 0) + 1
        totals[d] = totals.get(d, 0) + 1

    print(f"\n  {label} ({n} samples):")
    for d in (1, 2, 3, 4):
        c = totals.get(d, 0)
        add_c = by_op["+"].get(d, 0)
        sub_c = by_op["-"].get(d, 0)
        bar = "#" * int(30 * c / n) if n else ""
        print(
            f"    {d}-digit: {c:>6} ({100 * c / n:5.1f}%)"
            f"  add={add_c:>5} sub={sub_c:>5}  {bar}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(seed: int = 42) -> None:
    """Generate train/val/test CSVs and write them to data/.

    Training uses stratified sampling by answer digit length so that
    short-answer cases (1- and 2-digit) are well represented; without
    stratification, uniform sampling produces almost no 1-digit addition
    examples (only ~55 unique pairs exist in the [0,999] domain).
    Val and test are built from the remaining key space without stratification
    to give an unbiased evaluation distribution.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Seen sets grow across all three splits — no (a, op, b) triple is reused.
    seen_add: set = set()
    seen_sub: set = set()

    add_train = _build_split(rng, "+", _ADD_TRAIN, seen_add)
    sub_train = _build_split(rng, "-", _SUB_TRAIN, seen_sub)

    add_val  = _build_split(rng, "+", _ADD_VAL,  seen_add)
    add_test = _build_split(rng, "+", _ADD_TEST, seen_add)
    sub_val  = _build_split(rng, "-", _SUB_VAL,  seen_sub)
    sub_test = _build_split(rng, "-", _SUB_TEST, seen_sub)

    train = add_train + sub_train
    val   = add_val   + sub_val
    test  = add_test  + sub_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    _write_csv(DATA_DIR / "train.csv", train)
    _write_csv(DATA_DIR / "val.csv",   val)
    _write_csv(DATA_DIR / "test.csv",  test)

    print(f"Dataset written to {DATA_DIR}")
    print(f"  train : {len(train):>6}  ({len(add_train)} add, {len(sub_train)} sub)")
    print(f"  val   : {len(val):>6}  ({len(add_val)} add, {len(sub_val)} sub)")
    print(f"  test  : {len(test):>6}  ({len(add_test)} add, {len(sub_test)} sub)")

    print("\nAnswer-length distribution:")
    _print_dist("train", train)
    _print_dist("val",   val)
    _print_dist("test",  test)


if __name__ == "__main__":
    generate()
