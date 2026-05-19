# Evaluates the trained model on the test split: exact-match accuracy and hallucination rate.

import csv
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tokenizer import Tokenizer
from src.model import TinyDecoderLM

DATA_DIR = _ROOT / "data"
ARTIFACTS_DIR = _ROOT / "artifacts"
MAX_OPERAND = 999
MAX_ANSWER = MAX_OPERAND * 2   # 999 + 999 = 1998


def greedy_generate(
    model: TinyDecoderLM,
    tokenizer: Tokenizer,
    prefix_ids: list[int],
    max_new_tokens: int = 8,
) -> list[int]:
    """Autoregressively extend prefix_ids until <EOS> or max_new_tokens steps."""
    ids = list(prefix_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long)   # (1, T)
            logits = model(x)                            # (1, T, V)
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            if next_id == tokenizer.eos_id:
                break
    return ids


def _extract_answer(full_ids: list[int], prefix_len: int, tokenizer: Tokenizer) -> str:
    """Slice generated tokens after the prefix, strip EOS, decode to string."""
    gen = full_ids[prefix_len:]
    if gen and gen[-1] == tokenizer.eos_id:
        gen = gen[:-1]
    return tokenizer.decode(gen)


def evaluate(max_new_tokens: int = 8) -> None:
    """Run greedy decoding over the test split and print an accuracy report.

    Reports exact-match per operation, hallucination rate (non-numeric output),
    infinite-generation rate (EOS never produced), and out-of-range predictions.
    """
    tokenizer = Tokenizer()
    model = TinyDecoderLM(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "model.pt", weights_only=True)
    )
    model.eval()

    rows = list(csv.DictReader((DATA_DIR / "test.csv").open()))

    total = correct = hallucinations = infinite_gen = out_of_range = 0
    by_op: dict[str, dict[str, int]] = {
        "+": {"correct": 0, "total": 0},
        "-": {"correct": 0, "total": 0},
    }

    for row in rows:
        equation = row["equation"]      # e.g. "951+11=962"
        expected = row["answer"]        # e.g. "962"
        op = row["operation"]

        # Strip the answer from the equation to form the inference prefix.
        prefix = equation[:equation.index("=") + 1]   # "951+11="
        prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
        full_ids = greedy_generate(model, tokenizer, prefix_ids, max_new_tokens)

        eos_produced = full_ids[-1] == tokenizer.eos_id
        predicted = _extract_answer(full_ids, len(prefix_ids), tokenizer)

        # --- classify errors ---
        is_hallucination = not predicted or not predicted.isdigit()
        is_infinite = not eos_produced
        is_out_of_range = (
            predicted.isdigit() and int(predicted) > MAX_ANSWER
        )

        is_correct = predicted == expected

        total += 1
        correct += int(is_correct)
        hallucinations += int(is_hallucination)
        infinite_gen += int(is_infinite)
        out_of_range += int(is_out_of_range)

        by_op[op]["total"] += 1
        by_op[op]["correct"] += int(is_correct)

    def fmt(n: int, d: int) -> str:
        return f"{n / d:.4f}  ({n:>5} / {d})" if d else "n/a"

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║              Evaluation Report                   ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  test samples          : {total:<25}║")
    print(f"║  exact_match_accuracy  : {fmt(correct, total):<25}║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  By operation                                    ║")
    for op_sym in ("+", "-"):
        d = by_op[op_sym]
        print(f"║    [{op_sym}] exact_match   : {fmt(d['correct'], d['total']):<25}║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Error analysis                                  ║")
    print(f"║    hallucinations      : {fmt(hallucinations, total):<25}║")
    print(f"║    infinite_generation : {fmt(infinite_gen, total):<25}║")
    print(f"║    out_of_range        : {fmt(out_of_range, total):<25}║")
    print("╚══════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    evaluate()
