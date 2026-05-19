# SFT training loop: loads dataset and tokenizer, trains the model, saves checkpoint.
#
# LR Schedule — three phases (warmup → flat → cosine decay):
#   Phase A (epochs 1–5):   linear warmup from 1e-4 → 1e-3.
#   Phase B (epochs 6–40):  flat at 1e-3 — the model learns fastest here and
#                            needs the full LR sustained.
#   Phase C (epochs 41–60): cosine decay 1e-3 → 1e-5 for fine-grained
#                            convergence once the loss plateau approaches.
#
# Why this differs from the reverted T1-B (CosineAnnealingLR T_max=30):
#   T1-B decayed LR continuously from epoch 1. By epoch 10 — still in the
#   fastest learning phase — LR was already halved, starving the model.
#   Accuracy fell from 78 % to 37 %.  This schedule keeps LR at peak for
#   35 full epochs before touching it.

import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tokenizer import Tokenizer
from src.model import TinyDecoderLM

DATA_DIR      = _ROOT / "data"
ARTIFACTS_DIR = _ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

BATCH_SIZE              = 128
EPOCHS                  = 60
BASE_LR                 = 1e-3
WARMUP_EPOCHS           = 5     # linear ramp: 1e-4 → BASE_LR
FLAT_EPOCHS_END         = 40    # hold BASE_LR through this epoch
MIN_LR                  = 1e-5  # cosine floor
BETAS                   = (0.9, 0.95)
WEIGHT_DECAY            = 0.01
GRAD_CLIP               = 1.0
EARLY_STOPPING_PATIENCE = 8


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(epoch: int) -> float:
    """Three-phase piecewise LR for epoch in [1, EPOCHS]."""
    if epoch <= WARMUP_EPOCHS:
        # Linear warmup: 1e-4 at epoch 1, BASE_LR at epoch WARMUP_EPOCHS.
        t = (epoch - 1) / max(WARMUP_EPOCHS - 1, 1)
        return 1e-4 + t * (BASE_LR - 1e-4)
    if epoch <= FLAT_EPOCHS_END:
        return BASE_LR
    # Cosine decay from BASE_LR to MIN_LR over Phase C.
    progress = (epoch - FLAT_EPOCHS_END) / (EPOCHS - FLAT_EPOCHS_END)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + cosine * (BASE_LR - MIN_LR)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArithmeticDataset(Dataset):
    """Loads an equation CSV and returns (input_ids, target_ids) pairs for teacher-forcing."""

    def __init__(self, csv_path: Path, tokenizer: Tokenizer) -> None:
        self._samples: list[tuple[list[int], list[int]]] = []
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                seq = tokenizer.encode(
                    row["equation"],
                    add_bos=True,
                    add_eos=True,
                )
                self._samples.append((seq[:-1], seq[1:]))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self._samples[idx]


def _make_collate(pad_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]):
        inputs, targets = zip(*batch)
        max_len = max(len(s) for s in inputs)

        def pad(seqs: tuple) -> torch.Tensor:
            return torch.tensor(
                [list(s) + [pad_id] * (max_len - len(s)) for s in seqs],
                dtype=torch.long,
            )

        return pad(inputs), pad(targets)

    return collate


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def _val_pass(
    model: TinyDecoderLM,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    pad_id: int,
) -> tuple[float, float]:
    """Run one validation epoch. Returns (mean_loss, token_accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total_tok  = 0

    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            B, T, V = logits.shape
            total_loss += criterion(
                logits.reshape(B * T, V), targets.reshape(B * T)
            ).item()

            preds = logits.argmax(dim=-1)
            mask  = targets != pad_id
            correct   += (preds[mask] == targets[mask]).sum().item()
            total_tok += mask.sum().item()

    return total_loss / len(loader), correct / total_tok if total_tok else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
    """Train from scratch, save the best-val-loss checkpoint to artifacts/model.pt."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer()
    model     = TinyDecoderLM(vocab_size=tokenizer.vocab_size)

    collate     = _make_collate(tokenizer.pad_id)
    train_loader = DataLoader(
        ArithmeticDataset(DATA_DIR / "train.csv", tokenizer),
        batch_size=batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        ArithmeticDataset(DATA_DIR / "val.csv", tokenizer),
        batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=get_lr(1),
        betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    log_path = ARTIFACTS_DIR / "training_logs.txt"
    log_fh   = log_path.open("w")

    def log(line: str) -> None:
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    header = f"{'epoch':>5}  {'train_loss':>10}  {'val_loss':>8}  {'val_token_acc':>13}  {'lr':>10}"
    log(header)
    log("-" * len(header))

    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        current_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = _val_pass(model, val_loader, criterion, tokenizer.pad_id)

        log(
            f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}"
            f"  {val_acc:>13.4f}  {current_lr:>10.2e}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ARTIFACTS_DIR / "model_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Always save the best-val-loss checkpoint, not the last epoch.
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "model_best.pt", map_location="cpu", weights_only=True)
    )
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")

    log_fh.close()
    print(f"\nSaved best model (epoch {best_epoch}) → {ARTIFACTS_DIR / 'model.pt'}")
    print(f"Logs → {log_path}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) == 2 and _sys.argv[1] == "--lr-check":
        # Quick sanity-check for the schedule without training.
        print("LR schedule dry-check:")
        for ep in (1, 5, 6, 10, 40, 41, 50, 60):
            print(f"  Epoch {ep:>2}: lr={get_lr(ep):.6e}")
    else:
        train()
