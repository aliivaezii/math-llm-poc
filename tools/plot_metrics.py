"""
Generate the three summary charts for the Math-LLM-PoC audit report.

Run from the project root:
    python tools/plot_metrics.py

Outputs:
    docs/accuracy_phases.png
    docs/accuracy_by_length.png
    docs/loss_curve.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # headless: no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
LOGS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "training_logs.txt"

# Colour palette — consistent across all three charts
C_BLUE   = "#4878CF"
C_ORANGE = "#E8762E"
C_GREEN  = "#5BAD6F"
C_GREY   = "#9E9E9E"

PHASE_BG = "#F5F0FF"   # light lavender for LR-phase shading


# ---------------------------------------------------------------------------
# Chart 1 — Accuracy progression across training phases
# ---------------------------------------------------------------------------

def plot_accuracy_phases() -> None:
    labels   = ["Baseline\n(10 ep, 137K params)", "T2-A\n(30 ep, 1.07M params)", "Final\n(60 ep, stratified)"]
    values   = [4.22, 86.00, 98.44]
    colours  = [C_GREY, C_BLUE, C_GREEN]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colours, width=0.5, edgecolor="white", linewidth=0.8)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 112)
    ax.set_ylabel("Exact-match accuracy (%)", fontsize=11)
    ax.set_title("Exact-match Accuracy — Training Phase Progression", fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = DOCS_DIR / "accuracy_phases.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out}")


# ---------------------------------------------------------------------------
# Chart 2 — Accuracy by answer digit length (final model)
# ---------------------------------------------------------------------------

def plot_accuracy_by_length() -> None:
    labels   = ["1-digit\n(99.06%)", "2-digit\n(95.57%)", "3-digit\n(98.67%)", "4-digit\n(99.50%)"]
    values   = [99.06, 95.57, 98.67, 99.50]
    totals   = [106, 700, 3000, 1200]
    colours  = [C_GREEN, C_ORANGE, C_BLUE, C_GREEN]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colours, width=0.55, edgecolor="white", linewidth=0.8)

    # Annotate with exact counts
    for bar, val, tot in zip(bars, values, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            2,
            f"n={tot}",
            ha="center", va="bottom",
            fontsize=8.5, color="white", fontweight="bold",
        )

    ax.set_ylim(0, 108)
    ax.set_ylabel("Exact-match accuracy (%)", fontsize=11)
    ax.set_title("Final Model — Accuracy by Answer Digit Length", fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Dashed reference line at 95%
    ax.axhline(95, color=C_GREY, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(3.42, 95.4, "95%", fontsize=8, color=C_GREY)

    fig.tight_layout()
    out = DOCS_DIR / "accuracy_by_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out}")


# ---------------------------------------------------------------------------
# Chart 3 — Train vs val loss over 60 epochs (with LR phase shading)
# ---------------------------------------------------------------------------

def _parse_logs(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs, train_losses, val_losses = [], [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("epoch") or line.startswith("-"):
                continue
            # Could be "Early stopping" text line
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                epochs.append(int(parts[0]))
                train_losses.append(float(parts[1]))
                val_losses.append(float(parts[2]))
            except ValueError:
                continue
    return epochs, train_losses, val_losses


def plot_loss_curve() -> None:
    epochs, train_losses, val_losses = _parse_logs(LOGS_PATH)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Shade LR phases
    ax.axvspan(1, 5,  alpha=0.08, color="#FFD700", label="_nolegend_")  # warmup
    ax.axvspan(5, 40, alpha=0.06, color="#4878CF", label="_nolegend_")  # flat
    ax.axvspan(40, max(epochs), alpha=0.08, color="#E8762E", label="_nolegend_")  # decay

    # Phase labels along top
    y_top = max(train_losses) * 0.995
    ax.text(3,    y_top, "Warmup",     ha="center", va="top", fontsize=7.5, color="#888")
    ax.text(22.5, y_top, "Flat LR",    ha="center", va="top", fontsize=7.5, color="#888")
    ax.text(50,   y_top, "Cosine decay", ha="center", va="top", fontsize=7.5, color="#888")

    ax.plot(epochs, train_losses, color=C_BLUE,   linewidth=1.8, label="train_loss")
    ax.plot(epochs, val_losses,   color=C_ORANGE, linewidth=1.8, label="val_loss",  linestyle="--")

    # Mark best checkpoint (epoch 55)
    best_ep  = 55
    best_val = val_losses[best_ep - 1]
    ax.scatter([best_ep], [best_val], color=C_GREEN, zorder=5, s=60)
    ax.annotate(
        f"  best ckpt\n  ep {best_ep} ({best_val:.4f})",
        xy=(best_ep, best_val),
        xytext=(best_ep - 14, best_val + 0.04),
        fontsize=8, color=C_GREEN,
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1),
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Train vs Val Loss — 60-Epoch Run (Warmup → Flat → Cosine Decay)", fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Phase boundary lines
    for x in (5, 40):
        ax.axvline(x, color=C_GREY, linestyle=":", linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    out = DOCS_DIR / "loss_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating charts...")
    plot_accuracy_phases()
    plot_accuracy_by_length()
    plot_loss_curve()
    print("Done.")
