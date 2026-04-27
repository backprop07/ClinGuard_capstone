from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from clinguard_capstone.metrics import load_metrics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
PANEL = "#f8fafc"
BLUE = "#2563eb"
TEAL = "#0f766e"
GREEN = "#059669"
AMBER = "#d97706"
RED = "#dc2626"
VIOLET = "#6d28d9"

MODEL_COLORS = {
    "GPT-5.4": BLUE,
    "DeepSeek-V3.2": AMBER,
    "Qwen3.5-9B": GREEN,
}


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=320, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def pct(value: float) -> float:
    return 100.0 * value


def derived_metrics(metrics: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model, item in metrics.items():
        l2 = item["level2"]
        tp, tn, fp, fn = l2["tp"], l2["tn"], l2["fp"], l2["fn"]
        warning_sensitivity = tp / (tp + fn) if tp + fn else 0.0
        warning_specificity = tn / (tn + fp) if tn + fp else 0.0
        false_alert_rate = fp / (tn + fp) if tn + fp else 0.0
        warning_precision = tp / (tp + fp) if tp + fp else 0.0
        out[model] = {
            "attempted": item["counts"]["attempted"],
            "l2_eligible": item["counts"]["l2_eligible"],
            "l3_eligible": item["counts"]["l3_eligible"],
            "level1_accuracy": item["level1_accuracy"],
            "warning_precision": warning_precision,
            "warning_sensitivity": warning_sensitivity,
            "warning_specificity": warning_specificity,
            "false_alert_rate": false_alert_rate,
            "warning_f1": l2["f1"],
            "warning_type_accuracy": l2["warn_type_accuracy_conditional"],
            "draft_completion_rate": l2["draft_route_accuracy_conditional"],
            "strict_l2_success": l2["joint_accuracy"],
            "reaffirmation_pass": item["level3"]["intervention_pass_rate"],
            "recovery_pass": item["level3"]["control_recovery_pass_rate"],
            "level3_pass": item["level3"]["pass_rate"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_expected_warn": l2["counts"]["n_expected_warn"],
            "n_expected_draft": l2["counts"]["n_expected_draft"],
            "n_reaffirm": item["level3"]["counts"]["n_intervention_reaffirm"],
            "n_recovery": item["level3"]["counts"]["n_control_recovery"],
        }
    return out


def rounded_box(ax, x, y, w, h, text, *, face=PANEL, edge=INK, color=INK, size=9.5, weight="normal"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, *, color=INK, rad=0.0, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def clinical_guardian_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.7))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.7)
    ax.axis("off")

    ax.text(
        0.55,
        5.32,
        "Clinical guardian workflow with clinician-agent dialogue",
        ha="left",
        va="center",
        color=INK,
        fontsize=12.5,
        fontweight="bold",
    )
    ax.text(
        0.55,
        5.05,
        "A guardian agent acts as a second-checking layer: it may challenge a plan, receive clinician justification, and revise or maintain its recommendation.",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=8.6,
    )

    panels = [
        (0.45, 0.72, 2.55, 3.92, "#eef2ff", BLUE, "1", "Case and plan"),
        (3.35, 0.72, 3.15, 3.92, "#ecfeff", TEAL, "2", "Review dialogue"),
        (6.9, 0.72, 1.85, 3.92, "#fff7ed", AMBER, "3", "Output"),
        (9.05, 0.72, 1.9, 3.92, "#f0fdf4", GREEN, "4", "Decision"),
    ]
    for x, y, w, h, face, edge, number, title in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                facecolor=face,
                edgecolor="none",
            )
        )
        ax.scatter([x + 0.28], [y + h - 0.32], s=230, color=edge, edgecolor="white", linewidth=1.3, zorder=3)
        ax.text(x + 0.28, y + h - 0.32, number, ha="center", va="center", color="white", fontsize=9.2, fontweight="bold", zorder=4)
        ax.text(x + 0.55, y + h - 0.37, title, ha="left", va="center", color=INK, fontsize=9.4, fontweight="bold")

    rounded_box(ax, 0.82, 3.24, 1.78, 0.7, "Patient case", face="white", edge=BLUE, weight="bold", size=9.2)
    rounded_box(ax, 0.82, 2.08, 1.78, 0.78, "Clinician\nproposes plan", face="white", edge=BLUE, weight="bold", size=8.6)
    arrow(ax, 1.71, 3.24, 1.71, 2.86, color=BLUE, lw=1.4)
    ax.text(
        0.82,
        1.38,
        "The benchmark begins\nfrom a concrete plan,\nnot an abstract query.",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=7.8,
    )

    rounded_box(ax, 3.88, 3.15, 2.05, 0.76, "Guardian agent\nreviews plan", face="white", edge=TEAL, weight="bold", size=9.1)
    rounded_box(ax, 3.88, 1.72, 2.05, 0.76, "Clinician\nresponds", face="white", edge=BLUE, weight="bold", size=9.6)
    arrow(ax, 4.9, 3.15, 4.9, 2.48, color=TEAL, lw=1.4)
    arrow(ax, 5.94, 2.1, 5.94, 3.22, color=BLUE, rad=-0.42, lw=1.35)
    ax.text(4.02, 2.78, "challenge", ha="left", va="center", color=TEAL, fontsize=7.4, fontweight="bold")
    ax.text(6.08, 2.62, "justify\nor revise", ha="left", va="center", color=BLUE, fontsize=7.3, fontweight="bold")
    rounded_box(ax, 4.0, 1.02, 1.8, 0.42, "dialogue before output", face="white", edge=GRID, color=MUTED, size=7.2)

    rounded_box(ax, 7.18, 3.13, 1.24, 0.68, "Draft", face="white", edge=GREEN, weight="bold", size=9.6)
    rounded_box(ax, 7.18, 1.88, 1.24, 0.68, "Warning", face="white", edge=RED, weight="bold", size=9.4)
    ax.text(
        7.18,
        1.18,
        "Calibrated\nrouting",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=7.5,
    )

    rounded_box(
        ax,
        9.35,
        2.54,
        1.32,
        1.1,
        "Final clinical\ndecision",
        face="white",
        edge=GREEN,
        weight="bold",
        size=8.4,
    )
    rounded_box(
        ax,
        9.35,
        1.36,
        1.32,
        0.62,
        "Human\naccountable",
        face="white",
        edge=GREEN,
        weight="bold",
        size=7.9,
    )

    arrow(ax, 2.6, 2.47, 3.88, 3.53, color=TEAL, rad=0.05, lw=1.45)
    arrow(ax, 5.93, 3.5, 7.18, 3.47, color=GREEN, rad=0.02, lw=1.45)
    arrow(ax, 5.93, 3.5, 7.18, 2.22, color=RED, rad=-0.1, lw=1.45)
    arrow(ax, 8.42, 3.47, 9.35, 3.1, color=GREEN, rad=-0.08, lw=1.35)
    arrow(ax, 8.42, 2.22, 9.35, 3.0, color=RED, rad=0.1, lw=1.35)
    arrow(ax, 10.01, 2.54, 10.01, 1.98, color=GREEN, lw=1.25)

    save(fig, "clinical_guardian_workflow")


def benchmark_curation_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(
        0.45,
        6.02,
        "Automated MedQA-derived benchmark generation and validation",
        color=INK,
        fontsize=13.8,
        fontweight="bold",
    )
    ax.text(
        0.45,
        5.72,
        "The Level 1 medical answer is preserved from MedQA; encounter context and interaction text are generated, then checked by deterministic validators.",
        color=MUTED,
        fontsize=9.4,
    )

    lane_specs = [
        (4.36, "Preserved\nsource", BLUE, "#eff6ff"),
        (3.04, "Generated\ncontext", TEAL, "#ecfeff"),
        (1.72, "Deterministic\nchecks", VIOLET, "#f5f3ff"),
    ]
    for y, label, color, face in lane_specs:
        ax.add_patch(
            FancyBboxPatch(
                (0.42, y - 0.16),
                11.78,
                0.86,
                boxstyle="round,pad=0.02,rounding_size=0.11",
                linewidth=0.7,
                edgecolor="#dbeafe" if color == BLUE else GRID,
                facecolor=face,
                alpha=0.7,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.68, y + 0.02),
                1.02,
                0.40,
                boxstyle="round,pad=0.025,rounding_size=0.08",
                linewidth=1.0,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(
            1.19,
            y + 0.22,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=7.2,
            fontweight="bold",
            linespacing=0.95,
        )

    def box(x, y, w, h, text, *, edge, face="white", size=8.7, weight="normal"):
        rounded_box(ax, x, y, w, h, text, face=face, edge=edge, color=INK, size=size, weight=weight)

    # Column anchors and main boxes.
    x0, x1, x2, x3, x4 = 2.06, 4.36, 6.66, 8.96, 10.02
    w = 1.66

    box(x0, 4.38, w, 0.62, "MedQA text item", edge=BLUE, face="white", weight="bold")
    box(x1, 4.38, w, 0.62, "Question stem\nand options", edge=BLUE, face="white")
    box(x2, 4.38, w, 0.62, "Correct answer\nA-D", edge=BLUE, face="white")

    box(x0, 3.05, w, 0.62, "LLM constrained\ngeneration", edge=TEAL, face="white", size=8.0, weight="bold")
    box(x1, 3.05, w, 0.62, "Dialogue and\nmedical record", edge=TEAL, face="white")
    box(x2, 3.05, w, 0.62, "Clinician proposal\nand pushback text", edge=TEAL, face="white")

    box(x0, 1.73, w, 0.62, "Case scaffold\nfilled by code", edge=VIOLET, face="white", size=8.0, weight="bold")
    box(x1, 1.73, w, 0.62, "Level 2 route,\nwarning tag, tools", edge=VIOLET, face="white")
    box(x2, 1.73, w, 0.62, "Schema, ontology,\nand leakage checks", edge=VIOLET, face="white")

    final_box = FancyBboxPatch(
        (x4, 2.55),
        2.00,
        1.34,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.35,
        edgecolor=GREEN,
        facecolor="#f0fdf4",
    )
    ax.add_patch(final_box)
    ax.text(
        x4 + 1.00,
        3.22,
        "Accepted benchmark\ncandidate",
        ha="center",
        va="center",
        color=INK,
        fontsize=8.8,
        fontweight="bold",
    )
    ax.text(
        x4 + 1.00,
        2.78,
        "not clinically verified",
        ha="center",
        va="center",
        color=GREEN,
        fontsize=8.1,
        fontstyle="italic",
    )

    # Within-lane arrows.
    arrow(ax, x0 + w, 4.69, x1 - 0.14, 4.69, color=BLUE, lw=1.45)
    arrow(ax, x1 + w, 4.69, x2 - 0.14, 4.69, color=BLUE, lw=1.45)
    arrow(ax, x0 + w, 3.36, x1 - 0.14, 3.36, color=TEAL, lw=1.45)
    arrow(ax, x1 + w, 3.36, x2 - 0.14, 3.36, color=TEAL, lw=1.45)
    arrow(ax, x0 + w, 2.04, x1 - 0.14, 2.04, color=VIOLET, lw=1.45)
    arrow(ax, x1 + w, 2.04, x2 - 0.14, 2.04, color=VIOLET, lw=1.45)

    # Cross-lane dependencies.
    arrow(ax, x0 + w * 0.50, 4.38, x0 + w * 0.50, 3.67, color=MUTED, rad=0.0, lw=1.05)
    arrow(ax, x2 + w, 3.36, x4 - 0.10, 3.35, color=GREEN, lw=1.35)
    arrow(ax, x2 + w, 2.04, x4 - 0.10, 2.96, color=GREEN, rad=0.11, lw=1.35)

    # Evaluation output callouts.
    outputs = [
        (0.92, 0.67, "Level 1:\nanswer correctness", BLUE),
        (3.30, 0.67, "Level 2:\nroute, tag, tools", VIOLET),
        (5.68, 0.67, "Level 3:\nreaffirm or recover", AMBER),
        (8.06, 0.67, "Case audit:\nstructured trace", GREEN),
    ]
    for x, y, text, color in outputs:
        box(x, y, 1.72, 0.55, text, edge=color, face="white", size=8.2)
    for x in [1.78, 4.16, 6.54, 8.92]:
        arrow(ax, x, 1.73, x, 1.25, color=MUTED, rad=0.0, lw=0.95)

    ax.add_patch(
        Polygon(
            [[0.72, 0.32], [11.72, 0.32], [11.46, 0.08], [0.98, 0.08]],
            closed=True,
            facecolor="#f9fafb",
            edgecolor=GRID,
        )
    )
    ax.text(
        1.02,
        0.18,
        "Scope: 300 text-only MedQA-derived generated candidates; professional clinical verification is future work.",
        color=MUTED,
        fontsize=8.8,
        va="center",
    )
    save(fig, "benchmark_curation_pipeline")


def evaluation_protocol() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    ax.text(0.55, 4.55, "Three-level evaluation protocol", color=INK, fontsize=13, fontweight="bold")
    rounded_box(ax, 0.55, 3.5, 1.65, 0.7, "Level 1\nknowledge", face="#eff6ff", edge=BLUE, weight="bold")
    rounded_box(ax, 3.1, 3.5, 1.65, 0.7, "Level 2\nwarning gate", face="#ecfeff", edge=TEAL, weight="bold")
    rounded_box(ax, 5.65, 3.5, 1.65, 0.7, "Level 3\npushback", face="#fff7ed", edge=AMBER, weight="bold")
    rounded_box(ax, 8.05, 3.5, 1.45, 0.7, "Metrics\nreported", face="#f0fdf4", edge=GREEN, weight="bold")
    arrow(ax, 2.2, 3.85, 3.1, 3.85, color=MUTED)
    arrow(ax, 4.75, 3.85, 5.65, 3.85, color=MUTED)
    arrow(ax, 7.3, 3.85, 8.05, 3.85, color=MUTED)

    rounded_box(ax, 0.55, 2.15, 1.65, 0.58, "Incorrect:\nstop case", face="white", edge=RED, size=9)
    rounded_box(ax, 3.1, 2.15, 1.65, 0.58, "Draft\nor warn", face="white", edge=TEAL, size=9)
    rounded_box(ax, 5.25, 2.15, 1.55, 0.58, "Defend valid\nwarning", face="white", edge=AMBER, size=9)
    rounded_box(ax, 7.05, 2.15, 1.55, 0.58, "Withdraw false\nwarning", face="white", edge=AMBER, size=9)
    rounded_box(ax, 8.85, 2.15, 0.95, 0.58, "Audit\ntrace", face="white", edge=GREEN, size=9)

    arrow(ax, 1.38, 3.5, 1.38, 2.73, color=RED, rad=0.0)
    arrow(ax, 3.93, 3.5, 3.93, 2.73, color=TEAL, rad=0.0)
    arrow(ax, 6.48, 3.5, 6.02, 2.73, color=AMBER, rad=0.05)
    arrow(ax, 6.48, 3.5, 7.82, 2.73, color=AMBER, rad=-0.05)
    arrow(ax, 8.78, 3.5, 9.32, 2.73, color=GREEN, rad=0.05)

    metrics = [
        ("Accuracy", BLUE),
        ("Sensitivity", TEAL),
        ("False-alert rate", TEAL),
        ("Warning type", VIOLET),
        ("Reaffirmation", AMBER),
        ("Recovery", AMBER),
    ]
    for i, (label, color) in enumerate(metrics):
        x = 0.78 + i * 1.45
        rounded_box(ax, x, 0.92, 1.16, 0.38, label, face="white", edge=color, size=8.1)
    ax.text(0.55, 1.56, "Interpretation is component-level: no single headline score is treated as clinically authoritative.", color=INK, fontsize=9.6)
    save(fig, "evaluation_protocol")


def performance_landscape(d: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    ax.axhline(90, color=GRID, lw=1)
    ax.axvline(50, color=GRID, lw=1)
    for model, item in d.items():
        x = pct(item["false_alert_rate"])
        y = pct(item["warning_sensitivity"])
        size = 90 + item["l2_eligible"] * 1.6
        ax.scatter(x, y, s=size, color=MODEL_COLORS[model], alpha=0.88, edgecolor="white", linewidth=1.5)
        ax.text(x + 1.5, y - 1.8, model, color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xlim(-3, 93)
    ax.set_ylim(72, 104)
    ax.set_xlabel("False-alert rate on control cases (%)")
    ax.set_ylabel("Warning sensitivity on intervention cases (%)")
    ax.set_title("Warning-gate trade-off after the knowledge gate", pad=10)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.8)
    ax.text(3, 73.5, "Better region: high sensitivity with low false-alert burden", color=MUTED, fontsize=9)
    save(fig, "warning_gate_landscape")


def strict_success_heatmap(d: dict[str, dict]) -> None:
    models = list(d)
    labels = [
        "Knowledge\naccuracy",
        "Warning\ntype",
        "Draft\ncompletion",
        "Strict L2\nsuccess",
        "Defend valid\nwarning",
        "Withdraw false\nwarning",
    ]
    values = np.array(
        [
            [
                pct(d[m]["level1_accuracy"]),
                pct(d[m]["warning_type_accuracy"]),
                pct(d[m]["draft_completion_rate"]),
                pct(d[m]["strict_l2_success"]),
                pct(d[m]["reaffirmation_pass"]),
                pct(d[m]["recovery_pass"]),
            ]
            for m in models
        ]
    )
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(models)), models)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 62 else INK, fontweight="bold", fontsize=9)
    ax.set_title("Component-level success rates")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("%")
    save(fig, "component_success_heatmap")


def level3_tradeoff(d: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot([0, 100], [0, 100], color=GRID, lw=1.2, linestyle="--")
    label_offsets = {
        "GPT-5.4": (1.8, -2.6, "left"),
        "DeepSeek-V3.2": (1.8, 1.1, "left"),
        "Qwen3.5-9B": (1.8, -1.7, "left"),
    }
    for model, item in d.items():
        x = pct(item["recovery_pass"])
        y = pct(item["reaffirmation_pass"])
        ax.scatter(x, y, s=260, color=MODEL_COLORS[model], alpha=0.9, edgecolor="white", linewidth=1.4)
        dx, dy, ha = label_offsets[model]
        ax.text(x + dx, y + dy, model, color=INK, fontsize=9.5, fontweight="bold", ha=ha)
    ax.set_xlim(-3, 105)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("Withdraw false warning after valid correction (%)")
    ax.set_ylabel("Defend valid warning against invalid pushback (%)")
    ax.set_title("Level 3 separates firmness from corrigibility")
    ax.grid(color=GRID, linewidth=0.7, alpha=0.8)
    ax.text(6, 7, "Balanced behaviour would move toward the upper-right corner.", color=MUTED, fontsize=9)
    save(fig, "level3_tradeoff")


def level2_decision_matrices(d: dict[str, dict]) -> None:
    models = list(d)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True, sharey=True)
    for ax, model in zip(axes, models, strict=True):
        item = d[model]
        matrix = np.array([[item["tn"], item["fp"]], [item["fn"], item["tp"]]])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(item["tp"]), int(item["fp"]), 1))
        ax.set_title(model, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks([0, 1], ["No warning", "Warning"])
        ax.set_yticks([0, 1], ["Control", "Intervention"])
        ax.tick_params(axis="x", rotation=25)
        for i in range(2):
            for j in range(2):
                value = int(matrix[i, j])
                label = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i, f"{label}\n{value}", ha="center", va="center", color="white" if value > matrix.max() * 0.48 else INK, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Ground truth")
    for ax in axes:
        ax.set_xlabel("Predicted warning gate")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Cases")
    save(fig, "level2_decision_matrices")


def level3_recovery_split(d: dict[str, dict]) -> None:
    models = list(d)
    fig, ax = plt.subplots(figsize=(7.7, 4.2))
    y = np.arange(len(models))
    for i, model in enumerate(models):
        reaffirm = pct(d[model]["reaffirmation_pass"])
        recovery = pct(d[model]["recovery_pass"])
        ax.plot([recovery, reaffirm], [i, i], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
        ax.scatter([recovery], [i], s=160, color=TEAL, edgecolor="white", linewidth=1.2, zorder=3, label="Withdraw false warning" if i == 0 else None)
        ax.scatter([reaffirm], [i], s=160, color=AMBER, edgecolor="white", linewidth=1.2, zorder=3, label="Defend valid warning" if i == 0 else None)
        ax.text(recovery - 1.5, i + 0.16, f"{recovery:.1f}", ha="right", va="center", fontsize=8.8, color=TEAL, fontweight="bold")
        ax.text(reaffirm + 1.5, i + 0.16, f"{reaffirm:.1f}", ha="left", va="center", fontsize=8.8, color=AMBER, fontweight="bold")
    ax.set_yticks(y, models)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Pass rate (%)")
    ax.set_title("Level 3 split: firmness versus recovery", pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "level3_recovery_split")


def case_funnel(d: dict[str, dict]) -> None:
    models = list(d)
    stages = ["Attempted", "Passed L1", "Triggered L3"]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    x = np.arange(len(stages))
    for model in models:
        y = [d[model]["attempted"], d[model]["l2_eligible"], d[model]["l3_eligible"]]
        ax.plot(x, y, marker="o", lw=2.4, color=MODEL_COLORS[model], label=model)
        for xx, yy in zip(x, y, strict=True):
            ax.text(xx, yy + 8, str(int(yy)), ha="center", va="bottom", color=MODEL_COLORS[model], fontsize=8.5)
    ax.set_xticks(x, stages)
    ax.set_ylim(0, 330)
    ax.set_ylabel("Number of cases")
    ax.set_title("Evaluation funnel by model")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "evaluation_funnel")


def write_summary(d: dict[str, dict]) -> None:
    rows = []
    for model, item in d.items():
        row = {"model": model}
        row.update(item)
        rows.append(row)
    with (RESULTS / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    metrics = load_metrics(RESULTS)
    d = derived_metrics(metrics)
    clinical_guardian_workflow()
    benchmark_curation_pipeline()
    evaluation_protocol()
    performance_landscape(d)
    level2_decision_matrices(d)
    strict_success_heatmap(d)
    level3_tradeoff(d)
    level3_recovery_split(d)
    case_funnel(d)
    write_summary(d)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
