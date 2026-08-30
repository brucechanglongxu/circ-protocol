"""Public content analysis of CliniCARE's clinician-authored MUST-NOT criteria."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .clinicare import CodebookEntry, ProcessCriterion, RiskFamily, validate_codebook

_SCENARIO_NUMBER = re.compile(r"^scenario-(\d+)-")


def _scenario_order(criteria: tuple[ProcessCriterion, ...]) -> list[str]:
    number_by_scenario: dict[str, int] = {}
    for criterion in criteria:
        match = _SCENARIO_NUMBER.match(criterion.source.parent.name)
        if match is None:
            raise ValueError(f"cannot recover scenario number from {criterion.source}")
        number_by_scenario[criterion.scenario_id] = int(match.group(1))
    return sorted(number_by_scenario, key=number_by_scenario.get)


def summarize_mapping(
    criteria: tuple[ProcessCriterion, ...],
    risks: tuple[RiskFamily, ...],
    entries: tuple[CodebookEntry, ...],
) -> dict[str, object]:
    validate_codebook(criteria, risks, entries)
    scenarios = _scenario_order(criteria)
    primary_counts = Counter(entry.primary_risk for entry in entries)
    scenario_sets: dict[str, set[str]] = defaultdict(set)
    secondary_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        scenario_sets[entry.primary_risk].add(entry.scenario_id)
        matrix[entry.primary_risk][entry.scenario_id] += 1
        secondary_counts.update(entry.secondary_risks)
        boundary_counts[entry.primary_risk] += int(entry.review_priority == "boundary")

    rows = [
        {
            "risk_id": risk.risk_id,
            "risk_family": risk.name,
            "primary_criteria": primary_counts[risk.risk_id],
            "primary_percent": round(100.0 * primary_counts[risk.risk_id] / len(entries), 1),
            "scenarios_covered": len(scenario_sets[risk.risk_id]),
            "scenario_percent": round(
                100.0 * len(scenario_sets[risk.risk_id]) / len(scenarios), 1
            ),
            "secondary_mentions": secondary_counts[risk.risk_id],
            "boundary_review_items": boundary_counts[risk.risk_id],
        }
        for risk in risks
    ]
    return {
        "criterion_count": len(entries),
        "scenario_count": len(scenarios),
        "risk_family_count": len(risks),
        "scenarios": scenarios,
        "rows": rows,
        "matrix": {
            risk.risk_id: {scenario: matrix[risk.risk_id][scenario] for scenario in scenarios}
            for risk in risks
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_review_sheet(
    path: Path,
    criteria: tuple[ProcessCriterion, ...],
    *,
    seed: int = 20_260_830,
) -> None:
    prohibited = [criterion for criterion in criteria if criterion.is_prohibited]
    random.Random(seed).shuffle(prohibited)
    rows = [
        {
            "item_id": f"ITEM-{index:03d}",
            "source_ref": f"{criterion.scenario_id}/{criterion.criterion_id}",
            "criterion_text": criterion.title,
            "primary_risk": "",
            "secondary_risks": "",
            "confidence": "",
            "notes": "",
        }
        for index, criterion in enumerate(prohibited, start=1)
    ]
    _write_csv(path, rows)


def write_manuscript_insert(path: Path, summary: dict[str, object]) -> None:
    rows = sorted(
        summary["rows"], key=lambda row: int(row["primary_criteria"]), reverse=True
    )
    leading = rows[:3]
    lead_parts = [
        f"{row['risk_family']} ({row['primary_criteria']}/104; {row['primary_percent']}%)"
        for row in leading
    ]
    lead_text = f"{lead_parts[0]}, {lead_parts[1]}, and {lead_parts[2]}"
    minimum_coverage = min(int(row["scenarios_covered"]) for row in rows)
    maximum_coverage = max(int(row["scenarios_covered"]) for row in rows)
    path.write_text(
        "# Draft Manuscript Insert\n\n"
        "## Methods\n\n"
        "We conducted a structured content analysis of the 104 clinician-authored "
        "MUST-NOT process criteria released across the 25 CliniCARE-Bench scenarios. "
        "We developed a non-exhaustive codebook for risks observable in retrospective "
        "clinical-agent audit traces and assigned each criterion one primary risk family "
        "and zero or more secondary families. Two reviewers will independently code all "
        "criteria while blinded to the development labels and family totals. Disagreements "
        "will be adjudicated by a third reviewer. We will report raw agreement, Cohen's "
        "kappa, family-wise confusion, and the adjudicated distribution. Criterion counts "
        "describe rubric coverage rather than observed failure prevalence or severity.\n\n"
        "## Provisional Results\n\n"
        f"The development coding assigned all 104 criteria to nine primary risk families. "
        f"The most represented were {lead_text}. Each family appeared across at least "
        f"{minimum_coverage} scenarios, and the broadest appeared across {maximum_coverage} "
        "of 25 scenarios. These values remain provisional until independent review and "
        "should not be interpreted as frequencies of agent failure.\n\n"
        "## Figure Caption\n\n"
        "**Figure X | Development mapping of clinician-authored prohibited process "
        "shortcuts in CliniCARE-Bench.** (A) Number of criteria assigned to each primary "
        "risk family; parentheses show the number of 25 clinical scenarios represented. "
        "(B) Distribution of primary assignments across scenarios. Counts characterize "
        "the benchmark rubric, not observed agent failures. Coding is provisional pending "
        "independent double review.\n"
    )


def render_mapping_figure(summary: dict[str, object], output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(summary["rows"])
    scenarios = list(summary["scenarios"])
    matrix_by_risk = dict(summary["matrix"])
    risk_ids = [str(row["risk_id"]) for row in rows]
    names = [str(row["risk_family"]) for row in rows]
    counts = [int(row["primary_criteria"]) for row in rows]
    scenario_counts = [int(row["scenarios_covered"]) for row in rows]
    matrix = np.array(
        [[matrix_by_risk[risk_id][scenario] for scenario in scenarios] for risk_id in risk_ids]
    )

    figure = plt.figure(figsize=(15, 7.2), layout="constrained")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.15, 2.85))
    bars = figure.add_subplot(grid[0, 0])
    heatmap = figure.add_subplot(grid[0, 1])

    positions = np.arange(len(risk_ids))
    bars.barh(positions, counts, color="#176B87", height=0.68)
    bars.set_yticks(positions, labels=names, fontsize=9)
    bars.invert_yaxis()
    bars.set_xlabel("Primary MUST-NOT criteria")
    bars.set_title("A  Risk-family representation", loc="left", fontweight="bold")
    bars.spines[["top", "right", "left"]].set_visible(False)
    bars.grid(axis="x", color="#D6DCE1", linewidth=0.7)
    bars.set_axisbelow(True)
    for position, (count, covered) in enumerate(zip(counts, scenario_counts)):
        bars.text(count + 0.35, position, f"{count}  ({covered}/25)", va="center", fontsize=8)
    bars.set_xlim(0, max(counts) + 6)

    image = heatmap.imshow(matrix, aspect="auto", cmap="cividis", vmin=0, vmax=max(2, matrix.max()))
    heatmap.set_yticks(positions, labels=risk_ids, fontsize=9)
    heatmap.set_xticks(np.arange(len(scenarios)), labels=range(1, len(scenarios) + 1), fontsize=7)
    heatmap.set_xlabel("CliniCARE scenario number")
    heatmap.set_title("B  Primary criteria across clinical scenarios", loc="left", fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            if value:
                heatmap.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value >= 2 else "black",
                    fontsize=7,
                    fontweight="bold",
                )
    colorbar = figure.colorbar(
        image,
        ax=heatmap,
        fraction=0.025,
        pad=0.02,
        ticks=range(int(matrix.max()) + 1),
    )
    colorbar.set_label("Criteria")
    figure.suptitle(
        "Development coding of CliniCARE MUST-NOT criteria (n = 104)",
        fontsize=14,
        fontweight="bold",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_public_mapping_outputs(
    *,
    output_dir: Path,
    criteria: tuple[ProcessCriterion, ...],
    risks: tuple[RiskFamily, ...],
    entries: tuple[CodebookEntry, ...],
    render_figure: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_mapping(criteria, risks, entries)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(output_dir / "risk_summary.csv", list(summary["rows"]))

    matrix_rows = [
        {"risk_id": risk_id, **counts}
        for risk_id, counts in dict(summary["matrix"]).items()
    ]
    _write_csv(output_dir / "risk_by_scenario.csv", matrix_rows)
    _write_csv(
        output_dir / "risk_definitions.csv",
        [
            {
                "risk_id": risk.risk_id,
                "risk_family": risk.name,
                "definition": risk.definition,
                "excludes": risk.excludes,
            }
            for risk in risks
        ],
    )
    _write_csv(
        output_dir / "scenario_key.csv",
        [
            {"scenario_number": index, "scenario_id": scenario}
            for index, scenario in enumerate(summary["scenarios"], start=1)
        ],
    )
    write_review_sheet(output_dir / "reviewer_1.csv", criteria, seed=20_260_830)
    write_review_sheet(output_dir / "reviewer_2.csv", criteria, seed=20_260_831)
    write_manuscript_insert(output_dir / "manuscript_insert.md", summary)
    if render_figure:
        render_mapping_figure(summary, output_dir / "clinicare_risk_mapping")
    return summary