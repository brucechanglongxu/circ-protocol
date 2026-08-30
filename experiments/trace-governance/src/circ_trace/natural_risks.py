"""Descriptive summaries of mapped CliniCARE MUST-NOT rubric triggers."""

from __future__ import annotations

from collections import defaultdict

from .clinicare import CodebookEntry
from .schema import NaturalRiskObservation


def summarize_natural_risks(
    observations: tuple[NaturalRiskObservation, ...],
    codebook: tuple[CodebookEntry, ...],
    *,
    min_cell_size: int = 5,
) -> dict[str, object]:
    """Summarize primary-family triggers with complete within-family grading."""
    if min_cell_size < 1:
        raise ValueError("min_cell_size must be positive")

    expected: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for entry in codebook:
        expected[(entry.scenario_id, entry.primary_risk)].add(entry.key)

    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"denominator": 0, "triggered": 0, "incomplete": 0}
    )
    systems = sorted({observation.system_name for observation in observations})
    for observation in observations:
        families = {
            family
            for scenario_id, family in expected
            if scenario_id == observation.scenario_id
        }
        for family in families:
            needed = expected[(observation.scenario_id, family)]
            keys = (("ALL", family), (observation.system_name, family))
            if not needed <= observation.graded_criteria:
                for key in keys:
                    cells[key]["incomplete"] += 1
                continue
            is_triggered = any(key in observation.triggered_criteria for key in needed)
            for key in keys:
                cells[key]["denominator"] += 1
                cells[key]["triggered"] += int(is_triggered)

    def render(system_name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for _, family in sorted(key for key in cells if key[0] == system_name):
            cell = cells[(system_name, family)]
            denominator = cell["denominator"]
            suppressed = denominator < min_cell_size
            rows.append(
                {
                    "risk_family": family,
                    "denominator": denominator if not suppressed else None,
                    "triggered": cell["triggered"] if not suppressed else None,
                    "rubric_trigger_rate": (
                        cell["triggered"] / denominator
                        if denominator and not suppressed
                        else None
                    ),
                    "incomplete_grading": cell["incomplete"],
                    "suppressed": suppressed,
                }
            )
        return rows

    return {
        "interpretation": "descriptive rubric-trigger rates; not clinical risk prevalence",
        "min_cell_size": min_cell_size,
        "observation_count": len(observations),
        "pooled": render("ALL"),
        "by_system": {system: render(system) for system in systems},
    }