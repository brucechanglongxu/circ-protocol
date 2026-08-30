from __future__ import annotations

import os
import tomllib
from collections import Counter
from pathlib import Path

from circ_trace.clinicare import (
    load_codebook,
    load_process_criteria,
    load_risk_taxonomy,
    source_digest,
    validate_codebook,
)

ROOT = Path(__file__).resolve().parents[1]
CLINICARE_REPO = Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))


def test_upstream_source_is_exactly_pinned() -> None:
    provenance = tomllib.loads((ROOT / "configs" / "clinicare_source.toml").read_text())
    assert source_digest(CLINICARE_REPO) == provenance["scenario_rubric_digest"]


def test_codebook_exhaustively_maps_released_must_not_criteria() -> None:
    criteria = load_process_criteria(CLINICARE_REPO)
    risks = load_risk_taxonomy(ROOT / "configs" / "risk_taxonomy.toml")
    entries = load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv")

    validate_codebook(criteria, risks, entries)

    assert len(criteria) == 252
    assert sum(item.is_prohibited for item in criteria) == 104
    assert len(entries) == 104
    assert Counter(entry.primary_risk for entry in entries) == {
        "R-SCOPE": 4,
        "R-TIME": 12,
        "R-PROV": 25,
        "R-COVER": 8,
        "R-CLASS": 13,
        "R-RULE": 8,
        "R-INFER": 11,
        "R-UNCERT": 15,
        "R-CONTEXT": 8,
    }


def test_boundary_cases_are_explicitly_marked_for_second_review() -> None:
    entries = load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv")
    boundary = {entry.key for entry in entries if entry.review_priority == "boundary"}
    assert boundary == {
        ("advanced-imaging", "N5"),
        ("pe-workup", "N2"),
        ("discharge-audit", "N1"),
        ("discharge-audit", "N2"),
        ("sepsis-bundle", "N4"),
        ("empiric-abx", "N1"),
        ("postop-complications", "N2"),
        ("frequent-admitter", "N2"),
        ("ams-workup", "N2"),
    }