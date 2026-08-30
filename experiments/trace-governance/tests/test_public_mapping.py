from __future__ import annotations

import csv
import os
from pathlib import Path

from circ_trace.clinicare import load_codebook, load_process_criteria, load_risk_taxonomy
from circ_trace.public_mapping import summarize_mapping, write_public_mapping_outputs

ROOT = Path(__file__).resolve().parents[1]
CLINICARE_REPO = Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))


def _inputs():
    return (
        load_process_criteria(CLINICARE_REPO),
        load_risk_taxonomy(ROOT / "configs" / "risk_taxonomy.toml"),
        load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv"),
    )


def test_mapping_summary_has_broad_scenario_coverage() -> None:
    summary = summarize_mapping(*_inputs())
    rows = list(summary["rows"])

    assert summary["criterion_count"] == 104
    assert summary["scenario_count"] == 25
    assert summary["risk_family_count"] == 9
    assert sum(row["primary_criteria"] for row in rows) == 104
    assert min(row["scenarios_covered"] for row in rows) >= 4
    assert sum(sum(counts.values()) for counts in summary["matrix"].values()) == 104


def test_reviewer_packets_are_complete_randomized_and_label_blank(tmp_path: Path) -> None:
    criteria, risks, entries = _inputs()
    write_public_mapping_outputs(
        output_dir=tmp_path,
        criteria=criteria,
        risks=risks,
        entries=entries,
        render_figure=False,
    )
    with (tmp_path / "reviewer_1.csv").open(newline="") as handle:
        first = list(csv.DictReader(handle))
    with (tmp_path / "reviewer_2.csv").open(newline="") as handle:
        second = list(csv.DictReader(handle))

    assert len(first) == len(second) == 104
    assert {row["source_ref"] for row in first} == {row["source_ref"] for row in second}
    assert all(not row["primary_risk"] and not row["secondary_risks"] for row in first + second)
    assert [row["source_ref"] for row in first] != [row["source_ref"] for row in second]
    assert (tmp_path / "risk_definitions.csv").is_file()
    assert (tmp_path / "scenario_key.csv").is_file()
    assert "provisional" in (tmp_path / "manuscript_insert.md").read_text().lower()