from __future__ import annotations

import json
import os
from pathlib import Path

from circ_trace.clinicare_adapter import CliniCAREPolicy, load_required_tables, load_trial_artifact
from circ_trace.schema import ClinicalScope

ROOT = Path(__file__).resolve().parents[1]
CLINICARE_REPO = Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))
FIXTURE = ROOT / "fixtures" / "synthetic_med_reconciliation"


def _load_fixture():
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    return load_trial_artifact(
        artifact_id=metadata["artifact_id"],
        scenario_id=metadata["scenario_id"],
        scope=ClinicalScope(**metadata["scope"]),
        verdict=metadata["verdict"],
        evidence_path=FIXTURE / "evidence.jsonl",
        report_path=FIXTURE / "report.md",
        clinicare_repo=CLINICARE_REPO,
    )


def test_public_source_join_covers_all_scenarios() -> None:
    policy = CliniCAREPolicy(CLINICARE_REPO)
    required = load_required_tables(CLINICARE_REPO, policy)
    assert len(required) == 25
    assert required["med-reconciliation"] == {
        "discharge",
        "emar",
        "medrecon",
        "pharmacy",
        "prescriptions",
    }


def test_adapter_preserves_replay_metadata_and_drops_clinical_text() -> None:
    artifact = _load_fixture()
    retrieved = set().union(*(event.tables for event in artifact.events))
    trusted_ids = set().union(*(set(event.record_ids) for event in artifact.events))

    assert artifact.required_tables <= retrieved
    assert set(artifact.cited_record_ids) <= trusted_ids
    assert all(not hasattr(event, "text") for event in artifact.events)
    assert all(event.kind == "mimic" for event in artifact.events)


def test_adapter_recovers_run_sql_table_from_query() -> None:
    artifact = _load_fixture()
    run_sql = next(event for event in artifact.events if event.tool == "run_sql")
    assert run_sql.tables == {"pharmacy"}