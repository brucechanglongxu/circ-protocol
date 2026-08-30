from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from circ_trace.clinicare_adapter import load_trial_artifact
from circ_trace.controls import evaluate_variant
from circ_trace.endpoints import summarize_by_arm
from circ_trace.faults import clean_variant, inject_fault
from circ_trace.replay import factorial_arms, run_factorial
from circ_trace.schema import ClinicalScope, ControlArm, Decision, FaultKind

ROOT = Path(__file__).resolve().parents[1]
CLINICARE_REPO = Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))
FIXTURE = ROOT / "fixtures" / "synthetic_med_reconciliation"


def _artifact():
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


def test_clean_twin_is_preserved_by_every_factorial_arm() -> None:
    clean = clean_variant(_artifact())
    assert len(factorial_arms()) == 8
    assert all(evaluate_variant(clean, arm).decision is Decision.RELEASE for arm in factorial_arms())


def test_each_fault_is_detected_only_when_its_intended_control_is_enabled() -> None:
    artifact = _artifact()
    intended_flag = {
        FaultKind.WRONG_SCOPE: "scope_enforcement",
        FaultKind.MISSING_REQUIRED_SOURCE: "dependency_gate",
        FaultKind.REQUIRED_SOURCE_ERROR: "dependency_gate",
        FaultKind.FABRICATED_CITATION: "provenance_gate",
    }
    for fault_kind, flag in intended_flag.items():
        variant = inject_fault(artifact, fault_kind)
        for arm in factorial_arms():
            outcome = evaluate_variant(variant, arm)
            expected = Decision.HOLD if getattr(arm, flag) else Decision.RELEASE
            assert outcome.decision is expected, (fault_kind, arm.arm_id, outcome.findings)


def test_factorial_retains_matched_clean_and_faulted_denominators() -> None:
    outcomes = run_factorial((_artifact(),))
    rows = summarize_by_arm(outcomes)

    assert len(outcomes) == 40  # 1 clean + 4 faulted variants across 8 arms
    assert len(rows) == 8
    assert all(row["n_clean"] == 1 and row["n_faulted"] == 4 for row in rows)
    assert all(row["clean_preservation"] == 1.0 for row in rows)
    assert next(row for row in rows if row["arm_id"] == "S0D0P0")["fault_containment"] == 0.0
    assert next(row for row in rows if row["arm_id"] == "S1D1P1")["fault_containment"] == 1.0


def test_controls_do_not_receive_fault_oracle() -> None:
    variant = inject_fault(_artifact(), FaultKind.FABRICATED_CITATION)
    outcome = evaluate_variant(variant, ControlArm(False, False, True))
    assert outcome.decision is Decision.HOLD
    assert {finding.code for finding in outcome.findings} == {"unsupported_record_reference"}


def test_scope_fault_can_be_injected_into_sql_only_trace() -> None:
    artifact = _artifact()
    sql_only = replace(
        artifact,
        events=tuple(event for event in artifact.events if event.tool == "run_sql"),
    )
    variant = inject_fault(sql_only, FaultKind.WRONG_SCOPE)
    outcome = evaluate_variant(variant, ControlArm(True, False, False))
    assert outcome.decision is Decision.HOLD
    assert {finding.target for finding in outcome.findings} == {"subject_id"}