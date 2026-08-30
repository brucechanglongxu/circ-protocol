from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from circ_trace.clinicare import load_codebook
from circ_trace.natural_risks import summarize_natural_risks
from circ_trace.protected import load_natural_risk_job, load_protected_job, pseudonymize

ROOT = Path(__file__).resolve().parents[1]
CLINICARE_REPO = Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))
FIXTURE = ROOT / "fixtures" / "synthetic_med_reconciliation"


def _build_job(tmp_path: Path, system_name: str):
    task_id = "med-reconciliation-abcdef"
    cohort = tmp_path / "full750.csv"
    with cohort.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "scenario_id", "params_json", "expected_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": task_id,
                "scenario_id": "med-reconciliation",
                "params_json": json.dumps(
                    {"subject_id": 99000001, "hadm_id": 98000001, "stay_id": 97000001}
                ),
                "expected_label": "YES",
            }
        )
    job = tmp_path / system_name
    verifier = job / f"{task_id}__trial" / "verifier"
    verifier.mkdir(parents=True)
    shutil.copyfile(FIXTURE / "evidence.jsonl", verifier / "evidence.jsonl")
    shutil.copyfile(FIXTURE / "report.md", verifier / "report.md")
    return cohort, job, task_id


def test_protected_loader_uses_stable_cross_system_case_clusters(tmp_path: Path) -> None:
    cohort, first_job, task_id = _build_job(tmp_path, "system-a")
    _, second_job, _ = _build_job(tmp_path, "system-b")
    first = load_protected_job(
        job_dir=first_job,
        cohort_csv=cohort,
        system_name="system-a",
        study_salt="unit-test-only",
        clinicare_repo=CLINICARE_REPO,
    )
    second = load_protected_job(
        job_dir=second_job,
        cohort_csv=cohort,
        system_name="system-b",
        study_salt="unit-test-only",
        clinicare_repo=CLINICARE_REPO,
    )

    assert first.flow == {"loaded": 1, "trial_dirs": 1}
    assert first.artifacts[0].cluster_id == second.artifacts[0].cluster_id
    assert first.artifacts[0].artifact_id != second.artifacts[0].artifact_id
    assert task_id not in first.artifacts[0].artifact_id
    assert "99000001" not in first.artifacts[0].artifact_id


def test_pseudonymization_requires_a_nonempty_salt() -> None:
    assert pseudonymize("case-a", "salt") == pseudonymize("case-a", "salt")
    assert pseudonymize("case-a", "salt") != pseudonymize("case-a", "other")
    try:
        pseudonymize("case-a", "")
    except ValueError:
        pass
    else:
        raise AssertionError("empty salt was accepted")


def test_natural_risk_summary_uses_complete_family_denominators(tmp_path: Path) -> None:
    cohort, job, _ = _build_job(tmp_path, "system-a")
    trial = next(job.glob("*__*"))
    process = {
        "graded": [
            {"id": "N1", "weight": "-2", "grade": "met"},
            {"id": "N3", "weight": "-2", "grade": "not met"},
            {"id": "N4", "weight": "-2", "grade": "not met"},
            {"id": "N5", "weight": "-2", "grade": "not met"},
        ]
    }
    (trial / "verifier" / "process.json").write_text(json.dumps(process))
    codebook = load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv")
    loaded = load_natural_risk_job(
        job_dir=job,
        cohort_csv=cohort,
        system_name="system-a",
        study_salt="unit-test-only",
        codebook=codebook,
    )
    summary = summarize_natural_risks(loaded.observations, codebook, min_cell_size=1)
    rows = {row["risk_family"]: row for row in summary["pooled"]}

    assert loaded.flow == {"loaded": 1, "trial_dirs": 1}
    assert rows["R-CLASS"]["rubric_trigger_rate"] == 1.0
    assert rows["R-COVER"]["rubric_trigger_rate"] == 0.0
    assert rows["R-PROV"]["rubric_trigger_rate"] == 0.0
    assert rows["R-UNCERT"]["rubric_trigger_rate"] == 0.0