"""Load protected CliniCARE jobs without exporting patient-linked identifiers."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .clinicare import CodebookEntry
from .clinicare_adapter import load_trial_artifact
from .schema import ClinicalScope, NaturalRiskObservation, TrialArtifact

_ANSWER = re.compile(r"^\s*##\s*Answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class ProtectedLoadResult:
    artifacts: tuple[TrialArtifact, ...]
    flow: dict[str, int]


@dataclass(frozen=True)
class NaturalRiskLoadResult:
    observations: tuple[NaturalRiskObservation, ...]
    flow: dict[str, int]


def pseudonymize(value: str, salt: str) -> str:
    if not salt:
        raise ValueError("study salt must not be empty")
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:20]


def _cohort_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"empty CliniCARE cohort: {path}")
    return rows


def _predicted_verdict(report: str) -> str:
    match = _ANSWER.search(report)
    return match.group(1).strip().upper() if match else "UNPARSED"


def load_protected_job(
    *,
    job_dir: Path,
    cohort_csv: Path,
    system_name: str,
    study_salt: str,
    clinicare_repo: Path,
) -> ProtectedLoadResult:
    """Join one job to its gated cohort and retain only pseudonymous artifacts."""
    cohort = _cohort_rows(cohort_csv)
    artifacts: list[TrialArtifact] = []
    flow: Counter[str] = Counter()

    for trial_dir in sorted(path for path in job_dir.glob("*__*") if path.is_dir()):
        flow["trial_dirs"] += 1
        task_id = trial_dir.name.rsplit("__", 1)[0]
        row = cohort.get(task_id)
        if row is None:
            flow["unknown_task"] += 1
            continue
        evidence_path = trial_dir / "verifier" / "evidence.jsonl"
        report_path = trial_dir / "verifier" / "report.md"
        if not evidence_path.is_file():
            flow["missing_evidence"] += 1
            continue
        if not report_path.is_file():
            flow["missing_report"] += 1
            continue

        try:
            params = json.loads(row["params_json"])
            scope = ClinicalScope(
                subject_id=str(params["subject_id"]) if params.get("subject_id") is not None else None,
                hadm_id=str(params["hadm_id"]) if params.get("hadm_id") is not None else None,
                stay_id=str(params["stay_id"]) if params.get("stay_id") is not None else None,
            )
            report = report_path.read_text()
            cluster_id = pseudonymize(task_id, study_salt)
            artifact = load_trial_artifact(
                artifact_id=pseudonymize(f"{system_name}:{task_id}", study_salt),
                cluster_id=cluster_id,
                scenario_id=row["scenario_id"],
                scope=scope,
                verdict=_predicted_verdict(report),
                evidence_path=evidence_path,
                report_path=report_path,
                clinicare_repo=clinicare_repo,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            flow["parse_error"] += 1
            continue
        artifacts.append(artifact)
        flow["loaded"] += 1

    return ProtectedLoadResult(tuple(artifacts), dict(sorted(flow.items())))


def load_natural_risk_job(
    *,
    job_dir: Path,
    cohort_csv: Path,
    system_name: str,
    study_salt: str,
    codebook: tuple[CodebookEntry, ...],
) -> NaturalRiskLoadResult:
    """Load post-hoc process grades and retain no report or trajectory content."""
    cohort = _cohort_rows(cohort_csv)
    mapped = {entry.key for entry in codebook}
    observations: list[NaturalRiskObservation] = []
    flow: Counter[str] = Counter()

    for trial_dir in sorted(path for path in job_dir.glob("*__*") if path.is_dir()):
        flow["trial_dirs"] += 1
        task_id = trial_dir.name.rsplit("__", 1)[0]
        row = cohort.get(task_id)
        if row is None:
            flow["unknown_task"] += 1
            continue
        process_path = trial_dir / "verifier" / "process.json"
        if not process_path.is_file():
            flow["missing_process_grade"] += 1
            continue
        try:
            payload = json.loads(process_path.read_text())
        except (OSError, json.JSONDecodeError):
            flow["invalid_process_grade"] += 1
            continue
        if payload.get("error"):
            flow["judge_error"] += 1
            continue

        scenario_id = row["scenario_id"]
        graded: set[tuple[str, str]] = set()
        triggered: set[tuple[str, str]] = set()
        for grade in payload.get("graded") or ():
            key = (scenario_id, str(grade.get("id") or ""))
            if key not in mapped:
                continue
            graded.add(key)
            if str(grade.get("grade") or "").lower() in {"met", "pass"}:
                triggered.add(key)

        if not graded:
            flow["no_mapped_criteria"] += 1
            continue
        observations.append(
            NaturalRiskObservation(
                artifact_id=pseudonymize(f"{system_name}:{task_id}", study_salt),
                cluster_id=pseudonymize(task_id, study_salt),
                system_name=system_name,
                scenario_id=scenario_id,
                graded_criteria=frozenset(graded),
                triggered_criteria=frozenset(triggered),
            )
        )
        flow["loaded"] += 1

    return NaturalRiskLoadResult(tuple(observations), dict(sorted(flow.items())))