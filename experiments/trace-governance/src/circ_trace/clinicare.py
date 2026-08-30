"""Adapters for CliniCARE's public scenario and process-rubric artifacts."""

from __future__ import annotations

import csv
import hashlib
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessCriterion:
    scenario_id: str
    criterion_id: str
    weight: int
    title: str
    source: Path

    @property
    def is_prohibited(self) -> bool:
        return self.weight < 0


@dataclass(frozen=True)
class RiskFamily:
    risk_id: str
    name: str
    definition: str
    excludes: str


@dataclass(frozen=True)
class CodebookEntry:
    scenario_id: str
    criterion_id: str
    primary_risk: str
    secondary_risks: tuple[str, ...]
    review_priority: str
    rationale: str

    @property
    def key(self) -> tuple[str, str]:
        return self.scenario_id, self.criterion_id


def load_process_criteria(clinicare_repo: Path) -> tuple[ProcessCriterion, ...]:
    """Load all signed process criteria from a CliniCARE checkout."""
    scenarios_dir = clinicare_repo / "benchmark" / "_scenarios"
    criteria: list[ProcessCriterion] = []
    seen: set[tuple[str, str]] = set()

    for scenario_dir in sorted(scenarios_dir.glob("scenario-*")):
        scenario_path = scenario_dir / "scenario.toml"
        rubric_path = scenario_dir / "rubrics.toml"
        if not scenario_path.is_file() or not rubric_path.is_file():
            raise ValueError(f"invalid CliniCARE scenario directory: {scenario_dir}")

        scenario_id = str(tomllib.loads(scenario_path.read_text())["id"])
        raw_criteria = tomllib.loads(rubric_path.read_text()).get("rubric", [])
        for raw in raw_criteria:
            criterion_id = str(raw["id"])
            key = (scenario_id, criterion_id)
            if key in seen:
                raise ValueError(f"duplicate process criterion: {scenario_id}/{criterion_id}")
            seen.add(key)
            weight = int(raw["weight"])
            if weight == 0:
                raise ValueError(f"zero-weight process criterion: {scenario_id}/{criterion_id}")
            criteria.append(
                ProcessCriterion(
                    scenario_id=scenario_id,
                    criterion_id=criterion_id,
                    weight=weight,
                    title=str(raw["title"]).strip(),
                    source=rubric_path,
                )
            )
    return tuple(criteria)


def load_risk_taxonomy(path: Path) -> tuple[RiskFamily, ...]:
    raw = tomllib.loads(path.read_text())
    return tuple(
        RiskFamily(
            risk_id=str(item["id"]),
            name=str(item["name"]),
            definition=str(item["definition"]),
            excludes=str(item["excludes"]),
        )
        for item in raw["risk"]
    )


def load_codebook(path: Path) -> tuple[CodebookEntry, ...]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return tuple(
            CodebookEntry(
                scenario_id=row["scenario_id"],
                criterion_id=row["criterion_id"],
                primary_risk=row["primary_risk"],
                secondary_risks=tuple(filter(None, row["secondary_risks"].split("|"))),
                review_priority=row["review_priority"],
                rationale=row["rationale"],
            )
            for row in rows
        )


def source_digest(clinicare_repo: Path) -> str:
    """Match the preregistered digest over CliniCARE scenario and rubric files."""
    paths = sorted(
        path
        for pattern in ("scenario.toml", "rubrics.toml")
        for path in (clinicare_repo / "benchmark" / "_scenarios").glob(f"scenario-*/{pattern}")
    )
    digest = hashlib.sha256()
    for path in paths:
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(clinicare_repo).as_posix()
        digest.update(f"{file_hash}  {relative}\n".encode())
    return digest.hexdigest()


def validate_codebook(
    criteria: tuple[ProcessCriterion, ...],
    risks: tuple[RiskFamily, ...],
    entries: tuple[CodebookEntry, ...],
) -> None:
    """Fail loudly if the codebook is incomplete inconsistent or out of scope."""
    errors: list[str] = []
    valid_risks = {risk.risk_id for risk in risks}
    prohibited = {(item.scenario_id, item.criterion_id) for item in criteria if item.is_prohibited}
    entry_keys = [entry.key for entry in entries]

    duplicates = sorted(key for key, count in Counter(entry_keys).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate codebook keys: {duplicates}")

    missing = sorted(prohibited - set(entry_keys))
    extra = sorted(set(entry_keys) - prohibited)
    if missing:
        errors.append(f"unmapped prohibited criteria: {missing}")
    if extra:
        errors.append(f"non-prohibited or unknown criteria in codebook: {extra}")

    for entry in entries:
        labels = (entry.primary_risk, *entry.secondary_risks)
        unknown = sorted(set(labels) - valid_risks)
        if unknown:
            errors.append(f"unknown risks for {entry.key}: {unknown}")
        if entry.primary_risk in entry.secondary_risks:
            errors.append(f"primary risk repeated as secondary for {entry.key}")
        if entry.review_priority not in {"standard", "boundary"}:
            errors.append(f"invalid review priority for {entry.key}: {entry.review_priority}")

    if errors:
        raise ValueError("invalid CliniCARE risk codebook:\n- " + "\n- ".join(errors))