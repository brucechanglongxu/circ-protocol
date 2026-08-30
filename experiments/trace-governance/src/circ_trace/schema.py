"""Data-minimized schemas for trace replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class FaultKind(StrEnum):
    WRONG_SCOPE = "wrong_scope"
    MISSING_REQUIRED_SOURCE = "missing_required_source"
    REQUIRED_SOURCE_ERROR = "required_source_error"
    FABRICATED_CITATION = "fabricated_citation"


class Decision(StrEnum):
    RELEASE = "release"
    HOLD = "hold"


@dataclass(frozen=True)
class ClinicalScope:
    subject_id: str | None = None
    hadm_id: str | None = None
    stay_id: str | None = None

    def expected(self, field: str) -> str | None:
        if field not in {"subject_id", "hadm_id", "stay_id"}:
            raise KeyError(field)
        return getattr(self, field)


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    kind: str
    tool: str
    args: Mapping[str, object]
    status: str
    n_rows: int | None
    record_ids: tuple[str, ...]
    tables: frozenset[str]


@dataclass(frozen=True)
class TrialArtifact:
    artifact_id: str
    cluster_id: str
    scenario_id: str
    scope: ClinicalScope
    verdict: str
    events: tuple[EvidenceEvent, ...]
    cited_record_ids: tuple[str, ...]
    required_tables: frozenset[str]


@dataclass(frozen=True)
class FaultOracle:
    fault_kind: FaultKind
    target: str
    intended_control: str


@dataclass(frozen=True)
class TraceVariant:
    variant_id: str
    base_artifact_id: str
    cluster_id: str
    artifact: TrialArtifact
    oracle: FaultOracle | None

    @property
    def is_faulted(self) -> bool:
        return self.oracle is not None


@dataclass(frozen=True)
class ControlArm:
    scope_enforcement: bool
    dependency_gate: bool
    provenance_gate: bool

    @property
    def arm_id(self) -> str:
        return f"S{int(self.scope_enforcement)}D{int(self.dependency_gate)}P{int(self.provenance_gate)}"


@dataclass(frozen=True)
class GateFinding:
    control: str
    code: str
    target: str


@dataclass(frozen=True)
class ReplayOutcome:
    arm: ControlArm
    variant_id: str
    base_artifact_id: str
    cluster_id: str
    fault_kind: FaultKind | None
    intended_control: str | None
    decision: Decision
    findings: tuple[GateFinding, ...]


@dataclass(frozen=True)
class FaultExclusion:
    base_artifact_id: str
    cluster_id: str
    fault_kind: FaultKind
    reason: str


@dataclass(frozen=True)
class NaturalRiskObservation:
    artifact_id: str
    cluster_id: str
    system_name: str
    scenario_id: str
    graded_criteria: frozenset[tuple[str, str]]
    triggered_criteria: frozenset[tuple[str, str]]