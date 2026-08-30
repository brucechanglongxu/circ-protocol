"""Independent governance controls evaluated over normalized trial artifacts."""

from __future__ import annotations

import re

from .schema import ControlArm, Decision, GateFinding, ReplayOutcome, TraceVariant

_SCOPE_FIELDS = ("subject_id", "hadm_id", "stay_id")
_VALID_EVIDENCE_STATUSES = {"ok", "unattributed", "spilled"}


def _event_is_consulted(event) -> bool:
    if event.kind != "mimic":
        return False
    if event.status in _VALID_EVIDENCE_STATUSES:
        return True
    return event.status == "empty" and event.n_rows == 0


def _query_scope_values(query: str, field: str) -> set[str]:
    pattern = re.compile(rf"\b{field}\s*=\s*['\"]?(\d+)['\"]?", re.IGNORECASE)
    return {match.group(1) for match in pattern.finditer(query)}


def scope_findings(variant: TraceVariant) -> tuple[GateFinding, ...]:
    findings: list[GateFinding] = []
    for event in variant.artifact.events:
        if event.kind != "mimic":
            continue
        for field in _SCOPE_FIELDS:
            expected = variant.artifact.scope.expected(field)
            if expected is None:
                continue
            actual = {str(event.args[field])} if field in event.args else set()
            if event.tool == "run_sql":
                query = str(event.args.get("query") or event.args.get("sql") or "")
                actual.update(_query_scope_values(query, field))
            if actual and actual != {expected}:
                findings.append(GateFinding("scope", "scope_mismatch", field))
    return tuple(findings)


def dependency_findings(variant: TraceVariant) -> tuple[GateFinding, ...]:
    retrieved: set[str] = set()
    for event in variant.artifact.events:
        if _event_is_consulted(event):
            retrieved.update(event.tables)
    return tuple(
        GateFinding("dependency", "required_source_unresolved", table)
        for table in sorted(variant.artifact.required_tables - retrieved)
    )


def provenance_findings(variant: TraceVariant) -> tuple[GateFinding, ...]:
    trusted_ids = {
        identifier
        for event in variant.artifact.events
        if event.kind == "mimic" and _event_is_consulted(event)
        for identifier in event.record_ids
    }
    unsupported = set(variant.artifact.cited_record_ids) - trusted_ids
    return tuple(
        GateFinding("provenance", "unsupported_record_reference", "record_id")
        for _ in sorted(unsupported)
    )


def evaluate_variant(variant: TraceVariant, arm: ControlArm) -> ReplayOutcome:
    """Apply enabled controls without exposing the variant's fault oracle."""
    findings: list[GateFinding] = []
    if arm.scope_enforcement:
        findings.extend(scope_findings(variant))
    if arm.dependency_gate:
        findings.extend(dependency_findings(variant))
    if arm.provenance_gate:
        findings.extend(provenance_findings(variant))
    return ReplayOutcome(
        arm=arm,
        variant_id=variant.variant_id,
        base_artifact_id=variant.base_artifact_id,
        cluster_id=variant.cluster_id,
        fault_kind=variant.oracle.fault_kind if variant.oracle else None,
        intended_control=variant.oracle.intended_control if variant.oracle else None,
        decision=Decision.HOLD if findings else Decision.RELEASE,
        findings=tuple(findings),
    )