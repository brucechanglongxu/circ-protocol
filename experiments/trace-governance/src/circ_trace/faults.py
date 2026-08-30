"""Deterministic trace mutations with hidden fault oracles."""

from __future__ import annotations

import re
from dataclasses import replace

from .schema import EvidenceEvent, FaultKind, FaultOracle, TraceVariant, TrialArtifact


def clean_variant(artifact: TrialArtifact) -> TraceVariant:
    return TraceVariant(
        variant_id=f"{artifact.artifact_id}__clean",
        base_artifact_id=artifact.artifact_id,
        cluster_id=artifact.cluster_id,
        artifact=artifact,
        oracle=None,
    )


def _different_identifier(value: object) -> object:
    text = str(value)
    if text.isdigit():
        replacement = str(int(text) + 1)
        return int(replacement) if isinstance(value, int) else replacement
    return f"{text}-other"


def _replace_events(artifact: TrialArtifact, events: list[EvidenceEvent]) -> TrialArtifact:
    return replace(artifact, events=tuple(events))


def _mutate_sql_scope(artifact: TrialArtifact, events: list[EvidenceEvent]) -> tuple[int, str] | None:
    for index, event in enumerate(events):
        if event.kind != "mimic" or event.tool != "run_sql":
            continue
        args = dict(event.args)
        query_key = "query" if "query" in args else "sql" if "sql" in args else None
        if query_key is None:
            continue
        query = str(args[query_key])
        for field in ("subject_id", "hadm_id", "stay_id"):
            expected = artifact.scope.expected(field)
            if expected is None:
                continue
            pattern = re.compile(
                rf"(\b{field}\s*=\s*['\"]?){re.escape(expected)}(['\"]?)",
                re.IGNORECASE,
            )
            changed, count = pattern.subn(
                rf"\g<1>{_different_identifier(expected)}\g<2>", query, count=1
            )
            if count:
                args[query_key] = changed
                events[index] = replace(event, args=args)
                return index, field
    return None


def inject_fault(artifact: TrialArtifact, fault_kind: FaultKind) -> TraceVariant:
    """Create one matched faulted twin without exposing its oracle to the controls."""
    if fault_kind is FaultKind.WRONG_SCOPE:
        events = list(artifact.events)
        target_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.kind == "mimic"
                and any(field in event.args for field in ("subject_id", "hadm_id", "stay_id"))
            ),
            None,
        )
        if target_index is not None:
            event = events[target_index]
            args = dict(event.args)
            field = next(name for name in ("subject_id", "hadm_id", "stay_id") if name in args)
            args[field] = _different_identifier(args[field])
            events[target_index] = replace(event, args=args)
        else:
            sql_target = _mutate_sql_scope(artifact, events)
            if sql_target is None:
                raise ValueError(f"no explicit scope predicate in {artifact.artifact_id}")
            _, field = sql_target
        mutated = _replace_events(artifact, events)
        oracle = FaultOracle(fault_kind, field, "scope")

    elif fault_kind in {FaultKind.MISSING_REQUIRED_SOURCE, FaultKind.REQUIRED_SOURCE_ERROR}:
        coverage: dict[str, list[int]] = {table: [] for table in artifact.required_tables}
        for index, event in enumerate(artifact.events):
            for table in event.tables & artifact.required_tables:
                coverage[table].append(index)
        eligible = [table for table, indices in coverage.items() if indices]
        if not eligible:
            raise ValueError(f"no required-source event in {artifact.artifact_id}")

        def collateral_cost(table: str) -> tuple[int, int, str]:
            removed = set(coverage[table])
            uncovered = sum(
                bool(indices) and set(indices) <= removed for indices in coverage.values()
            )
            return uncovered, len(removed), table

        target = min(eligible, key=collateral_cost)
        target_indices = set(coverage[target])
        if fault_kind is FaultKind.MISSING_REQUIRED_SOURCE:
            events = [
                event for index, event in enumerate(artifact.events) if index not in target_indices
            ]
        else:
            events = [
                replace(event, status="error", n_rows=None, record_ids=())
                if index in target_indices
                else event
                for index, event in enumerate(artifact.events)
            ]
        mutated = _replace_events(artifact, events)
        oracle = FaultOracle(fault_kind, target, "dependency")

    elif fault_kind is FaultKind.FABRICATED_CITATION:
        trusted = {identifier for event in artifact.events for identifier in event.record_ids}
        fabricated = "99999999"
        while fabricated in trusted:
            fabricated = str(int(fabricated) - 1)
        mutated = replace(
            artifact,
            cited_record_ids=tuple(sorted({*artifact.cited_record_ids, fabricated})),
        )
        oracle = FaultOracle(fault_kind, "unsupported_record_id", "provenance")

    else:  # pragma: no cover - exhaustive over FaultKind
        raise ValueError(f"unsupported fault kind: {fault_kind}")

    return TraceVariant(
        variant_id=f"{artifact.artifact_id}__{fault_kind.value}",
        base_artifact_id=artifact.artifact_id,
        cluster_id=artifact.cluster_id,
        artifact=mutated,
        oracle=oracle,
    )