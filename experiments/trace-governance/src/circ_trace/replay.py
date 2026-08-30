"""Matched factorial replay over clean and faulted trace twins."""

from __future__ import annotations

from itertools import product

from .controls import evaluate_variant
from .faults import clean_variant, inject_fault
from .schema import ControlArm, FaultExclusion, FaultKind, ReplayOutcome, TrialArtifact


def factorial_arms() -> tuple[ControlArm, ...]:
    return tuple(ControlArm(*values) for values in product((False, True), repeat=3))


def build_variants(artifact: TrialArtifact) -> tuple:
    return (
        clean_variant(artifact),
        *(inject_fault(artifact, fault_kind) for fault_kind in FaultKind),
    )


def run_factorial(artifacts: tuple[TrialArtifact, ...]) -> tuple[ReplayOutcome, ...]:
    outcomes: list[ReplayOutcome] = []
    for artifact in artifacts:
        for variant in build_variants(artifact):
            for arm in factorial_arms():
                outcomes.append(evaluate_variant(variant, arm))
    return tuple(outcomes)


def run_eligible_factorial(
    artifacts: tuple[TrialArtifact, ...],
) -> tuple[tuple[ReplayOutcome, ...], tuple[FaultExclusion, ...]]:
    """Run every eligible fault while retaining explicit fault-specific exclusions."""
    outcomes: list[ReplayOutcome] = []
    exclusions: list[FaultExclusion] = []
    for artifact in artifacts:
        variants = [clean_variant(artifact)]
        for fault_kind in FaultKind:
            try:
                variants.append(inject_fault(artifact, fault_kind))
            except ValueError as exc:
                exclusions.append(
                    FaultExclusion(
                        base_artifact_id=artifact.artifact_id,
                        cluster_id=artifact.cluster_id,
                        fault_kind=fault_kind,
                        reason=str(exc),
                    )
                )
        for variant in variants:
            for arm in factorial_arms():
                outcomes.append(evaluate_variant(variant, arm))
    return tuple(outcomes), tuple(exclusions)