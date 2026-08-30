"""Paired estimands with base-trace-clustered uncertainty."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .schema import Decision, FaultKind, ReplayOutcome

LEVERS = ("scope_enforcement", "dependency_gate", "provenance_gate")


@dataclass(frozen=True)
class EffectEstimate:
    lever: str
    endpoint: str
    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    case_cluster_count: int
    paired_comparison_count: int
    fault_kinds: tuple[str, ...]


def _arm_without(arm, lever: str) -> tuple[bool, bool]:
    return tuple(getattr(arm, name) for name in LEVERS if name != lever)  # type: ignore[return-value]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def estimate_main_effect(
    outcomes: tuple[ReplayOutcome, ...],
    *,
    lever: str,
    endpoint: str,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
    fault_kinds: frozenset[FaultKind] | None = None,
) -> EffectEstimate:
    """Estimate an arm-matched lever effect and cluster by immutable base trace."""
    if lever not in LEVERS:
        raise ValueError(f"unknown lever: {lever}")
    if endpoint not in {"fault_containment", "base_preservation"}:
        raise ValueError(f"unknown endpoint: {endpoint}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    eligible = [
        outcome
        for outcome in outcomes
        if (outcome.fault_kind is not None) == (endpoint == "fault_containment")
        and (
            endpoint != "fault_containment"
            or fault_kinds is None
            or outcome.fault_kind in fault_kinds
        )
    ]
    indexed: dict[tuple[str, str, tuple[bool, bool]], dict[bool, ReplayOutcome]] = defaultdict(dict)
    for outcome in eligible:
        key = (
            outcome.base_artifact_id,
            outcome.variant_id,
            _arm_without(outcome.arm, lever),
        )
        indexed[key][getattr(outcome.arm, lever)] = outcome

    cluster_by_base: dict[str, str] = {}
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for (base_id, _, _), pair in indexed.items():
        if set(pair) != {False, True}:
            raise ValueError(f"incomplete factorial pair for {base_id} and {lever}")
        clusters = {outcome.cluster_id for outcome in pair.values()}
        if len(clusters) != 1:
            raise ValueError(f"inconsistent cluster ids for {base_id}: {sorted(clusters)}")
        cluster_by_base[base_id] = next(iter(clusters))
        if endpoint == "fault_containment":
            off = float(pair[False].decision is Decision.HOLD)
            on = float(pair[True].decision is Decision.HOLD)
        else:
            off = float(pair[False].decision is Decision.RELEASE)
            on = float(pair[True].decision is Decision.RELEASE)
        by_cluster[cluster_by_base[base_id]].append(on - off)

    cluster_ids = sorted(by_cluster)
    if not cluster_ids:
        raise ValueError(f"no eligible outcomes for {endpoint}")
    observed_values = [value for cluster_id in cluster_ids for value in by_cluster[cluster_id]]
    estimate = sum(observed_values) / len(observed_values)

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in sampled for value in by_cluster[cluster_id]]
        replicates.append(sum(values) / len(values))
    alpha = 1.0 - confidence_level

    return EffectEstimate(
        lever=lever,
        endpoint=endpoint,
        estimate=estimate,
        ci_low=_percentile(replicates, alpha / 2.0),
        ci_high=_percentile(replicates, 1.0 - alpha / 2.0),
        confidence_level=confidence_level,
        case_cluster_count=len(cluster_ids),
        paired_comparison_count=len(observed_values),
        fault_kinds=tuple(sorted(item.value for item in fault_kinds)) if fault_kinds else (),
    )


def estimate_all_main_effects(
    outcomes: tuple[ReplayOutcome, ...],
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> list[EffectEstimate]:
    return [
        estimate_main_effect(
            outcomes,
            lever=lever,
            endpoint=endpoint,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        for index, (endpoint, lever) in enumerate(
            (endpoint, lever)
            for endpoint in ("fault_containment", "base_preservation")
            for lever in LEVERS
        )
    ]


def estimate_targeted_effects(
    outcomes: tuple[ReplayOutcome, ...],
    *,
    confidence_level: float = 0.9833,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> list[EffectEstimate]:
    targets = {
        "scope_enforcement": frozenset({FaultKind.WRONG_SCOPE}),
        "dependency_gate": frozenset(
            {FaultKind.MISSING_REQUIRED_SOURCE, FaultKind.REQUIRED_SOURCE_ERROR}
        ),
        "provenance_gate": frozenset({FaultKind.FABRICATED_CITATION}),
    }
    return [
        estimate_main_effect(
            outcomes,
            lever=lever,
            endpoint="fault_containment",
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
            fault_kinds=fault_kinds,
        )
        for index, (lever, fault_kinds) in enumerate(targets.items())
    ]