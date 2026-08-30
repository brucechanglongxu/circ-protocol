"""Preregistered paired endpoints for governance-control replay."""

from __future__ import annotations

from collections import defaultdict

from .schema import Decision, ReplayOutcome


def summarize_by_arm(outcomes: tuple[ReplayOutcome, ...]) -> list[dict[str, object]]:
    grouped: dict[str, list[ReplayOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.arm.arm_id].append(outcome)

    rows: list[dict[str, object]] = []
    for arm_id, arm_outcomes in sorted(grouped.items()):
        clean = [outcome for outcome in arm_outcomes if outcome.fault_kind is None]
        faulted = [outcome for outcome in arm_outcomes if outcome.fault_kind is not None]
        by_fault = {
            fault_kind.value: sum(
                outcome.decision is Decision.HOLD
                for outcome in faulted
                if outcome.fault_kind is fault_kind
            )
            / sum(outcome.fault_kind is fault_kind for outcome in faulted)
            for fault_kind in sorted(
                {outcome.fault_kind for outcome in faulted if outcome.fault_kind is not None},
                key=lambda item: item.value,
            )
        }
        rows.append(
            {
                "arm_id": arm_id,
                "scope_enforcement": arm_outcomes[0].arm.scope_enforcement,
                "dependency_gate": arm_outcomes[0].arm.dependency_gate,
                "provenance_gate": arm_outcomes[0].arm.provenance_gate,
                "n_clean": len(clean),
                "n_faulted": len(faulted),
                "clean_preservation": sum(
                    outcome.decision is Decision.RELEASE for outcome in clean
                )
                / len(clean),
                "fault_containment": sum(
                    outcome.decision is Decision.HOLD for outcome in faulted
                )
                / len(faulted),
                "fault_containment_by_type": by_fault,
            }
        )
    return rows