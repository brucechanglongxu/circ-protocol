"""Generate no-PHI assay traces from CliniCARE's public required-source profiles."""

from __future__ import annotations

from pathlib import Path

from .clinicare_adapter import CliniCAREPolicy, load_required_tables
from .schema import ClinicalScope, EvidenceEvent, TrialArtifact


def generate_public_panel(clinicare_repo: Path) -> tuple[TrialArtifact, ...]:
    """Create one clean metadata-only trace for each released CliniCARE scenario."""
    policy = CliniCAREPolicy(clinicare_repo)
    required_by_scenario = load_required_tables(clinicare_repo, policy)
    artifacts: list[TrialArtifact] = []

    for index, (scenario_id, required_tables) in enumerate(
        sorted(required_by_scenario.items()), start=1
    ):
        subject_id = str(90_000_000 + index)
        hadm_id = str(91_000_000 + index)
        stay_id = str(92_000_000 + index)
        scope = ClinicalScope(subject_id=subject_id, hadm_id=hadm_id, stay_id=stay_id)

        events = [
            EvidenceEvent(
                event_id="scope-anchor",
                kind="mimic",
                tool="get_data_availability",
                args={"subject_id": int(subject_id)},
                status="ok",
                n_rows=1,
                record_ids=(subject_id, hadm_id, stay_id),
                # CliniCARE deliberately assigns this tool no evidence-table credit. It
                # isolates scope enforcement from dependency coverage in the assay.
                tables=policy.tables_for("get_data_availability", {}),
            )
        ]
        for table_index, table in enumerate(sorted(required_tables), start=1):
            query = f"SELECT subject_id FROM {table} WHERE subject_id = {subject_id}"
            events.append(
                EvidenceEvent(
                    event_id=f"required-{table_index}",
                    kind="mimic",
                    tool="run_sql",
                    args={"query": query},
                    status="ok",
                    n_rows=1,
                    record_ids=(subject_id,),
                    tables=policy.tables_for("run_sql", {"query": query}),
                )
            )

        artifacts.append(
            TrialArtifact(
                artifact_id=f"synthetic-{scenario_id}",
                cluster_id=f"synthetic-case-{index:02d}",
                scenario_id=scenario_id,
                scope=scope,
                verdict="YES",
                events=tuple(events),
                cited_record_ids=(subject_id, hadm_id, stay_id),
                required_tables=required_tables,
            )
        )
    return tuple(artifacts)