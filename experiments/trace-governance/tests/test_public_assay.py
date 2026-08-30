from __future__ import annotations

import os
from pathlib import Path

from circ_trace.analysis import estimate_all_main_effects, estimate_targeted_effects
from circ_trace.endpoints import summarize_by_arm
from circ_trace.replay import run_factorial
from circ_trace.synthetic import generate_public_panel

ROOT = Path(__file__).resolve().parents[1]
CLINICARE_REPO = Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))


def test_public_assay_covers_all_scenarios_and_factorial_cells() -> None:
    artifacts = generate_public_panel(CLINICARE_REPO)
    outcomes = run_factorial(artifacts)
    rows = summarize_by_arm(outcomes)

    assert len(artifacts) == 25
    assert len({artifact.scenario_id for artifact in artifacts}) == 25
    assert len(outcomes) == 1_000
    assert all(row["n_clean"] == 25 for row in rows)
    assert all(row["n_faulted"] == 100 for row in rows)
    assert all(row["clean_preservation"] == 1.0 for row in rows)


def test_public_assay_has_expected_mechanism_specific_sensitivity() -> None:
    rows = {
        row["arm_id"]: row
        for row in summarize_by_arm(run_factorial(generate_public_panel(CLINICARE_REPO)))
    }
    assert rows["S0D0P0"]["fault_containment"] == 0.0
    assert rows["S1D0P0"]["fault_containment"] == 0.25
    assert rows["S0D1P0"]["fault_containment"] == 0.5
    assert rows["S0D0P1"]["fault_containment"] == 0.25
    assert rows["S1D1P1"]["fault_containment"] == 1.0

    assert rows["S1D0P0"]["fault_containment_by_type"] == {
        "fabricated_citation": 0.0,
        "missing_required_source": 0.0,
        "required_source_error": 0.0,
        "wrong_scope": 1.0,
    }


def test_paired_estimators_recover_known_assay_effects() -> None:
    outcomes = run_factorial(generate_public_panel(CLINICARE_REPO))
    effects = {
        (effect.endpoint, effect.lever): effect
        for effect in estimate_all_main_effects(outcomes, bootstrap_samples=200, seed=13)
    }

    assert effects[("fault_containment", "scope_enforcement")].estimate == 0.25
    assert effects[("fault_containment", "dependency_gate")].estimate == 0.5
    assert effects[("fault_containment", "provenance_gate")].estimate == 0.25
    assert all(
        effects[("base_preservation", lever)].estimate == 0.0
        for lever in ("scope_enforcement", "dependency_gate", "provenance_gate")
    )
    assert all(effect.case_cluster_count == 25 for effect in effects.values())

    targeted = {
        effect.lever: effect
        for effect in estimate_targeted_effects(outcomes, bootstrap_samples=200, seed=17)
    }
    assert all(effect.estimate == 1.0 for effect in targeted.values())
    assert targeted["scope_enforcement"].fault_kinds == ("wrong_scope",)
    assert targeted["dependency_gate"].fault_kinds == (
        "missing_required_source",
        "required_source_error",
    )