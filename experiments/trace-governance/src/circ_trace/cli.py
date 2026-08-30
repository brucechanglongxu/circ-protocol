"""Command-line entry points for public validation and trace replay."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .analysis import estimate_all_main_effects, estimate_targeted_effects
from .clinicare import (
    load_codebook,
    load_process_criteria,
    load_risk_taxonomy,
    source_digest,
    validate_codebook,
)
from .endpoints import summarize_by_arm
from .natural_risks import summarize_natural_risks
from .protected import load_natural_risk_job, load_protected_job
from .public_mapping import write_public_mapping_outputs
from .replay import run_eligible_factorial, run_factorial
from .review import write_review_comparison
from .synthetic import generate_public_panel

ROOT = Path(__file__).resolve().parents[2]


def _default_clinicare_repo() -> Path:
    return Path(os.environ.get("CLINICARE_REPO", ROOT.parents[2] / "clinicare"))


def _verify(clinicare_repo: Path) -> dict[str, object]:
    provenance = tomllib.loads((ROOT / "configs" / "clinicare_source.toml").read_text())
    actual_digest = source_digest(clinicare_repo)
    if actual_digest != provenance["scenario_rubric_digest"]:
        raise ValueError(
            "CliniCARE source digest mismatch: "
            f"expected {provenance['scenario_rubric_digest']} got {actual_digest}"
        )

    criteria = load_process_criteria(clinicare_repo)
    risks = load_risk_taxonomy(ROOT / "configs" / "risk_taxonomy.toml")
    entries = load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv")
    validate_codebook(criteria, risks, entries)
    return {
        "source_commit": provenance["commit"],
        "source_digest": actual_digest,
        "scenario_count": len({criterion.scenario_id for criterion in criteria}),
        "criterion_count": len(criteria),
        "prohibited_criterion_count": sum(item.is_prohibited for item in criteria),
        "risk_family_count": len(risks),
        "primary_mapping_counts": dict(sorted(Counter(x.primary_risk for x in entries).items())),
        "boundary_review_count": sum(x.review_priority == "boundary" for x in entries),
    }


def _write(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinicare-repo", type=Path, default=_default_clinicare_repo())
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify-public-source")
    verify_parser.add_argument("--out", type=Path)

    replay_parser = subparsers.add_parser("run-public-assay")
    replay_parser.add_argument("--out", type=Path)

    mapping_parser = subparsers.add_parser("run-public-mapping")
    mapping_parser.add_argument("--out-dir", type=Path, required=True)

    review_parser = subparsers.add_parser("score-codebook-review")
    review_parser.add_argument("--reviewer-1", type=Path, required=True)
    review_parser.add_argument("--reviewer-2", type=Path, required=True)
    review_parser.add_argument("--out-dir", type=Path, required=True)

    protected_parser = subparsers.add_parser("run-protected-replay")
    protected_parser.add_argument("--cohort-csv", type=Path, required=True)
    protected_parser.add_argument(
        "--job",
        action="append",
        required=True,
        metavar="SYSTEM=PATH",
        help="repeat for each CliniCARE system job",
    )
    protected_parser.add_argument("--out", type=Path, required=True)
    protected_parser.add_argument("--bootstrap-samples", type=int, default=10_000)

    args = parser.parse_args()
    if args.command == "verify-public-source":
        _write({"schema": "circ/public-source-verification.v1", **_verify(args.clinicare_repo)}, args.out)
        return 0

    if args.command == "run-public-mapping":
        _verify(args.clinicare_repo)
        criteria = load_process_criteria(args.clinicare_repo)
        risks = load_risk_taxonomy(ROOT / "configs" / "risk_taxonomy.toml")
        entries = load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv")
        write_public_mapping_outputs(
            output_dir=args.out_dir,
            criteria=criteria,
            risks=risks,
            entries=entries,
        )
        print(args.out_dir)
        return 0

    if args.command == "score-codebook-review":
        risks = load_risk_taxonomy(ROOT / "configs" / "risk_taxonomy.toml")
        summary = write_review_comparison(
            first_path=args.reviewer_1,
            second_path=args.reviewer_2,
            risks=risks,
            output_dir=args.out_dir,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "run-protected-replay":
        study_salt = os.environ.get("CIRC_STUDY_SALT", "")
        if not study_salt:
            parser.error("set CIRC_STUDY_SALT directly in the terminal before protected replay")
        artifacts = []
        flow: dict[str, dict[str, int]] = {}
        natural_observations = []
        natural_flow: dict[str, dict[str, int]] = {}
        codebook = load_codebook(ROOT / "configs" / "clinicare_must_not_codebook.csv")
        for job_spec in args.job:
            system_name, separator, raw_path = job_spec.partition("=")
            if not separator or not system_name or not raw_path:
                parser.error(f"invalid --job {job_spec!r}; expected SYSTEM=PATH")
            loaded = load_protected_job(
                job_dir=Path(raw_path),
                cohort_csv=args.cohort_csv,
                system_name=system_name,
                study_salt=study_salt,
                clinicare_repo=args.clinicare_repo,
            )
            artifacts.extend(loaded.artifacts)
            flow[system_name] = loaded.flow
            natural = load_natural_risk_job(
                job_dir=Path(raw_path),
                cohort_csv=args.cohort_csv,
                system_name=system_name,
                study_salt=study_salt,
                codebook=codebook,
            )
            natural_observations.extend(natural.observations)
            natural_flow[system_name] = natural.flow
        outcomes, exclusions = run_eligible_factorial(tuple(artifacts))
        payload = {
            "schema": "circ/protected-trace-replay.v1",
            "source": _verify(args.clinicare_repo),
            "system_flow": flow,
            "natural_risk_flow": natural_flow,
            "loaded_trace_count": len(artifacts),
            "fault_exclusions": dict(
                sorted(Counter(item.fault_kind.value for item in exclusions).items())
            ),
            "outcome_count": len(outcomes),
            "arms": summarize_by_arm(outcomes),
            "paired_main_effects": [
                asdict(item)
                for item in estimate_all_main_effects(
                    outcomes,
                    confidence_level=0.9833,
                    bootstrap_samples=args.bootstrap_samples,
                )
            ],
            "confirmatory_targeted_effects": [
                asdict(item)
                for item in estimate_targeted_effects(
                    outcomes,
                    confidence_level=0.9833,
                    bootstrap_samples=args.bootstrap_samples,
                )
            ],
            "natural_risk_summary": summarize_natural_risks(
                tuple(natural_observations), codebook
            ),
        }
        _write(payload, args.out)
        return 0

    verification = _verify(args.clinicare_repo)
    artifacts = generate_public_panel(args.clinicare_repo)
    outcomes = run_factorial(artifacts)
    payload = {
        "schema": "circ/public-synthetic-assay.v1",
        "interpretation": "software assay only; not an estimate of clinical effectiveness",
        "source": verification,
        "base_trace_count": len(artifacts),
        "faults_per_trace": 4,
        "arm_count": 8,
        "outcome_count": len(outcomes),
        "arms": summarize_by_arm(outcomes),
        "paired_main_effects": [asdict(item) for item in estimate_all_main_effects(outcomes)],
        "confirmatory_targeted_effects": [
            asdict(item)
            for item in estimate_targeted_effects(outcomes, bootstrap_samples=2_000)
        ],
    }
    _write(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())