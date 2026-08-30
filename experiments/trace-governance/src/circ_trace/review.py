"""Agreement analysis for independently completed risk-codebook review sheets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .clinicare import RiskFamily


@dataclass(frozen=True)
class ReviewLabel:
    source_ref: str
    criterion_text: str
    primary_risk: str
    secondary_risks: tuple[str, ...]
    confidence: str


def load_completed_review(path: Path, risks: tuple[RiskFamily, ...]) -> dict[str, ReviewLabel]:
    valid_risks = {risk.risk_id for risk in risks}
    valid_confidence = {"high", "moderate", "low"}
    labels: dict[str, ReviewLabel] = {}
    errors: list[str] = []
    with path.open(newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            source_ref = row["source_ref"].strip()
            primary = row["primary_risk"].strip()
            secondary = tuple(filter(None, row["secondary_risks"].replace(";", "|").split("|")))
            confidence = row["confidence"].strip().lower()
            if not source_ref or source_ref in labels:
                errors.append(f"row {row_number}: missing or duplicate source_ref")
                continue
            if primary not in valid_risks:
                errors.append(f"row {row_number}: invalid primary_risk {primary!r}")
            if unknown := set(secondary) - valid_risks:
                errors.append(f"row {row_number}: invalid secondary_risks {sorted(unknown)}")
            if primary in secondary:
                errors.append(f"row {row_number}: primary risk repeated as secondary")
            if confidence not in valid_confidence:
                errors.append(f"row {row_number}: confidence must be high moderate or low")
            labels[source_ref] = ReviewLabel(
                source_ref=source_ref,
                criterion_text=row["criterion_text"],
                primary_risk=primary,
                secondary_risks=secondary,
                confidence=confidence,
            )
    if errors:
        raise ValueError(f"invalid review sheet {path}:\n- " + "\n- ".join(errors))
    return labels


def _cohen_kappa(first: list[str], second: list[str]) -> float:
    observed = sum(left == right for left, right in zip(first, second)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] / len(first) ** 2
        for label in set(first_counts) | set(second_counts)
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def _krippendorff_alpha_nominal(first: list[str], second: list[str]) -> float:
    observed_disagreement = sum(left != right for left, right in zip(first, second)) / len(first)
    pooled = Counter((*first, *second))
    assignments = 2 * len(first)
    expected_disagreement = (
        assignments**2 - sum(count**2 for count in pooled.values())
    ) / (assignments * (assignments - 1))
    return (
        1.0 - observed_disagreement / expected_disagreement
        if expected_disagreement > 0.0
        else 1.0
    )


def compare_reviews(
    first: dict[str, ReviewLabel], second: dict[str, ReviewLabel]
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if set(first) != set(second):
        missing_first = sorted(set(second) - set(first))
        missing_second = sorted(set(first) - set(second))
        raise ValueError(
            f"review sheets cover different items; missing from first={missing_first}; "
            f"missing from second={missing_second}"
        )
    refs = sorted(first)
    first_labels = [first[ref].primary_risk for ref in refs]
    second_labels = [second[ref].primary_risk for ref in refs]
    agreements = sum(left == right for left, right in zip(first_labels, second_labels))
    disagreements = [
        {
            "source_ref": ref,
            "criterion_text": first[ref].criterion_text,
            "reviewer_1_primary": first[ref].primary_risk,
            "reviewer_2_primary": second[ref].primary_risk,
            "reviewer_1_secondary": "|".join(first[ref].secondary_risks),
            "reviewer_2_secondary": "|".join(second[ref].secondary_risks),
            "reviewer_1_confidence": first[ref].confidence,
            "reviewer_2_confidence": second[ref].confidence,
            "adjudicated_primary": "",
            "adjudicated_secondary": "",
            "adjudication_notes": "",
        }
        for ref in refs
        if first[ref].primary_risk != second[ref].primary_risk
    ]
    return (
        {
            "item_count": len(refs),
            "primary_agreements": agreements,
            "primary_disagreements": len(refs) - agreements,
            "raw_primary_agreement": agreements / len(refs),
            "cohen_kappa": _cohen_kappa(first_labels, second_labels),
            "krippendorff_alpha_nominal": _krippendorff_alpha_nominal(
                first_labels, second_labels
            ),
            "reviewer_1_counts": dict(sorted(Counter(first_labels).items())),
            "reviewer_2_counts": dict(sorted(Counter(second_labels).items())),
        },
        disagreements,
    )


def write_review_comparison(
    *,
    first_path: Path,
    second_path: Path,
    risks: tuple[RiskFamily, ...],
    output_dir: Path,
) -> dict[str, object]:
    first = load_completed_review(first_path, risks)
    second = load_completed_review(second_path, risks)
    summary, disagreements = compare_reviews(first, second)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agreement.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "source_ref",
        "criterion_text",
        "reviewer_1_primary",
        "reviewer_2_primary",
        "reviewer_1_secondary",
        "reviewer_2_secondary",
        "reviewer_1_confidence",
        "reviewer_2_confidence",
        "adjudicated_primary",
        "adjudicated_secondary",
        "adjudication_notes",
    ]
    with (output_dir / "adjudication.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(disagreements)
    return summary