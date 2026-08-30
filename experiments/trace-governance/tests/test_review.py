from __future__ import annotations

import csv
from pathlib import Path

import pytest

from circ_trace.clinicare import RiskFamily
from circ_trace.review import compare_reviews, load_completed_review, write_review_comparison

RISKS = (
    RiskFamily("R-A", "A", "A risk", ""),
    RiskFamily("R-B", "B", "B risk", ""),
)


def _write_review(path: Path, labels: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "source_ref",
                "criterion_text",
                "primary_risk",
                "secondary_risks",
                "confidence",
                "notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, label in enumerate(labels, start=1):
            writer.writerow(
                {
                    "item_id": f"ITEM-{index:03d}",
                    "source_ref": f"scenario/N{index}",
                    "criterion_text": f"Criterion {index}",
                    "primary_risk": label,
                    "secondary_risks": "",
                    "confidence": "high",
                    "notes": "",
                }
            )


def test_review_agreement_and_adjudication_output(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_review(first_path, ["R-A", "R-A", "R-B", "R-B"])
    _write_review(second_path, ["R-A", "R-B", "R-B", "R-B"])

    first = load_completed_review(first_path, RISKS)
    second = load_completed_review(second_path, RISKS)
    summary, disagreements = compare_reviews(first, second)

    assert summary["raw_primary_agreement"] == 0.75
    assert summary["cohen_kappa"] == 0.5
    assert summary["krippendorff_alpha_nominal"] == pytest.approx(0.5333333333)
    assert len(disagreements) == 1

    written = write_review_comparison(
        first_path=first_path,
        second_path=second_path,
        risks=RISKS,
        output_dir=tmp_path / "out",
    )
    assert written == summary
    assert (tmp_path / "out" / "agreement.json").is_file()
    with (tmp_path / "out" / "adjudication.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["adjudicated_primary"] == ""


def test_incomplete_review_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.csv"
    _write_review(path, ["R-A", ""])
    with pytest.raises(ValueError, match="invalid primary_risk"):
        load_completed_review(path, RISKS)