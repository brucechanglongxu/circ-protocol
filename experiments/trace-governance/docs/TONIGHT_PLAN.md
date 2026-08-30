# Tonight Plan

This path requires no MIMIC data, Docker, model credentials, or paid inference.

1. Run `make mapping` from this directory.
2. Give `results/public_mapping/reviewer_1.csv` and `reviewer_2.csv` to two independent reviewers.
3. Reviewers use `risk_definitions.csv` and fill `primary_risk`, optional `secondary_risks`, and
   `confidence`. They should not inspect the development codebook first.
4. Run `circ-trace score-codebook-review` as shown in `CODEBOOK_REVIEW.md`.
5. Adjudicate only the rows in `results/codebook_review/adjudication.csv`.
6. Replace the development labels with the adjudicated labels, rerun `make mapping`, and use the
   generated `manuscript_insert.md`, CSV tables, and PDF figure in the revision.

A single reviewer can produce a provisional draft tonight. Two independent reviewers are required
before describing the taxonomy mapping as validated. This analysis characterizes clinician-authored
benchmark risks; it does not measure observed agent failures or clinical effectiveness.