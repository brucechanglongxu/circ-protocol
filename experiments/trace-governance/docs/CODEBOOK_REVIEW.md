# Risk Codebook Review Protocol

The current 104-row mapping is a complete first pass, not a validated annotation standard.

## Review procedure

1. Two reviewers independently receive the nine family definitions, inclusion rules, exclusions, and
   the 104 criterion texts in randomized order without current labels or target family counts.
2. Each reviewer assigns one primary family, zero or more secondary families, and confidence
   (`high`, `moderate`, or `low`). A reviewer may nominate a new family rather than force a fit.
3. Report raw agreement, Cohen's kappa for primary labels, and Krippendorff's alpha as a sensitivity
   analysis. Report family-wise confusion rather than only one aggregate coefficient.
4. Resolve disagreements by discussion with a third adjudicator. Preserve pre-adjudication labels.
5. Version any changed family definition or criterion mapping. Recompute the source and codebook
   digests before protected-trace analysis.

Generate the packets with `make mapping`. After both reviewers complete their respective files:

```bash
uv run circ-trace score-codebook-review \
   --reviewer-1 results/public_mapping/reviewer_1.csv \
   --reviewer-2 results/public_mapping/reviewer_2.csv \
   --out-dir results/codebook_review
```

This writes `agreement.json` and a blank-final-label `adjudication.csv` containing only disagreements.

## Priority review set

Nine rows are marked `boundary` because their primary-family assignment has a plausible alternative:

- `advanced-imaging/N5`
- `pe-workup/N2`
- `discharge-audit/N1`
- `discharge-audit/N2`
- `sepsis-bundle/N4`
- `empiric-abx/N1`
- `postop-complications/N2`
- `frequent-admitter/N2`
- `ams-workup/N2`

These rows receive review first, but reviewers remain blinded to that designation during independent
coding.

## Interpretation

Criterion counts measure the design of CliniCARE's process rubric. They do not measure observed
failure prevalence, severity, preventability, or patient harm. Those require executed traces,
criterion applicability, and separate outcome evidence.