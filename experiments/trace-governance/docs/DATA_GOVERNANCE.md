# Data Governance

CliniCARE reports and trajectories can quote MIMIC-IV records. Task identifiers are also sensitive
because they are unsalted hashes of patient-linked parameters. This repository therefore treats the
entire protected run directory as governed data.

## Repository boundary

Allowed to commit:

- CliniCARE's public MIT-licensed scenarios and derived public metadata through an external pinned
  checkout;
- the CIRC taxonomy and criterion codebook;
- synthetic no-PHI fixtures;
- source code, tests, preregistration, and disclosure-reviewed aggregate tables.

Never commit:

- cohort bundles (`full750.csv` or `dev148.csv`);
- MIMIC tables, converted Parquet files, or DuckDB databases;
- raw reports, trajectories, evidence text, task IDs, or patient identifiers;
- per-case verdict tables or mutation manifests tied to protected task IDs.

## Analysis boundary

Protected runs execute inside the PhysioNet-authorized environment. The adapter retains only the
metadata needed for replay: tool name, scoped arguments, call health, row count, record identifiers,
and attributed table names. Clinical text is dropped immediately after normalization.

Before export, case IDs are replaced with salted study IDs and outputs are aggregated. A second person
must review every exported artifact for small cells, free text, task IDs, and record identifiers. The
salt and case-level crosswalk remain outside the repository.

## Operational controls

- Keep CliniCARE's credential-isolated sidecar and network restrictions unchanged across arms.
- Do not pass labels, reference specifications, rubric answers, or fault manifests to controls.
- Log source commit, source digest, codebook digest, environment lock, and analysis version.
- Treat malformed or incomplete traces as infrastructure outcomes rather than clinical failures.

Set `CIRC_STUDY_SALT` directly in the protected terminal. Do not send it through chat or commit it.
The protected command accepts one or more `SYSTEM=JOB_PATH` inputs and writes aggregate results:

```bash
uv run circ-trace --clinicare-repo ../clinicare run-protected-replay \
  --cohort-csv "$CLINICARE_COHORT_DIR/full750.csv" \
  --job codex-gpt55=/protected/jobs/codex-gpt55 \
  --job claude-opus5=/protected/jobs/claude-opus5 \
  --out results/protected_replay_aggregate.json
```