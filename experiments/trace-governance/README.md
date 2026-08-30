# CIRC Trace Governance

Trace-based experiments for evaluating independent governance controls for clinical AI agents.
The study uses the public scenario specifications and evaluation interfaces from
[CliniCARE-Bench](https://github.com/scaleapi/clinicare), pinned to an exact upstream revision.

This project does **not** treat CIRC as a maturity model and does not claim that CliniCARE is a
multi-agent benchmark. It separates two evidence layers:

1. A public, patient-free study of the 104 clinician-authored prohibited process shortcuts across
   all 25 CliniCARE scenarios.
2. A matched trace-replay study of independent governance controls. Synthetic fixtures run in CI;
   real CliniCARE traces remain inside a PhysioNet-authorized environment and contribute only
   disclosure-reviewed aggregate results.

## Release boundary

CliniCARE's code, scenario specifications, and scoring pipeline are MIT-licensed and public. Its
MIMIC-IV cohort, case labels, reports, and trajectories are not public artifacts. They require
PhysioNet credentials, the separately distributed cohort bundle, and a local MIMIC backend. This
repository never vendors those artifacts.

## Current status

- Frozen upstream source: `scaleapi/clinicare@bbce780bc0dab07aae584f4dbc097601916b0905`
- Public process inventory: 252 criteria, including 104 MUST-NOT criteria
- Draft risk codebook: complete one-primary-label mapping for all 104 MUST-NOT criteria
- Next executable stage: matched clean/faulted trace replay under a factorial set of governance
  controls

## Setup

```bash
git clone https://github.com/scaleapi/clinicare.git /path/to/clinicare
git -C /path/to/clinicare checkout bbce780bc0dab07aae584f4dbc097601916b0905

python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
CLINICARE_REPO=/path/to/clinicare pytest -q
```

The Makefile defaults to a sibling checkout at `../../../clinicare`. Override `CLINICARE_REPO`
when the upstream repository is elsewhere.

```bash
make verify
make assay
```

`make assay` runs 25 base traces, four prespecified fault types, and eight factorial control
arms, producing 1,000 aggregate outcomes. This is an instrument check with known answers. The
confirmatory study uses protected traces under the protocol in
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).

The codebook is a preregistration artifact, not an empirical estimate of failure prevalence.
Counts describe the released rubric design. Failure rates require executed traces and
applicability denominators.