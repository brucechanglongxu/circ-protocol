.PHONY: test lint verify assay mapping

CLINICARE_REPO ?= ../../../clinicare

test:
	uv run --extra dev pytest -q

lint:
	uv run --extra dev ruff check src tests

verify:
	uv run circ-trace --clinicare-repo $(CLINICARE_REPO) verify-public-source

assay:
	uv run circ-trace --clinicare-repo $(CLINICARE_REPO) run-public-assay \
		--out results/public_synthetic_assay.json

mapping:
	uv run --extra analysis circ-trace --clinicare-repo $(CLINICARE_REPO) \
		run-public-mapping --out-dir results/public_mapping