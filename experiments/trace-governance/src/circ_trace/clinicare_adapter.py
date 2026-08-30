"""Normalize public CliniCARE artifacts without retaining clinical text."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from .schema import ClinicalScope, EvidenceEvent, TrialArtifact

_SCENARIO_NUMBER = re.compile(r"^scenario-(\d+)-")
_TASK_NUMBER = re.compile(r"^Task\s+(\d+)\b")
_FROM_OR_JOIN = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?)?\b"
)
_HYPHENATED_ID = re.compile(r"\b\d{6,}-(?:[A-Z]{1,4}-)?\d+\b")
_INTEGER_ID = re.compile(r"\b\d{6,}\b")


def _literal(node: ast.AST):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and not node.args
        and not node.keywords
    ):
        return set()
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def _assignments(path: Path, names: set[str]) -> dict[str, object]:
    found: dict[str, object] = {}
    for node in ast.parse(path.read_text()).body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id in names and value is not None:
            found[target.id] = _literal(value)
    missing = names - set(found)
    if missing:
        raise ValueError(f"missing CliniCARE scoring assignments in {path}: {sorted(missing)}")
    return found


class CliniCAREPolicy:
    """Pinned table-attribution policy loaded from CliniCARE's own scorer source."""

    def __init__(self, clinicare_repo: Path) -> None:
        metric = (
            clinicare_repo
            / "benchmark"
            / "eval"
            / "metrics"
            / "grounding"
            / "evidence_retrieval_recall.py"
        )
        raw = _assignments(metric, {"TOOL_TABLES", "GOLD_VOCAB", "_DROP", "_CANON"})
        self.tool_tables = {
            str(tool): tuple(str(table) for table in tables)
            for tool, tables in dict(raw["TOOL_TABLES"]).items()
        }
        self.drop = {str(table) for table in set(raw["_DROP"])}
        self.canonical = {
            str(table): {str(target) for target in targets}
            for table, targets in dict(raw["_CANON"]).items()
        }

        all_tables = {str(table) for table in list(raw["GOLD_VOCAB"])}
        all_tables.update(table for tables in self.tool_tables.values() for table in tables)
        all_tables.update(self.canonical)
        all_tables.update(target for targets in self.canonical.values() for target in targets)
        self.recognized_tables = all_tables - self.drop

    def canon(self, tables: set[str]) -> frozenset[str]:
        canonical: set[str] = set()
        for table in tables:
            normalized = table.lower()
            if normalized in self.drop:
                continue
            canonical.update(self.canonical.get(normalized, {normalized}))
        return frozenset(canonical)

    def tables_for(self, tool: str, args: dict[str, object]) -> frozenset[str]:
        if tool == "run_sql":
            query = str(args.get("query") or args.get("sql") or "")
            raw = {
                match.group(1).lower()
                for match in _FROM_OR_JOIN.finditer(query)
                if match.group(1).lower() in self.recognized_tables
            }
        else:
            raw = set(self.tool_tables.get(tool, ()))
        return self.canon(raw)


def load_required_tables(clinicare_repo: Path, policy: CliniCAREPolicy) -> dict[str, frozenset[str]]:
    scenarios_dir = clinicare_repo / "benchmark" / "_scenarios"
    scenario_by_number: dict[int, str] = {}
    for scenario_dir in scenarios_dir.glob("scenario-*"):
        match = _SCENARIO_NUMBER.match(scenario_dir.name)
        if match is None:
            continue
        import tomllib

        scenario_id = tomllib.loads((scenario_dir / "scenario.toml").read_text())["id"]
        scenario_by_number[int(match.group(1))] = str(scenario_id)

    gold_path = clinicare_repo / "benchmark" / "eval" / "gold" / "gold_sources.json"
    required: dict[str, frozenset[str]] = {}
    for item in json.loads(gold_path.read_text()):
        match = _TASK_NUMBER.match(str(item["title"]))
        if match is None or int(match.group(1)) not in scenario_by_number:
            raise ValueError(f"cannot join CliniCARE gold-source row: {item['title']}")
        required[scenario_by_number[int(match.group(1))]] = policy.canon(set(item["tables"]))
    if len(required) != len(scenario_by_number):
        raise ValueError(
            f"CliniCARE source join covered {len(required)}/{len(scenario_by_number)} scenarios"
        )
    return required


def extract_report_record_ids(report: str) -> tuple[str, ...]:
    """Use CliniCARE's report-side identifier grammar while excluding timestamps."""
    text = _TIMESTAMP.sub(" ", report)
    identifiers = {match.group() for match in _HYPHENATED_ID.finditer(text)}
    text = _HYPHENATED_ID.sub(" ", text)
    identifiers.update(match.group() for match in _INTEGER_ID.finditer(text))
    return tuple(sorted(identifiers))


def load_trial_artifact(
    *,
    artifact_id: str,
    cluster_id: str | None = None,
    scenario_id: str,
    scope: ClinicalScope,
    verdict: str,
    evidence_path: Path,
    report_path: Path,
    clinicare_repo: Path,
) -> TrialArtifact:
    """Load a CliniCARE trial while retaining only replay-relevant metadata."""
    policy = CliniCAREPolicy(clinicare_repo)
    required = load_required_tables(clinicare_repo, policy)
    if scenario_id not in required:
        raise ValueError(f"unknown CliniCARE scenario: {scenario_id}")

    events: list[EvidenceEvent] = []
    for index, line in enumerate(evidence_path.read_text().splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        events.append(
            EvidenceEvent(
                event_id=str(raw.get("call_id") or raw.get("step") or index),
                kind=str(raw.get("kind") or "mimic"),
                tool=str(raw.get("tool") or ""),
                args=dict(args),
                status=str(raw.get("status") or "error"),
                n_rows=raw.get("n_rows") if isinstance(raw.get("n_rows"), int) else None,
                record_ids=tuple(str(value) for value in raw.get("record_ids") or ()),
                tables=policy.tables_for(str(raw.get("tool") or ""), dict(args)),
            )
        )

    return TrialArtifact(
        artifact_id=artifact_id,
        cluster_id=cluster_id or artifact_id,
        scenario_id=scenario_id,
        scope=scope,
        verdict=verdict.upper(),
        events=tuple(events),
        cited_record_ids=extract_report_record_ids(report_path.read_text()),
        required_tables=required[scenario_id],
    )