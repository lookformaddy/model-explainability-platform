"""Explanation registry: versioned, immutable explanation records.

Every record is tied to a model version and to a hash of the exact dataset
snapshot it was computed on, so an explanation can never silently drift
away from the data it describes. Records are frozen dataclasses; the
registry refuses to overwrite an existing (model_version, dataset_hash,
explanation_kind) key, and returns copies on read.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def dataset_snapshot_hash(rows: List[List[float]], feature_names: List[str]) -> str:
    """Deterministic SHA-256 over the dataset snapshot + feature names."""
    payload = json.dumps(
        {"features": feature_names, "rows": rows},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExplanationRecord:
    """One immutable explanation artifact."""

    model_version: str
    dataset_hash: str
    explanation_kind: str  # e.g. "permutation_importance", "pdp", "local"
    explanation_version: int
    payload: Tuple[Tuple[str, str], ...]  # JSON-serialisable (key, json) pairs
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def payload_dict(self) -> Dict[str, object]:
        return {k: json.loads(v) for k, v in self.payload}


def freeze_payload(payload: Dict[str, object]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in payload.items()))


class ExplanationRegistry:
    """Append-only store of versioned explanation records."""

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str, str, int], ExplanationRecord] = {}

    def register(
        self,
        model_version: str,
        dataset_hash: str,
        explanation_kind: str,
        payload: Dict[str, object],
    ) -> ExplanationRecord:
        """Store a new version; raises if that exact version already exists."""
        existing = [
            v
            for (mv, dh, kind, v) in self._records
            if mv == model_version and dh == dataset_hash and kind == explanation_kind
        ]
        version = max(existing, default=0) + 1
        record = ExplanationRecord(
            model_version=model_version,
            dataset_hash=dataset_hash,
            explanation_kind=explanation_kind,
            explanation_version=version,
            payload=freeze_payload(payload),
        )
        key = (model_version, dataset_hash, explanation_kind, version)
        if key in self._records:  # pragma: no cover - guarded by version bump
            raise ValueError(f"Explanation record already exists: {key}")
        self._records[key] = record
        return deepcopy(record)

    def latest(
        self, model_version: str, dataset_hash: str, explanation_kind: str
    ) -> Optional[ExplanationRecord]:
        matches = [
            rec
            for (mv, dh, kind, _), rec in self._records.items()
            if mv == model_version and dh == dataset_hash and kind == explanation_kind
        ]
        if not matches:
            return None
        return deepcopy(max(matches, key=lambda r: r.explanation_version))

    def history(
        self, model_version: str, dataset_hash: str, explanation_kind: str
    ) -> List[ExplanationRecord]:
        matches = [
            rec
            for (mv, dh, kind, _), rec in self._records.items()
            if mv == model_version and dh == dataset_hash and kind == explanation_kind
        ]
        return deepcopy(sorted(matches, key=lambda r: r.explanation_version))

    def all_records(self) -> List[ExplanationRecord]:
        return deepcopy(sorted(self._records.values(), key=lambda r: r.created_at))
