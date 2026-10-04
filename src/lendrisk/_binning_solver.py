"""Adapted OptBinning binary model data and contiguous-interval CP-SAT solver.

Portions adapted from OptBinning 1.0.0, binning/model_data.py and binning/cp.py.
Copyright (C) 2019 Guillermo Navas-Palencia.
Copyright 2026 datakim and lendrisk contributors.
SPDX-License-Identifier: Apache-2.0
See NOTICE and docs/upstream.md for provenance and modifications.
"""

from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

OBJECTIVE_SCALE = 100_000_000


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    events: int
    nonevents: int
    iv: float

    @property
    def count(self) -> int:
        return self.events + self.nonevents


def candidates_from_counts(n_event, n_nonevent, min_count, max_count, min_event, min_nonevent):
    """Specialize upstream's reversed cumulative-count/IV construction.

    Changes: explicit contiguous intervals, early invalid-candidate filtering,
    IV-only objective, and unquantized event counts for monotonic comparisons.
    """
    total_event, total_nonevent = n_event.sum(), n_nonevent.sum()
    candidates = []
    for i in range(1, len(n_event) + 1):
        s_event = n_event[:i][::-1].cumsum()[::-1]
        s_nonevent = n_nonevent[:i][::-1].cumsum()[::-1]
        for j, (event, nonevent) in enumerate(zip(s_event, s_nonevent, strict=True)):
            count = int(event + nonevent)
            if not min_count <= count <= max_count or event < min_event or nonevent < min_nonevent:
                continue
            p, q = float(event / total_event), float(nonevent / total_nonevent)
            candidates.append(
                Candidate(j, i - 1, int(event), int(nonevent), float((p - q) * np.log(p / q)))
            )
    return candidates


def solve_partition(
    candidates, n_prebins, min_bins, max_bins, trend, min_event_rate_diff, time_limit
):
    """Replace triangular assignments with explicit interval indicators.

    Infeasible/unknown solves return no partition, never a fabricated fallback.
    Rate-order comparisons use count cross-products instead of rounded rates.
    """
    model = cp_model.CpModel()
    selected = [model.NewBoolVar(f"interval_{c.start}_{c.end}") for c in candidates]
    for prebin in range(n_prebins):
        model.Add(
            sum(v for c, v in zip(candidates, selected, strict=True) if c.start <= prebin <= c.end)
            == 1
        )
    model.Add(sum(selected) >= min_bins)
    model.Add(sum(selected) <= max_bins)
    by_start = {}
    for index, candidate in enumerate(candidates):
        by_start.setdefault(candidate.start, []).append(index)
    for a_index, a in enumerate(candidates):
        for b_index in by_start.get(a.end + 1, []):
            b = candidates[b_index]
            difference = b.events * a.count - a.events * b.count
            violates = (trend == "ascending" and difference < 0) or (
                trend == "descending" and difference > 0
            )
            if min_event_rate_diff > 0:
                violates |= abs(difference) < min_event_rate_diff * a.count * b.count
            if violates:
                model.Add(selected[a_index] + selected[b_index] <= 1)
    coefficients = [int(round(c.iv * OBJECTIVE_SCALE)) for c in candidates]
    model.Maximize(sum(k * v for k, v in zip(coefficients, selected, strict=True)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = 0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    success = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    chosen = [
        c for c, v in zip(candidates, selected, strict=True) if success and solver.BooleanValue(v)
    ]
    return {
        "status": solver.StatusName(status),
        "chosen": sorted(chosen, key=lambda c: c.start),
        "objective": solver.ObjectiveValue() / OBJECTIVE_SCALE if success else None,
        "bound": solver.BestObjectiveBound() / OBJECTIVE_SCALE if success else np.inf,
        "seconds": solver.WallTime(),
    }
