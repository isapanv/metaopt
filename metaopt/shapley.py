from __future__ import annotations
import math
import random
import itertools
from framework.core.problem import BaseProblem, Configuration

def compute_shapley(
    problem: BaseProblem,
    selected_ids: list[str],
    method: str = "auto",
    n_samples: int = 512,
    seed: int = 42,
) -> list[dict]:

    rng = random.Random(seed)

    comp_map = {c.id: c for c in problem.components}
    players = [comp_map[cid] for cid in selected_ids if cid in comp_map]
    n = len(players)

    if n == 0:
        return []

    if method == "auto":
        method = "exact" if n <= 12 else "approx"

    def v(subset: list) -> float:
        if not subset:
            return 0.0
        cfg = Configuration(components=subset)
        if not problem.is_feasible(cfg):
            return 0.0
        score = problem.evaluate(cfg)
        return max(0.0, score)

    shapley_vals = {c.id: 0.0 for c in players}

    if method == "exact":

        for i, player in enumerate(players):
            others = [c for c in players if c.id != player.id]
            phi = 0.0
            for r in range(len(others) + 1):
                weight = (math.factorial(r) * math.factorial(n - r - 1)
                          / math.factorial(n))
                for subset in itertools.combinations(others, r):
                    subset_list = list(subset)
                    phi += weight * (v(subset_list + [player]) - v(subset_list))
            shapley_vals[player.id] = phi

    else:

        counts = {c.id: 0.0 for c in players}
        for _ in range(n_samples):
            perm = players[:]
            rng.shuffle(perm)
            current_coalition = []
            current_val = 0.0
            for player in perm:
                new_coalition = current_coalition + [player]
                new_val = v(new_coalition)
                counts[player.id] += new_val - current_val
                current_coalition = new_coalition
                current_val = new_val
        for cid in shapley_vals:
            shapley_vals[cid] = counts[cid] / n_samples

    total_abs = sum(abs(v) for v in shapley_vals.values()) or 1.0

    result = []
    for comp in players:
        phi = shapley_vals[comp.id]

        display_name = (
            comp.params.get("hero_name")
            or comp.params.get("name")
            or comp.params.get("title")
            or comp.id
        )
        result.append({
            "comp_id":     comp.id,
            "name":        display_name,
            "params":      comp.params,
            "tags":        comp.tags,
            "shapley":     round(phi, 4),
            "shapley_pct": round(phi / total_abs * 100, 1),
        })

    result.sort(key=lambda x: x["shapley"], reverse=True)
    for i, r in enumerate(result):
        r["rank"] = i + 1

    return result