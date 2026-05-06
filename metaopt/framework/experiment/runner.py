from __future__ import annotations
import csv
import time
import json
import statistics
import uuid
from dataclasses import dataclass, field
from typing import Type, Optional
from ..core.problem import BaseProblem
from ..algorithms.metaheuristics import BaseMetaheuristic, Individual

@dataclass
class RunResult:

    run_id: int
    seed: int
    algorithm: str
    problem: str
    best_fitness: float
    elapsed_sec: float
    n_evals: int
    convergence: list[float] = field(default_factory=list)
    convergence_avg: list[float] = field(default_factory=list)
    best_config_ids: list[str] = field(default_factory=list)
    objective_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "problem": self.problem,
            "best_fitness": self.best_fitness,
            "elapsed_sec": self.elapsed_sec,
            "n_evals": self.n_evals,
            "convergence": self.convergence,
            "convergence_avg": self.convergence_avg,
            "best_config_ids": self.best_config_ids,
            "objective_vector": self.objective_vector,
        }

@dataclass
class ExperimentSummary:

    algorithm: str
    problem: str
    n_runs: int
    best: float
    worst: float
    mean: float
    median: float
    std: float
    mean_time: float

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "problem": self.problem,
            "n_runs": self.n_runs,
            "best": round(self.best, 6),
            "worst": round(self.worst, 6),
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "std": round(self.std, 6),
            "mean_time": round(self.mean_time, 4),
        }

class ExperimentRunner:

    def __init__(self):
        self._problems: dict[str, BaseProblem] = {}
        self._algorithms: list[tuple[str, Type[BaseMetaheuristic], dict]] = []
        self._results: list[RunResult] = []
        self.experiment_id: str = str(uuid.uuid4())[:8]
        self._progress_callback = None

    def add_problem(self, name: str, problem: BaseProblem):
        self._problems[name] = problem

    def add_algorithm(self, name: str, cls: Type[BaseMetaheuristic], params: dict = None):
        self._algorithms.append((name, cls, params or {}))

    def set_progress_callback(self, fn):
        self._progress_callback = fn

    def run(self, n_runs: int = 5, max_evals: int = 500, base_seed: int = 0) -> list[RunResult]:

        total = len(self._problems) * len(self._algorithms) * n_runs
        done = 0

        for prob_name, problem in self._problems.items():
            for alg_name, AlgClass, params in self._algorithms:
                for run_id in range(n_runs):
                    seed = base_seed + run_id * 100
                    problem.reset_eval_count()
                    alg = AlgClass(problem=problem, max_evals=max_evals, seed=seed, **params)

                    t0 = time.perf_counter()
                    best: Individual = alg.run()
                    elapsed = time.perf_counter() - t0

                    obj_vec = problem.evaluate_vector(best.config) if best.fitness > -1e8 else []

                    result = RunResult(
                        run_id=run_id,
                        seed=seed,
                        algorithm=alg_name,
                        problem=prob_name,
                        best_fitness=best.fitness,
                        elapsed_sec=round(elapsed, 4),
                        n_evals=max_evals,
                        convergence=alg.history,
                        convergence_avg=alg.history_avg,
                        best_config_ids=[c.id for c in best.config.components],
                        objective_vector=obj_vec,
                    )
                    self._results.append(result)
                    done += 1

                    if self._progress_callback:
                        self._progress_callback({
                            "done": done, "total": total,
                            "algorithm": alg_name, "problem": prob_name,
                            "run_id": run_id, "fitness": best.fitness,
                            "elapsed": elapsed,
                        })

        return self._results

    def get_summary(self) -> list[ExperimentSummary]:
        keys = {(r.algorithm, r.problem) for r in self._results}
        summaries = []
        for alg, prob in sorted(keys):
            vals = [r.best_fitness for r in self._results if r.algorithm == alg and r.problem == prob]
            times = [r.elapsed_sec for r in self._results if r.algorithm == alg and r.problem == prob]
            valid = [v for v in vals if v > -1e8]
            if not valid:
                valid = vals
            summaries.append(ExperimentSummary(
                algorithm=alg, problem=prob, n_runs=len(vals),
                best=max(valid), worst=min(valid), mean=statistics.mean(valid),
                median=statistics.median(valid),
                std=statistics.stdev(valid) if len(valid) > 1 else 0.0,
                mean_time=statistics.mean(times),
            ))
        return summaries

    def export_csv(self, path: str):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm","problem","run_id","seed","best_fitness",
                             "elapsed_sec","n_evals","best_config"])
            for r in self._results:
                writer.writerow([r.algorithm, r.problem, r.run_id, r.seed,
                                 r.best_fitness, r.elapsed_sec, r.n_evals,
                                 "|".join(r.best_config_ids)])

    def export_json(self, path: str):
        data = {
            "experiment_id": self.experiment_id,
            "results": [r.to_dict() for r in self._results],
            "summary": [s.to_dict() for s in self.get_summary()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"{'Алгоритм':<20} {'Задача':<20} {'Лучшее':>10} {'Среднее':>10} {'Худшее':>10} {'StdDev':>10}")
        print("=" * 70)
        for s in self.get_summary():
            print(f"{s.algorithm:<20} {s.problem:<20} {s.best:>10.4f} {s.mean:>10.4f} {s.worst:>10.4f} {s.std:>10.4f}")
        print("=" * 70)