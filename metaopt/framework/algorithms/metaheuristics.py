from __future__ import annotations
import random
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ..core.problem import BaseProblem, Configuration


@dataclass
class Individual:
    config: Configuration
    fitness: float = -1e9

    def to_dict(self) -> dict:
        return {
            "fitness": self.fitness,
            "config": self.config.to_dict(),
        }


class BaseMetaheuristic:
    NAME = "Base"
    DESCRIPTION = ""
    PARAMS_SCHEMA: dict = {}

    def __init__(self, problem: BaseProblem, max_evals: int = 1000, seed: int = 42, **kwargs):
        self.problem = problem
        self.max_evals = max_evals
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.history: list[float] = []        # best
        self.history_avg: list[float] = []    # avg
        self.best: Individual | None = None
        self._stopped = False
        self._start_time: float = 0.0
        self._elapsed: float = 0.0

    def run(self) -> Individual:
        raise NotImplementedError

    def _eval(self, config: Configuration) -> Individual:
        return Individual(config=config, fitness=self.problem.evaluate(config))

    @classmethod
    def get_params_schema(cls) -> dict:
        return cls.PARAMS_SCHEMA

    def get_info(self) -> dict:
        return {
            "name": self.NAME,
            "description": self.DESCRIPTION,
            "params_schema": self.PARAMS_SCHEMA,
        }

class GeneticAlgorithm(BaseMetaheuristic):
    NAME = "Genetic Algorithm (GA)"
    DESCRIPTION = ("Классический ГА с турнирным отбором, одноточечным кроссовером "
                   "и мутацией замены. Хорошо работает на дискретных задачах.")
    PARAMS_SCHEMA = {
        "pop_size":       {"type": "int",   "default": 30,  "min": 5,   "max": 200, "label": "Размер популяции"},
        "crossover_rate": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "label": "Вероятность кроссовера"},
        "mutation_rate":  {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "label": "Вероятность мутации"},
        "tournament_size":{"type": "int",   "default": 3,   "min": 2,   "max": 10,  "label": "Размер турнира"},
        "elitism":        {"type": "bool",  "default": True,                         "label": "Элитизм"},
    }

    def __init__(self, problem, max_evals=1000, seed=42,
                 pop_size=30, crossover_rate=0.8, mutation_rate=0.2,
                 tournament_size=3, elitism=True, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def run(self) -> Individual:
        self._start_time = time.perf_counter()
        population = [self._eval(self.problem.generate_random()) for _ in range(self.pop_size)]
        self.best = max(population, key=lambda x: x.fitness)
        evals = self.pop_size

        while evals < self.max_evals:
            new_pop = []
            if self.elitism:
                new_pop.append(copy.deepcopy(self.best))

            while len(new_pop) < self.pop_size:
                pa = self._tournament(population)
                pb = self._tournament(population)
                child_cfg = self._crossover(pa.config, pb.config) if random.random() < self.crossover_rate else pa.config.copy()
                if random.random() < self.mutation_rate:
                    child_cfg = self._mutate(child_cfg)
                child = self._eval(child_cfg)
                new_pop.append(child)
                evals += 1
                if child.fitness > self.best.fitness:
                    self.best = copy.deepcopy(child)
                if evals >= self.max_evals:
                    break

            population = new_pop
            avg = sum(x.fitness for x in population if x.fitness > -1e8) / max(1, len(population))
            self.history.append(self.best.fitness)
            self.history_avg.append(avg)

        self._elapsed = time.perf_counter() - self._start_time
        return self.best

    def _tournament(self, pop):
        contenders = random.sample(pop, min(self.tournament_size, len(pop)))
        return max(contenders, key=lambda x: x.fitness)

    def _crossover(self, a: Configuration, b: Configuration) -> Configuration:
        size = len(a.components)
        point = random.randint(1, size - 1)
        child = list(a.components[:point])
        seen = {c.id for c in child}
        for c in b.components:
            if c.id not in seen and len(child) < size:
                child.append(c)
                seen.add(c.id)
        remaining = [c for c in self.problem.components if c.id not in seen]
        while len(child) < size and remaining:
            pick = random.choice(remaining)
            child.append(pick)
            seen.add(pick.id)
            remaining = [c for c in remaining if c.id not in seen]
        return Configuration(components=child)

    def _mutate(self, config: Configuration) -> Configuration:
        config = config.copy()
        current = {c.id for c in config.components}
        outside = [c for c in self.problem.components if c.id not in current]
        if not outside:
            return config
        idx = random.randrange(len(config.components))
        config.components[idx] = random.choice(outside)
        return config



class ParticleSwarmOptimization(BaseMetaheuristic):
    NAME = "Particle Swarm Optimization (PSO)"
    DESCRIPTION = ("Адаптированный бинарный PSO (BPSO). Быстрая начальная сходимость. "
                   "Каждая частица — вектор вероятностей выбора компонентов.")
    PARAMS_SCHEMA = {
        "pop_size": {"type": "int",   "default": 30,  "min": 5,   "max": 200, "label": "Число частиц"},
        "w":        {"type": "float", "default": 0.7, "min": 0.1, "max": 1.0, "label": "Инерция (ω)"},
        "c1":       {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0, "label": "Когнитивный коэф. (c1)"},
        "c2":       {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0, "label": "Социальный коэф. (c2)"},
    }

    def __init__(self, problem, max_evals=1000, seed=42,
                 pop_size=30, w=0.7, c1=1.5, c2=1.5, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def run(self) -> Individual:
        self._start_time = time.perf_counter()
        n = len(self.problem.components)
        size = self.problem.config_size

        pos = np.random.rand(self.pop_size, n)
        vel = np.random.uniform(-1, 1, (self.pop_size, n))

        def decode(p):
            top_k = np.argsort(p)[-size:]
            return Configuration(components=[self.problem.components[i] for i in sorted(top_k)])

        pop = [self._eval(decode(pos[i])) for i in range(self.pop_size)]
        pbest = list(pop)
        pbest_pos = pos.copy()
        self.best = max(pop, key=lambda x: x.fitness)
        gbest_pos = pos[max(range(self.pop_size), key=lambda i: pop[i].fitness)].copy()
        evals = self.pop_size

        while evals < self.max_evals:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(n), np.random.rand(n)
                vel[i] = (self.w * vel[i]
                          + self.c1 * r1 * (pbest_pos[i] - pos[i])
                          + self.c2 * r2 * (gbest_pos - pos[i]))
                pos[i] = np.clip(pos[i] + vel[i], 0, 1)
                ind = self._eval(decode(pos[i]))
                evals += 1
                if ind.fitness > pbest[i].fitness:
                    pbest[i] = ind
                    pbest_pos[i] = pos[i].copy()
                if ind.fitness > self.best.fitness:
                    self.best = copy.deepcopy(ind)
                    gbest_pos = pos[i].copy()
                if evals >= self.max_evals:
                    break
            avg = sum(x.fitness for x in pbest if x.fitness > -1e8) / max(1, len(pbest))
            self.history.append(self.best.fitness)
            self.history_avg.append(avg)

        self._elapsed = time.perf_counter() - self._start_time
        return self.best


class DifferentialEvolution(BaseMetaheuristic):
    NAME = "Differential Evolution (DE)"
    DESCRIPTION = ("DE с непрерывным представлением и декодированием top-k. "
                   "Мощная локальная оптимизация, хорошо на смешанных задачах.")
    PARAMS_SCHEMA = {
        "pop_size": {"type": "int",   "default": 30,  "min": 5,   "max": 200, "label": "Размер популяции"},
        "F":        {"type": "float", "default": 0.8, "min": 0.1, "max": 2.0, "label": "Масштаб мутации (F)"},
        "CR":       {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "label": "Вероятность кроссовера (CR)"},
    }

    def __init__(self, problem, max_evals=1000, seed=42,
                 pop_size=30, F=0.8, CR=0.7, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

    def run(self) -> Individual:
        self._start_time = time.perf_counter()
        n = len(self.problem.components)
        size = self.problem.config_size

        pop_pos = np.random.rand(self.pop_size, n)

        def decode(p):
            top_k = np.argsort(p)[-size:]
            return Configuration(components=[self.problem.components[i] for i in sorted(top_k)])

        pop = [self._eval(decode(pop_pos[i])) for i in range(self.pop_size)]
        self.best = max(pop, key=lambda x: x.fitness)
        evals = self.pop_size

        while evals < self.max_evals:
            new_pos = pop_pos.copy()
            for i in range(self.pop_size):
                idxs = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = random.sample(idxs, 3)
                mutant = np.clip(pop_pos[r1] + self.F * (pop_pos[r2] - pop_pos[r3]), 0, 1)
                cross_mask = np.random.rand(n) < self.CR
                if not cross_mask.any():
                    cross_mask[random.randint(0, n - 1)] = True
                trial = np.where(cross_mask, mutant, pop_pos[i])
                trial_ind = self._eval(decode(trial))
                evals += 1
                if trial_ind.fitness >= pop[i].fitness:
                    new_pos[i] = trial
                    pop[i] = trial_ind
                if trial_ind.fitness > self.best.fitness:
                    self.best = copy.deepcopy(trial_ind)
                if evals >= self.max_evals:
                    break
            pop_pos = new_pos
            avg = sum(x.fitness for x in pop if x.fitness > -1e8) / max(1, len(pop))
            self.history.append(self.best.fitness)
            self.history_avg.append(avg)

        self._elapsed = time.perf_counter() - self._start_time
        return self.best


class AntColonyOptimization(BaseMetaheuristic):
    NAME = "Ant Colony Optimization (ACO)"
    DESCRIPTION = ("Муравьиный алгоритм. Феромоны накапливаются на лучших компонентах. "
                   "Хорошо работает на задачах с выраженной структурой.")
    PARAMS_SCHEMA = {
        "n_ants":  {"type": "int",   "default": 20,  "min": 5,   "max": 100, "label": "Число муравьёв"},
        "alpha":   {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "label": "Вес феромона (α)"},
        "beta":    {"type": "float", "default": 2.0, "min": 0.1, "max": 5.0, "label": "Вес эвристики (β)"},
        "rho":     {"type": "float", "default": 0.1, "min": 0.01,"max": 0.9, "label": "Испарение феромона (ρ)"},
        "q0":      {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Жадность (q0)"},
    }

    def __init__(self, problem, max_evals=1000, seed=42,
                 n_ants=20, alpha=1.0, beta=2.0, rho=0.1, q0=0.5, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0

    def run(self) -> Individual:
        self._start_time = time.perf_counter()
        n = len(self.problem.components)
        size = self.problem.config_size

        pheromones = np.ones(n) * 0.1

        def heuristic(i):
            comp = self.problem.components[i]
            for v in comp.params.values():
                if isinstance(v, (int, float)):
                    return max(0.01, float(v))
            return 1.0

        eta = np.array([heuristic(i) for i in range(n)])
        eta = eta / (eta.max() + 1e-9)

        self.best = None
        evals = 0

        while evals < self.max_evals:
            generation_best = None
            delta_pheromones = np.zeros(n)

            for _ in range(self.n_ants):
                chosen = []
                available = list(range(n))
                for _ in range(size):
                    if not available:
                        break
                    tau = pheromones[available] ** self.alpha
                    eta_vals = eta[available] ** self.beta
                    probs = tau * eta_vals
                    total = probs.sum()
                    if total < 1e-12:
                        probs = np.ones(len(available))
                        total = probs.sum()
                    probs /= total

                    if random.random() < self.q0:
                        idx = available[np.argmax(probs)]
                    else:
                        idx = np.random.choice(available, p=probs)
                    chosen.append(idx)
                    available.remove(idx)

                config = Configuration(components=[self.problem.components[i] for i in sorted(chosen)])
                ind = self._eval(config)
                evals += 1

                if generation_best is None or ind.fitness > generation_best.fitness:
                    generation_best = ind
                if self.best is None or ind.fitness > self.best.fitness:
                    self.best = copy.deepcopy(ind)

                if ind.fitness > -1e8:
                    for idx in chosen:
                        delta_pheromones[idx] += max(0, ind.fitness)

                if evals >= self.max_evals:
                    break

            pheromones = (1 - self.rho) * pheromones + self.rho * delta_pheromones
            pheromones = np.clip(pheromones, 0.01, 10.0)

            self.history.append(self.best.fitness)
            self.history_avg.append(generation_best.fitness if generation_best else self.best.fitness)

        self._elapsed = time.perf_counter() - self._start_time
        return self.best



class SimulatedAnnealing(BaseMetaheuristic):
    NAME = "Simulated Annealing (SA)"
    DESCRIPTION = ("Траекторный метод. Принимает ухудшающие решения с вероятностью "
                   "exp(-ΔF/T). Хорош для задач с острыми локальными оптимумами.")
    PARAMS_SCHEMA = {
        "T_init":     {"type": "float", "default": 1.0,  "min": 0.01, "max": 100.0, "label": "Начальная температура"},
        "T_min":      {"type": "float", "default": 1e-4, "min": 1e-6, "max": 0.1,   "label": "Конечная температура"},
        "cooling":    {"type": "float", "default": 0.995,"min": 0.9,  "max": 0.9999,"label": "Коэффициент охлаждения"},
        "n_neighbors":{"type": "int",   "default": 5,    "min": 1,    "max": 20,    "label": "Соседей за итерацию"},
    }

    def __init__(self, problem, max_evals=1000, seed=42,
                 T_init=1.0, T_min=1e-4, cooling=0.995, n_neighbors=5, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.T_init = T_init
        self.T_min = T_min
        self.cooling = cooling
        self.n_neighbors = n_neighbors

    def run(self) -> Individual:
        self._start_time = time.perf_counter()
        current = self._eval(self.problem.generate_random())
        self.best = copy.deepcopy(current)
        T = self.T_init
        evals = 1

        while evals < self.max_evals and T > self.T_min:
            iter_best = current
            for _ in range(self.n_neighbors):
                neighbor = self._eval(self._neighbor(current.config))
                evals += 1
                delta = neighbor.fitness - current.fitness
                if delta > 0 or (T > 1e-12 and random.random() < math.exp(delta / T)):
                    current = neighbor
                if current.fitness > self.best.fitness:
                    self.best = copy.deepcopy(current)
                if neighbor.fitness > iter_best.fitness:
                    iter_best = neighbor
                if evals >= self.max_evals:
                    break
            T *= self.cooling
            self.history.append(self.best.fitness)
            self.history_avg.append(iter_best.fitness)

        self._elapsed = time.perf_counter() - self._start_time
        return self.best

    def _neighbor(self, config: Configuration) -> Configuration:
        config = config.copy()
        current_ids = {c.id for c in config.components}
        outside = [c for c in self.problem.components if c.id not in current_ids]
        if not outside or not config.components:
            return config
        idx = random.randrange(len(config.components))
        config.components[idx] = random.choice(outside)
        return config

class HybridDEGA(BaseMetaheuristic):
\
    NAME = "Hybrid DE + GA"
    DESCRIPTION = ("Последовательная гибридизация: DE исследует глобально, "
                   "GA-кроссовер уточняет лучшие регионы. +20-30% vs отдельные методы.")
    PARAMS_SCHEMA = {
        "pop_size":       {"type": "int",   "default": 30,  "min": 5,   "max": 200, "label": "Размер популяции"},
        "F":              {"type": "float", "default": 0.8, "min": 0.1, "max": 2.0, "label": "Масштаб мутации DE (F)"},
        "CR":             {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "label": "Кроссовер DE (CR)"},
        "ga_fraction":    {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "label": "Доля GA-операторов"},
        "mutation_rate":  {"type": "float", "default": 0.15,"min": 0.0, "max": 1.0, "label": "Мутация GA"},
    }

    def __init__(self, problem, max_evals=1000, seed=42,
                 pop_size=30, F=0.8, CR=0.7, ga_fraction=0.3, mutation_rate=0.15, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.ga_fraction = ga_fraction
        self.mutation_rate = mutation_rate

    def run(self) -> Individual:
        self._start_time = time.perf_counter()
        n = len(self.problem.components)
        size = self.problem.config_size

        pop_pos = np.random.rand(self.pop_size, n)

        def decode(p):
            top_k = np.argsort(p)[-size:]
            return Configuration(components=[self.problem.components[i] for i in sorted(top_k)])

        pop = [self._eval(decode(pop_pos[i])) for i in range(self.pop_size)]
        self.best = max(pop, key=lambda x: x.fitness)
        evals = self.pop_size

        while evals < self.max_evals:
            new_pos = pop_pos.copy()
            for i in range(self.pop_size):
                if random.random() > self.ga_fraction:
                    idxs = [j for j in range(self.pop_size) if j != i]
                    r1, r2, r3 = random.sample(idxs, 3)
                    mutant = np.clip(pop_pos[r1] + self.F * (pop_pos[r2] - pop_pos[r3]), 0, 1)
                    cross_mask = np.random.rand(n) < self.CR
                    if not cross_mask.any():
                        cross_mask[random.randint(0, n-1)] = True
                    trial_pos = np.where(cross_mask, mutant, pop_pos[i])
                else:
                    j = random.choice([k for k in range(self.pop_size) if k != i])
                    child_cfg = self._ga_crossover(decode(pop_pos[i]), decode(pop_pos[j]))
                    if random.random() < self.mutation_rate:
                        child_cfg = self._ga_mutate(child_cfg)
                    child_ind = self._eval(child_cfg)
                    evals += 1
                    if child_ind.fitness >= pop[i].fitness:
                        new_pos[i] = (pop_pos[i] + pop_pos[j]) / 2
                        pop[i] = child_ind
                    if child_ind.fitness > self.best.fitness:
                        self.best = copy.deepcopy(child_ind)
                    if evals >= self.max_evals:
                        break
                    continue

                trial_ind = self._eval(decode(trial_pos))
                evals += 1
                if trial_ind.fitness >= pop[i].fitness:
                    new_pos[i] = trial_pos
                    pop[i] = trial_ind
                if trial_ind.fitness > self.best.fitness:
                    self.best = copy.deepcopy(trial_ind)
                if evals >= self.max_evals:
                    break

            pop_pos = new_pos
            avg = sum(x.fitness for x in pop if x.fitness > -1e8) / max(1, len(pop))
            self.history.append(self.best.fitness)
            self.history_avg.append(avg)

        self._elapsed = time.perf_counter() - self._start_time
        return self.best

    def _ga_crossover(self, a: Configuration, b: Configuration) -> Configuration:
        size = len(a.components)
        point = random.randint(1, size - 1)
        child = list(a.components[:point])
        seen = {c.id for c in child}
        for c in b.components:
            if c.id not in seen and len(child) < size:
                child.append(c)
                seen.add(c.id)
        remaining = [c for c in self.problem.components if c.id not in seen]
        while len(child) < size and remaining:
            pick = random.choice(remaining)
            child.append(pick)
            seen.add(pick.id)
            remaining = [c for c in remaining if c.id not in seen]
        return Configuration(components=child)

    def _ga_mutate(self, config: Configuration) -> Configuration:
        config = config.copy()
        current = {c.id for c in config.components}
        outside = [c for c in self.problem.components if c.id not in current]
        if not outside:
            return config
        idx = random.randrange(len(config.components))
        config.components[idx] = random.choice(outside)
        return config


ALGORITHM_REGISTRY: dict[str, type] = {
    "GA":       GeneticAlgorithm,
    "PSO":      ParticleSwarmOptimization,
    "DE":       DifferentialEvolution,
    "ACO":      AntColonyOptimization,
    "SA":       SimulatedAnnealing,
    "HybridDEGA": HybridDEGA,
}


def get_algorithm(name: str) -> type:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Алгоритм '{name}' не найден. Доступные: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]


def list_algorithms() -> list[dict]:
    return [
        {
            "id": k,
            "name": cls.NAME,
            "description": cls.DESCRIPTION,
            "params_schema": cls.PARAMS_SCHEMA,
        }
        for k, cls in ALGORITHM_REGISTRY.items()
    ]
