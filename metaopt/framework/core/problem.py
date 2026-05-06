from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import random
import numpy as np

@dataclass
class Component:

    id: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Component) and self.id == other.id

    def __repr__(self):
        return f"Component({self.id})"

    def to_dict(self) -> dict:
        return {"id": self.id, "params": self.params, "tags": self.tags}

class InteractionMatrix:

    def __init__(self, components: list[Component]):
        self.components = components
        self._index = {c.id: i for i, c in enumerate(components)}
        n = len(components)
        self._matrix = np.zeros((n, n))

    def set(self, id_a: str, id_b: str, value: float):
        i, j = self._index[id_a], self._index[id_b]
        self._matrix[i][j] = value
        self._matrix[j][i] = value

    def get(self, id_a: str, id_b: str) -> float:
        if id_a not in self._index or id_b not in self._index:
            return 0.0
        i, j = self._index[id_a], self._index[id_b]
        return float(self._matrix[i][j])

    def synergy_score(self, config: "Configuration") -> float:

        ids = [c.id for c in config.components]
        total = 0.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += self.get(ids[i], ids[j])
        return total

    def to_dict(self) -> dict:

        ids = [c.id for c in self.components]
        data = {}
        for i, id_a in enumerate(ids):
            for j, id_b in enumerate(ids):
                if i < j:
                    val = float(self._matrix[i][j])
                    if val != 0.0:
                        data[f"{id_a}:{id_b}"] = val
        return {"ids": ids, "interactions": data}

@dataclass
class Configuration:

    components: list[Component] = field(default_factory=list)

    def copy(self) -> "Configuration":
        return Configuration(components=list(self.components))

    def __repr__(self):
        ids = [c.id for c in self.components]
        return f"Configuration({ids})"

    def to_dict(self) -> dict:
        return {"components": [c.to_dict() for c in self.components],
                "ids": [c.id for c in self.components]}

class BaseProblem:

    def __init__(
        self,
        name: str = "Unnamed Problem",
        description: str = "",
        components: list[Component] = None,
        interaction_matrix: Optional[InteractionMatrix] = None,
        constraints: list[Callable[[Configuration], bool]] = None,
        constraint_names: list[str] = None,
        objectives: list[Callable[[Configuration], float]] = None,
        objective_names: list[str] = None,
        weights: list[float] = None,
        config_size: int = None,
    ):
        self.name = name
        self.description = description
        self.components = components or []
        self.interaction_matrix = interaction_matrix or InteractionMatrix(self.components)
        self.constraints = constraints or []
        self.constraint_names = constraint_names or [f"C{i}" for i in range(len(self.constraints))]
        self.objectives = objectives or []
        self.objective_names = objective_names or [f"f{i}" for i in range(len(self.objectives))]
        self.weights = weights or [1.0 / max(1, len(self.objectives))] * len(self.objectives)
        self.config_size = config_size or max(1, len(self.components) // 2)
        self._eval_count = 0

    def is_feasible(self, config: Configuration) -> bool:
        return all(c(config) for c in self.constraints)

    def evaluate(self, config: Configuration) -> float:

        self._eval_count += 1
        if not self.is_feasible(config):
            return -1e9
        scores = [f(config) for f in self.objectives]
        return sum(w * s for w, s in zip(self.weights, scores))

    def evaluate_vector(self, config: Configuration) -> list[float]:

        return [f(config) for f in self.objectives]

    def generate_random(self) -> Configuration:

        for _ in range(2000):
            sample = random.sample(self.components, min(self.config_size, len(self.components)))
            config = Configuration(components=sample)
            if self.is_feasible(config):
                return config
        return Configuration(components=random.sample(
            self.components, min(self.config_size, len(self.components))
        ))

    def reset_eval_count(self):
        self._eval_count = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_components": len(self.components),
            "config_size": self.config_size,
            "n_constraints": len(self.constraints),
            "constraint_names": self.constraint_names,
            "n_objectives": len(self.objectives),
            "objective_names": self.objective_names,
            "weights": self.weights,
            "components": [c.to_dict() for c in self.components],
            "interaction_matrix": self.interaction_matrix.to_dict(),
        }