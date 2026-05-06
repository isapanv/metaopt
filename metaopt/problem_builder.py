from __future__ import annotations
from framework.core.problem import BaseProblem, Component, Configuration, InteractionMatrix
import database as db

AGG_FUNCTIONS = {
    "avg":  lambda vals: sum(vals) / len(vals) if vals else 0.0,
    "sum":  lambda vals: sum(vals),
    "min":  lambda vals: min(vals) if vals else 0.0,
    "max":  lambda vals: max(vals) if vals else 0.0,
}

OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "<":  lambda a, b: a  < b,
    ">":  lambda a, b: a  > b,
}

def build_problem_from_db(problem_id: int) -> BaseProblem:

    prob_data = db.get_custom_problem(problem_id)
    if not prob_data:
        raise ValueError(f"Задача #{problem_id} не найдена в БД")

    catalog_id = prob_data["catalog_id"]

    raw_components = db.get_components(catalog_id, active_only=True)
    if not raw_components:
        raise ValueError("В каталоге нет активных компонентов")

    components = [
        Component(
            id=c["comp_id"],
            params=c["params"],
            tags=c["tags"],
        )
        for c in raw_components
    ]

    matrix = InteractionMatrix(components)
    for syn in db.get_synergies(catalog_id):
        try:
            matrix.set(syn["comp_id_a"], syn["comp_id_b"], float(syn["value"]))
        except KeyError:
            pass

    constraints = []
    constraint_names = []
    for c in prob_data["constraints"]:
        fn = _build_constraint(c)
        if fn:
            constraints.append(fn)
            constraint_names.append(c.get("label", f"{c['param']} {c['op']} {c['value']}"))

    objectives = []
    objective_names = []
    weights = []

    for obj in prob_data["objectives"]:
        fn = _build_objective(obj, matrix)
        if fn:
            objectives.append(fn)
            objective_names.append(obj.get("label", obj["param"]))
            weights.append(float(obj.get("weight", 1.0)))

    if not objectives:
        raise ValueError("Не задано ни одного критерия оптимизации")

    total_w = sum(weights)
    weights = [w / total_w for w in weights] if total_w > 0 else weights

    return BaseProblem(
        name=prob_data["name"],
        description=prob_data["description"],
        components=components,
        interaction_matrix=matrix,
        constraints=constraints,
        constraint_names=constraint_names,
        objectives=objectives,
        objective_names=objective_names,
        weights=weights,
        config_size=prob_data["config_size"],
    )

def _build_constraint(c: dict):

    ctype = c.get("type", "param_agg")

    if ctype == "param_agg":
        param = c["param"]
        op_str = c.get("op", "<=")
        limit = float(c["value"])
        agg_fn = AGG_FUNCTIONS.get(c.get("agg", "sum"), AGG_FUNCTIONS["sum"])
        op_fn = OPERATORS.get(op_str, OPERATORS["<="])

        def constraint_fn(cfg: Configuration, _param=param, _agg=agg_fn, _op=op_fn, _lim=limit):
            vals = [c.params.get(_param, 0) for c in cfg.components if isinstance(c.params.get(_param), (int, float))]
            return _op(_agg(vals), _lim) if vals else True
        return constraint_fn

    elif ctype == "tag_required":

        tag = c.get("param") or c.get("value") or ""
        def constraint_fn(cfg: Configuration, _tag=tag):
            return any(_tag in comp.tags for comp in cfg.components)
        return constraint_fn

    elif ctype == "tag_forbidden":
        tag = c.get("param") or c.get("value") or ""
        def constraint_fn(cfg: Configuration, _tag=tag):
            return not any(_tag in comp.tags for comp in cfg.components)
        return constraint_fn

    elif ctype == "count_tag":
        tag = c["param"]
        op_str = c.get("op", ">=")
        limit = int(c["value"])
        op_fn = OPERATORS.get(op_str, OPERATORS[">="])
        def constraint_fn(cfg: Configuration, _tag=tag, _op=op_fn, _lim=limit):
            count = sum(1 for comp in cfg.components if _tag in comp.tags)
            return _op(count, _lim)
        return constraint_fn

    return None

def _build_objective(obj: dict, matrix: InteractionMatrix):

    param = obj.get("param", "")
    agg_str = obj.get("agg", "avg")
    agg_fn = AGG_FUNCTIONS.get(agg_str, AGG_FUNCTIONS["avg"])

    if param == "__synergy__":
        def synergy_fn(cfg: Configuration, _m=matrix):
            return _m.synergy_score(cfg)
        return synergy_fn

    if param == "__diversity__":
        tag_key = obj.get("tag_key", "role")
        def diversity_fn(cfg: Configuration, _key=tag_key):
            vals = {c.params.get(_key) for c in cfg.components if c.params.get(_key)}
            return float(len(vals))
        return diversity_fn

    def objective_fn(cfg: Configuration, _param=param, _agg=agg_fn):
        vals = [c.params.get(_param, 0) for c in cfg.components if isinstance(c.params.get(_param), (int, float))]
        return _agg(vals) if vals else 0.0
    return objective_fn