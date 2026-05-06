from __future__ import annotations
from typing import Optional


def generate_explanation(
    team: list[dict],           # best_config_detail из эксперимента
    shapley_values: list[dict], # результат compute_shapley
    objective_names: list[str], # названия критериев
    weights: list[float],       # нормализованные веса
    fitness: float,             # итоговый fitness
    algorithm: str,             # название алгоритма
    objective_vector: list[float] = None,  # значения критериев
    problem_name: str = "",
) -> dict:

    n = len(team)
    shap_map = {s["comp_id"]: s for s in shapley_values}

    def get_name(c):
        for key in ("hero_name", "char_name", "name", "title"):
            if c.get("params", {}).get(key):
                return c["params"][key]
        return c.get("id", "?")

    team_names = [get_name(c) for c in team]

    key_facts = []

    key_facts.append(f"Итоговый fitness: {fitness:.4f}")

    if objective_vector and objective_names:
        for i, (name, val) in enumerate(zip(objective_names, objective_vector)):
            w_pct = round((weights[i] if i < len(weights) else 0) * 100)
            key_facts.append(f"{name} (вес {w_pct}%): {val:.3f}")

    shap_sorted = sorted(shapley_values, key=lambda x: x["shapley"], reverse=True)
    positive_contribs = [s for s in shap_sorted if s["shapley"] > 0]
    negative_contribs = [s for s in shap_sorted if s["shapley"] < 0]
    zero_contribs     = [s for s in shap_sorted if s["shapley"] == 0]

    top_contributor = shap_sorted[0] if shap_sorted else None
    bottom_contributor = shap_sorted[-1] if len(shap_sorted) > 1 else None


    all_param_keys = set()
    for c in team:
        for k, v in c.get("params", {}).items():
            if isinstance(v, (int, float)):
                all_param_keys.add(k)

    numeric_params = {}
    for k in all_param_keys:
        vals = [c["params"][k] for c in team if isinstance(c.get("params", {}).get(k), (int, float))]
        if vals:
            numeric_params[k] = {
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "vals": vals
            }

    top_params = sorted(numeric_params.items(), key=lambda x: x[1]["avg"], reverse=True)[:3]

    diverse_param = max(numeric_params.items(),
        key=lambda x: x[1]["max"] - x[1]["min"]) if numeric_params else None

    all_tags = []
    for c in team:
        all_tags.extend(c.get("tags", []))
    tag_counts = {}
    for t in all_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1

    role_diversity = len(set(all_tags)) / max(len(all_tags), 1)

    sections = []

    problem_str = f' для задачи «{problem_name}»' if problem_name else ''
    intro = (
        f"Алгоритм {algorithm} нашёл оптимальный состав{problem_str} "
        f"из {n} участников с итоговым fitness {fitness:.4f}. "
    )
    if objective_vector and objective_names:
        best_obj_idx = weights.index(max(weights)) if weights else 0
        best_obj_val = objective_vector[best_obj_idx] if best_obj_idx < len(objective_vector) else None
        if best_obj_val is not None:
            intro += (
                f"Приоритетный критерий «{objective_names[best_obj_idx]}» "
                f"(вес {round(weights[best_obj_idx]*100)}%) достиг значения {best_obj_val:.3f}."
            )
    sections.append({"title": "Результат", "text": intro})

    comp_lines = []
    for c in team:
        name = get_name(c)
        tags = c.get("tags", [])
        shap = shap_map.get(c.get("id", ""), {})
        phi = shap.get("shapley")

        num_params = [(k, v) for k, v in c.get("params", {}).items()
                     if isinstance(v, (int, float)) and k not in ("hero_name","char_name","name","title")]
        top_c_params = sorted(num_params, key=lambda x: x[1], reverse=True)[:3]
        params_str = ", ".join(f"{k}={v}" for k, v in top_c_params)

        role_str = f" [{'/'.join(tags)}]" if tags else ""
        phi_str = f", φ={phi:+.3f}" if phi is not None else ""
        comp_lines.append(f"• {name}{role_str}: {params_str}{phi_str}")

    sections.append({"title": "Состав конфигурации", "text": "\n".join(comp_lines)})

    if shap_sorted:
        shap_text_parts = []

        if top_contributor:
            tc_name = top_contributor["name"]
            tc_phi  = top_contributor["shapley"]
            tc_pct  = top_contributor["shapley_pct"]
            shap_text_parts.append(
                f"Наибольший вклад вносит {tc_name} (φ={tc_phi:+.3f}, {tc_pct:.1f}% от суммарного влияния). "
                f"Это означает что в среднем при любом порядке формирования команды {tc_name} "
                f"увеличивает её качество на {tc_phi:.3f}."
            )

        if len(positive_contribs) >= 2:
            runner_up = positive_contribs[1]
            shap_text_parts.append(
                f"На втором месте — {runner_up['name']} (φ={runner_up['shapley']:+.3f}), "
                f"обеспечивающий стабильный положительный вклад."
            )

        if negative_contribs:
            neg_names = ", ".join(s["name"] for s in negative_contribs)
            avg_neg = sum(s["shapley"] for s in negative_contribs) / len(negative_contribs)
            shap_text_parts.append(
                f"Обратите внимание: {neg_names} имеет отрицательное значение Шепли "
                f"(φ≈{avg_neg:.3f}). Это может указывать на антагонизм с другими участниками "
                f"или слабость по ключевым критериям. Однако наличие этого участника "
                f"может быть обусловлено выполнением ограничений задачи."
            )

        if zero_contribs:
            zero_names = ", ".join(s["name"] for s in zero_contribs)
            shap_text_parts.append(
                f"{zero_names} имеет нулевой вклад Шепли - "
                f"участник нейтрален и не влияет на синергии команды."
            )

        sections.append({"title": "Анализ вкладов (значения Шепли)", "text": " ".join(shap_text_parts)})

    synergy_text_parts = []

    if tag_counts:
        dominant_tag = max(tag_counts, key=tag_counts.get)
        dominant_count = tag_counts[dominant_tag]
        if dominant_count == n:
            synergy_text_parts.append(
                f"Все {n} участников имеют тег «{dominant_tag}» — "
                f"команда однородна по этому признаку."
            )
        elif dominant_count > n // 2:
            synergy_text_parts.append(
                f"Большинство участников ({dominant_count} из {n}) имеют тег «{dominant_tag}»."
            )

        unique_tags = len(tag_counts)
        if unique_tags > n:
            synergy_text_parts.append(
                f"Команда демонстрирует высокое разнообразие: {unique_tags} уникальных тегов "
                f"на {n} участников, что положительно влияет на критерий разнообразия."
            )

    if diverse_param:
        k, stats = diverse_param
        synergy_text_parts.append(
            f"Наибольший разброс по параметру «{k}»: от {stats['min']:.0f} до {stats['max']:.0f} — "
            f"команда сочетает участников с разным уровнем этой характеристики."
        )

    if top_params:
        best_k, best_stats = top_params[0]
        synergy_text_parts.append(
            f"Сильнейшая сторона команды — «{best_k}» "
            f"(среднее {best_stats['avg']:.1f})."
        )

    if synergy_text_parts:
        sections.append({"title": "Структура и разнообразие", "text": " ".join(synergy_text_parts)})

    confidence = "высокая" if len(positive_contribs) == n else \
                 "средняя" if len(positive_contribs) >= n // 2 else "низкая"

    rec_parts = [
        f"Данная конфигурация рекомендована с уверенностью «{confidence}». "
    ]

    if len(positive_contribs) == n:
        rec_parts.append(
            f"Все {n} участников вносят положительный вклад — "
            f"команда сбалансирована и устойчива."
        )
    elif negative_contribs:
        rec_parts.append(
            f"При возможности рассмотрите замену {negative_contribs[-1]['name']} "
            f"на участника с более высокими показателями по ключевым критериям."
        )

    if fitness > 0:
        rec_parts.append(
            f"Для улучшения результата попробуйте увеличить max_evals "
            f"или добавить синергии между участниками в матрицу взаимодействий."
        )

    sections.append({"title": "Рекомендация", "text": " ".join(rec_parts)})

    top_name = top_contributor["name"] if top_contributor else team_names[0]
    summary = (
        f"Оптимальная команда из {n} участников (fitness={fitness:.4f}): "
        f"{', '.join(team_names)}. "
        f"Ключевой вкладчик — {top_name}."
    )

    return {
        "summary":   summary,
        "sections":  sections,
        "key_facts": key_facts,
        "confidence": confidence,
        "top_contributor": top_contributor["name"] if top_contributor else None,
        "has_negative": len(negative_contribs) > 0,
    }