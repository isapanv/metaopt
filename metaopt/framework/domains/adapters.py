from __future__ import annotations
from ..core.problem import BaseProblem, Component, Configuration, InteractionMatrix

def create_team_building_problem(config_size: int = 5) -> BaseProblem:

    raw = [
        {"id": "alice",  "role": "teamlead", "skill": 9, "cost": 120, "seniority": "senior", "lang": "Python"},
        {"id": "bob",    "role": "backend",  "skill": 8, "cost": 100, "seniority": "senior", "lang": "Java"},
        {"id": "carol",  "role": "frontend", "skill": 7, "cost": 90,  "seniority": "middle", "lang": "React"},
        {"id": "dave",   "role": "backend",  "skill": 6, "cost": 80,  "seniority": "middle", "lang": "Python"},
        {"id": "eve",    "role": "devops",   "skill": 8, "cost": 110, "seniority": "senior", "lang": "Go"},
        {"id": "frank",  "role": "frontend", "skill": 5, "cost": 70,  "seniority": "junior", "lang": "Vue"},
        {"id": "grace",  "role": "qa",       "skill": 7, "cost": 85,  "seniority": "middle", "lang": "Python"},
        {"id": "heidi",  "role": "backend",  "skill": 9, "cost": 115, "seniority": "senior", "lang": "Go"},
        {"id": "ivan",   "role": "teamlead", "skill": 8, "cost": 105, "seniority": "senior", "lang": "Java"},
        {"id": "judy",   "role": "qa",       "skill": 6, "cost": 75,  "seniority": "middle", "lang": "Python"},
        {"id": "karl",   "role": "devops",   "skill": 7, "cost": 95,  "seniority": "middle", "lang": "Bash"},
        {"id": "laura",  "role": "frontend", "skill": 8, "cost": 100, "seniority": "senior", "lang": "React"},
    ]
    components = [Component(id=m["id"], params={k: v for k, v in m.items() if k != "id"}, tags=[m["role"]]) for m in raw]
    matrix = InteractionMatrix(components)
    for a, b, v in [
        ("alice", "eve",   2.0), ("alice", "heidi", 1.5), ("bob", "carol", 1.2),
        ("heidi","laura",  1.2), ("eve",   "karl",  1.8), ("grace","judy", 1.5),
        ("alice","frank", -1.0), ("ivan",  "alice", -2.0), ("bob", "heidi", -0.5),
    ]:
        matrix.set(a, b, v)

    budget = max(400, config_size * 95)

    def budget_ok(cfg): return sum(c.params["cost"] for c in cfg.components) <= budget
    def has_lead(cfg): return any(c.params["role"] == "teamlead" for c in cfg.components)
    def skill_score(cfg): return sum(c.params["skill"] for c in cfg.components) / len(cfg.components)
    def synergy_score(cfg): return matrix.synergy_score(cfg)
    def diversity_score(cfg): return float(len({c.params["role"] for c in cfg.components}))

    return BaseProblem(
        name="IT Team Building",
        description=f"Выбрать команду из {config_size} специалистов. Бюджет ≤ {budget}. Нужен тимлид.",
        components=components,
        interaction_matrix=matrix,
        constraints=[budget_ok, has_lead],
        constraint_names=[f"Бюджет ≤ {budget}", "Наличие тимлида"],
        objectives=[skill_score, synergy_score, diversity_score],
        objective_names=["Средний skill", "Синергия", "Разнообразие ролей"],
        weights=[0.5, 0.3, 0.2],
        config_size=config_size,
    )

def create_microservices_problem(config_size: int = 6) -> BaseProblem:

    raw = [
        {"id": "api-gateway",    "type": "gateway",  "throughput": 10, "reliability": 0.99, "cost": 50,  "cores": 4, "memory": 8},
        {"id": "auth-service",   "type": "auth",     "throughput": 8,  "reliability": 0.98, "cost": 40,  "cores": 2, "memory": 4},
        {"id": "user-service",   "type": "crud",     "throughput": 7,  "reliability": 0.97, "cost": 35,  "cores": 2, "memory": 4},
        {"id": "order-service",  "type": "business", "throughput": 9,  "reliability": 0.98, "cost": 55,  "cores": 4, "memory": 8},
        {"id": "payment-svc",    "type": "payment",  "throughput": 6,  "reliability": 0.999,"cost": 80,  "cores": 4, "memory": 8},
        {"id": "notification",   "type": "async",    "throughput": 5,  "reliability": 0.95, "cost": 25,  "cores": 2, "memory": 2},
        {"id": "cache-redis",    "type": "cache",    "throughput": 10, "reliability": 0.99, "cost": 30,  "cores": 2, "memory": 16},
        {"id": "db-postgres",    "type": "database", "throughput": 8,  "reliability": 0.999,"cost": 70,  "cores": 4, "memory": 16},
        {"id": "db-mongo",       "type": "database", "throughput": 7,  "reliability": 0.99, "cost": 60,  "cores": 4, "memory": 8},
        {"id": "search-elastic", "type": "search",   "throughput": 8,  "reliability": 0.97, "cost": 65,  "cores": 4, "memory": 16},
        {"id": "monitoring",     "type": "ops",      "throughput": 3,  "reliability": 0.99, "cost": 20,  "cores": 2, "memory": 4},
        {"id": "ml-recommender", "type": "ml",       "throughput": 4,  "reliability": 0.95, "cost": 90,  "cores": 8, "memory": 16},
    ]
    components = [Component(id=m["id"], params={k:v for k,v in m.items() if k!="id"}, tags=[m["type"]]) for m in raw]
    matrix = InteractionMatrix(components)
    for a, b, v in [
        ("api-gateway","auth-service",  1.5), ("api-gateway","cache-redis", 1.8),
        ("order-service","db-postgres", 1.5), ("order-service","payment-svc", 1.2),
        ("cache-redis","db-postgres",   2.0), ("user-service","db-postgres", 1.0),
        ("search-elastic","db-mongo",   1.5), ("db-postgres","db-mongo",    -1.5),
        ("ml-recommender","cache-redis",1.2), ("monitoring","db-postgres",  0.5),
    ]:
        matrix.set(a, b, v)

    max_cores = 32

    def cores_ok(cfg): return sum(c.params["cores"] for c in cfg.components) <= max_cores
    def has_gateway(cfg): return any(c.params["type"] == "gateway" for c in cfg.components)
    def throughput_score(cfg): return sum(c.params["throughput"] for c in cfg.components) / len(cfg.components)
    def reliability_score(cfg):
        r = 1.0
        for c in cfg.components: r *= c.params["reliability"]
        return r * 10
    def cost_efficiency(cfg):
        total_cost = sum(c.params["cost"] for c in cfg.components)
        return max(0.0, 10.0 - total_cost / 40.0)

    return BaseProblem(
        name="Cloud Microservices",
        description=f"Выбрать {config_size} микросервисов. Ядра ≤ {max_cores}. Обязателен API-шлюз.",
        components=components,
        interaction_matrix=matrix,
        constraints=[cores_ok, has_gateway],
        constraint_names=[f"CPU-ядра ≤ {max_cores}", "Наличие API-шлюза"],
        objectives=[throughput_score, reliability_score, cost_efficiency],
        objective_names=["Пропускная способность", "Надёжность", "Экономичность"],
        weights=[0.4, 0.4, 0.2],
        config_size=config_size,
    )

def create_production_shift_problem(config_size: int = 4) -> BaseProblem:

    raw = [
        {"id": "master-a",   "role": "master",    "productivity": 9, "quality": 9, "safety": 8, "wage": 400, "experience": 15},
        {"id": "master-b",   "role": "master",    "productivity": 8, "quality": 8, "safety": 9, "wage": 380, "experience": 12},
        {"id": "operator-1", "role": "operator",  "productivity": 7, "quality": 7, "safety": 7, "wage": 280, "experience": 5},
        {"id": "operator-2", "role": "operator",  "productivity": 8, "quality": 6, "safety": 6, "wage": 290, "experience": 7},
        {"id": "operator-3", "role": "operator",  "productivity": 6, "quality": 8, "safety": 8, "wage": 275, "experience": 4},
        {"id": "technician", "role": "technician","productivity": 6, "quality": 9, "safety": 7, "wage": 320, "experience": 8},
        {"id": "inspector",  "role": "inspector", "productivity": 4, "quality": 10,"safety": 9, "wage": 350, "experience": 10},
        {"id": "loader-1",   "role": "loader",    "productivity": 8, "quality": 5, "safety": 6, "wage": 230, "experience": 2},
        {"id": "loader-2",   "role": "loader",    "productivity": 7, "quality": 5, "safety": 7, "wage": 235, "experience": 3},
        {"id": "electrician","role": "electrician","productivity":5, "quality": 8, "safety": 9, "wage": 340, "experience": 9},
    ]
    components = [Component(id=m["id"], params={k:v for k,v in m.items() if k!="id"}, tags=[m["role"]]) for m in raw]
    matrix = InteractionMatrix(components)
    for a, b, v in [
        ("master-a","technician",   1.8), ("master-a","inspector",  1.5),
        ("master-b","electrician",  1.5), ("inspector","operator-3",1.2),
        ("technician","electrician",1.5), ("master-a","master-b",  -2.0),
        ("loader-1","loader-2",    -0.5), ("operator-1","operator-2",0.8),
    ]:
        matrix.set(a, b, v)

    wage_limit = 1500

    def wage_ok(cfg): return sum(c.params["wage"] for c in cfg.components) <= wage_limit
    def has_master(cfg): return any(c.params["role"] == "master" for c in cfg.components)
    def productivity(cfg): return sum(c.params["productivity"] for c in cfg.components) / len(cfg.components)
    def quality(cfg): return sum(c.params["quality"] for c in cfg.components) / len(cfg.components)
    def safety(cfg): return sum(c.params["safety"] for c in cfg.components) / len(cfg.components)

    return BaseProblem(
        name="Production Shift",
        description=f"Сформировать смену из {config_size} работников. Зарплата ≤ {wage_limit}. Нужен мастер.",
        components=components,
        interaction_matrix=matrix,
        constraints=[wage_ok, has_master],
        constraint_names=[f"Зарплата ≤ {wage_limit}", "Наличие мастера"],
        objectives=[productivity, quality, safety],
        objective_names=["Производительность", "Качество", "Безопасность"],
        weights=[0.4, 0.35, 0.25],
        config_size=config_size,
    )

def create_electronic_system_problem(config_size: int = 5) -> BaseProblem:

    raw = [
        {"id": "cpu-a53",   "type": "cpu",     "perf": 6, "power": 3.0, "cost": 15, "arch": "ARM",  "freq": 1.4},
        {"id": "cpu-a72",   "type": "cpu",     "perf": 9, "power": 6.0, "cost": 35, "arch": "ARM",  "freq": 1.8},
        {"id": "cpu-x86",   "type": "cpu",     "perf": 10,"power": 15.0,"cost": 60, "arch": "x86",  "freq": 3.2},
        {"id": "ram-4gb",   "type": "memory",  "perf": 6, "power": 1.5, "cost": 20, "arch": "DDR4", "freq": 2400},
        {"id": "ram-8gb",   "type": "memory",  "perf": 8, "power": 2.5, "cost": 35, "arch": "DDR4", "freq": 3200},
        {"id": "gpu-mali",  "type": "gpu",     "perf": 6, "power": 3.0, "cost": 20, "arch": "ARM",  "freq": 800},
        {"id": "gpu-nvidia","type": "gpu",     "perf": 10,"power": 10.0,"cost": 80, "arch": "CUDA", "freq": 1200},
        {"id": "ssd-128",   "type": "storage", "perf": 7, "power": 1.5, "cost": 25, "arch": "NVMe", "freq": 0},
        {"id": "wifi-module","type": "network", "perf": 5, "power": 0.5, "cost": 10, "arch": "BCM",  "freq": 2400},
        {"id": "eth-module", "type": "network", "perf": 8, "power": 1.0, "cost": 15, "arch": "RJ45", "freq": 1000},
        {"id": "pmic",      "type": "power",   "perf": 4, "power": 0.5, "cost": 8,  "arch": "TI",   "freq": 0},
    ]
    components = [Component(id=m["id"], params={k:v for k,v in m.items() if k!="id"}, tags=[m["type"]]) for m in raw]
    matrix = InteractionMatrix(components)
    for a, b, v in [
        ("cpu-a72","ram-8gb",   1.5), ("cpu-x86","ram-8gb",    1.8),
        ("gpu-nvidia","ram-8gb",1.5), ("cpu-a53","wifi-module", 0.8),
        ("cpu-a72","gpu-mali",  1.2), ("cpu-x86","gpu-nvidia",  2.0),
        ("cpu-a53","cpu-a72",  -1.5), ("cpu-a72","cpu-x86",    -2.0),
        ("pmic","cpu-a53",     1.0),  ("ssd-128","eth-module",  0.8),
    ]:
        matrix.set(a, b, v)

    max_power = 25.0
    max_cost = 200

    def power_ok(cfg): return sum(c.params["power"] for c in cfg.components) <= max_power
    def cost_ok(cfg): return sum(c.params["cost"] for c in cfg.components) <= max_cost
    def has_cpu(cfg): return any(c.params["type"] == "cpu" for c in cfg.components)
    def perf_score(cfg): return sum(c.params["perf"] for c in cfg.components) / len(cfg.components)
    def energy_eff(cfg):
        total_perf = sum(c.params["perf"] for c in cfg.components)
        total_power = sum(c.params["power"] for c in cfg.components) + 0.1
        return total_perf / total_power
    def cost_eff(cfg):
        total_cost = sum(c.params["cost"] for c in cfg.components)
        return max(0.0, 10.0 - total_cost / 20.0)

    return BaseProblem(
        name="Electronic System",
        description=f"Выбрать {config_size} компонентов. Мощность ≤ {max_power}Вт, стоимость ≤ {max_cost}$.",
        components=components,
        interaction_matrix=matrix,
        constraints=[power_ok, cost_ok, has_cpu],
        constraint_names=[f"Мощность ≤ {max_power}Вт", f"Стоимость ≤ {max_cost}$", "Наличие CPU"],
        objectives=[perf_score, energy_eff, cost_eff],
        objective_names=["Производительность", "Энергоэффективность", "Экономичность"],
        weights=[0.4, 0.35, 0.25],
        config_size=config_size,
    )

DOMAIN_REGISTRY = {
    "team_building": {
        "name": "IT Team Building",
        "description": "Подбор команды разработчиков для IT-проекта с учётом бюджета и синергий",
        "factory": create_team_building_problem,
        "default_config_size": 5,
        "min_config_size": 2,
        "max_config_size": 8,
        "icon": "👥",
    },
    "microservices": {
        "name": "Cloud Microservices",
        "description": "Выбор набора микросервисов для облачного приложения с ограничением ресурсов",
        "factory": create_microservices_problem,
        "default_config_size": 6,
        "min_config_size": 3,
        "max_config_size": 9,
        "icon": "☁️",
    },
    "production_shift": {
        "name": "Production Shift",
        "description": "Формирование производственной смены с учётом компетенций и бюджета зарплат",
        "factory": create_production_shift_problem,
        "default_config_size": 4,
        "min_config_size": 2,
        "max_config_size": 7,
        "icon": "🏭",
    },
    "electronic_system": {
        "name": "Electronic System",
        "description": "Конфигурирование встраиваемой электронной системы по критериям мощности и стоимости",
        "factory": create_electronic_system_problem,
        "default_config_size": 5,
        "min_config_size": 3,
        "max_config_size": 8,
        "icon": "💻",
    },
}

def create_problem(domain_id: str, config_size: int = None) -> BaseProblem:
    if domain_id not in DOMAIN_REGISTRY:
        raise ValueError(f"Домен '{domain_id}' не найден. Доступные: {list(DOMAIN_REGISTRY.keys())}")
    info = DOMAIN_REGISTRY[domain_id]
    size = config_size or info["default_config_size"]
    return info["factory"](config_size=size)