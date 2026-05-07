# MetaOpt — Техническая документация

## Содержание

1. [Архитектура](#1-архитектура)
2. [Формальная модель](#2-формальная-модель)
3. [Слой задач](#3-слой-задач)
4. [Слой алгоритмов](#4-слой-алгоритмов)
5. [Слой экспериментов](#5-слой-экспериментов)
6. [Значения Шепли](#6-значения-шепли)
7. [Схема базы данных](#7-схема-базы-данных)
8. [REST API](#8-rest-api)
9. [Фронтенд](#9-фронтенд)
10. [Встроенные домены](#10-встроенные-домены)
11. [Расширение фреймворка](#11-расширение-фреймворка)

---

## 1. Архитектура

MetaOpt следует трёхслойной архитектуре с инверсией зависимостей:

```
┌────────────────────────────────────────────────────┐
│              Веб-интерфейс (SPA)                   │
│              HTTP + Server-Sent Events             │
├────────────────────────────────────────────────────┤
│              Flask REST API (app.py)               │
├──────────────────────┬─────────────────────────────┤
│   Слой экспериментов │    Шепли / Объяснитель      │
│   (runner.py)        │    (shapley.py)             │
├──────────────────────┴─────────────────────────────┤
│              Слой алгоритмов                       │
│   GA  ·  DE  ·  PSO  ·  ACO  ·  SA  ·  HybridDEGA  │
├────────────────────────────────────────────────────┤
│                Слой задач                          │
│   Component · Configuration · BaseProblem · I      │
├────────────────────────────────────────────────────┤
│       Слой данных (database.py + PostgreSQL)       │
│               (problem_builder.py)                 │
└────────────────────────────────────────────────────┘
```

**Инверсия зависимостей:**  Алгоритмы взаимодействуют с задачами только через абстрактный интерфейс `BaseProblem`. Слой экспериментов работает с абстрактным интерфейсом алгоритмов.

---

## 2. Формальная модель

Задача оптимизации конфигурации моделируется как:

$$\Omega = \langle \mathcal{A},\, \Theta,\, I,\, \mathcal{E},\, \mathcal{G},\, \mathcal{F} \rangle$$

| Символ | Тип | Описание |
|--------|-----|----------|
| $\mathcal{A} = \{a_1, \ldots, a_m\}$ | множество | Доступные компоненты (агенты) |
| $\Theta = \Theta_1 \times \cdots \times \Theta_m$ | пространство | Пространство параметров компонентов (дискретное / непрерывное / гибридное) |
| $I = [I_{ij}]_{m \times m}$ | матрица | Попарные значения взаимодействий; $I_{ij} > 0$ — синергия, $I_{ij} < 0$ — антагонизм |
| $\mathcal{E}$ | множество | Параметры среды |
| $\mathcal{G} = \{g_1(S) \leq 0, \ldots\}$ | ограничения | Условия допустимости |
| $\mathcal{F}: \mathcal{C} \times \mathcal{E} \to \mathbb{R}^p$ | функция | Многокритериальная оценка |

**Задача оптимизации:**

$$S^* = \underset{S \subseteq \mathcal{A}}{\arg\max} \sum_{l=1}^{p} w_l \cdot f_l(S,\, \theta,\, I) \quad \text{при условии} \quad |S| = k, \quad g_j(S) \leq 0 \; \forall j$$

**Декомпозиция функции приспособленности** (формула 16):

$$f_l(S) = \underbrace{\sum_{i \in S} w_i f_{l,i}(a_i)}_{\text{индивидуальный}} + \underbrace{\sum_{i,k \in S,\, i<k} I_{ik} \cdot g_{l,ik}(a_i, a_k)}_{\text{попарная синергия}} + \text{члены высших порядков}$$

---

## 3. Слой задач

**Расположение:** `framework/core/problem.py`

### 3.1 Component

Представляет единичный элемент системы $a_i \in \mathcal{A}$.

```python
@dataclass
class Component:
    id: str                        # Уникальный идентификатор
    params: dict[str, Any]         # Числовые и категориальные параметры
    tags: list[str]                # Категориальные метки для ограничений
```

### 3.2 Configuration

Кандидат на решение — подмножество компонентов размером $k$.

```python
@dataclass
class Configuration:
    components: list[Component]

    def __len__(self) -> int: ...
    def __contains__(self, comp: Component) -> bool: ...
```

### 3.3 InteractionMatrix

Разреженное представление $I = [I_{ij}]$.

```python
class InteractionMatrix:
    def __init__(self, synergies: dict[tuple[str,str], float]): ...
    def get(self, id_a: str, id_b: str) -> float: ...
    def total_synergy(self, config: Configuration) -> float: ...
```

### 3.4 BaseProblem

Абстрактный интерфейс задачи.

```python
class BaseProblem:
    components: list[Component]
    config_size: int
    weights: list[float]           
    objectives: list[Callable]
    constraints: list[Callable]
    objective_names: list[str]
    constraint_names: list[str]

    def is_feasible(self, config: Configuration) -> bool:
        """Возвращает True тогда и только тогда, когда все ограничения выполнены."""

    def evaluate(self, config: Configuration) -> float:
        """Возвращает Σ wₗ·fₗ(S) или -1e9, если конфигурация недопустима."""

    def evaluate_vector(self, config: Configuration) -> list[float]:
        """Возвращает [f₁(S), ..., fₚ(S)] без агрегации."""

    def generate_random(self) -> Configuration:
        """Генерирует случайную допустимую конфигурацию (до 2000 попыток)."""
```

---

## 4. Слой алгоритмов

**Расположение:** `framework/algorithms/metaheuristics.py`

### 4.1 Общий интерфейс

Все алгоритмы наследуются от `BaseMetaheuristic`:

```python
class BaseMetaheuristic:
    def __init__(self, problem: BaseProblem, max_evals: int, seed: int, **kwargs): ...

    def run(self) -> AlgorithmResult: ...
```

Поля `AlgorithmResult`:
- `config: Configuration` — лучшая найденная конфигурация
- `fitness: float` — лучшее значение приспособленности
- `convergence: list[float]` — значение приспособленности на каждом шаге оценки
- `eval_count: int` — общее количество использованных оценок
- `elapsed_sec: float` — астрономическое время выполнения

### 4.2 Генетический алгоритм (GA)

```
Параметры:
  pop_size      Размер популяции (по умолчанию: 30)
  cx_prob       Вероятность скрещивания (по умолчанию: 0.8)
  mut_prob      Вероятность мутации (по умолчанию: 0.2)
  tournament_k  Размер турнира (по умолчанию: 3)
  elite_n       Число выбранных особей (по умолчанию: 1)

Кодирование: каждая особь — фиксированный сет идентификаторов компонентов размером k.
Селекция: турнирная — выбирает tournament_k случайных особей и возвращает лучшую.
Скрещивание: одноточечный кроссовер на отсортированных списках компонентов, восстановливает до размера k.
Мутация: замена одного случайного компонента на не входящий в конфигурацию.
```

### 4.3 Дифференциальная эволюция (DE)

```
Стратегия: DE/rand/1/bin

Параметры:
  F   Масштабный коэффициент мутации (по умолчанию: 0.8)
  CR  Коэффициент скрещивания (по умолчанию: 0.7)
  NP  Размер популяции (по умолчанию: 30)

Дискретная адаптация:
  Непрерывные векторы проецируются в конфигурации:
  компоненты сортируются по значению вектора, выбираются top-k с проверкой допустимости.
```

### 4.4 Бинарный PSO (PSO)

```
Параметры:
  n_particles  Размер роя (по умолчанию: 30)
  w            Вес инерции (по умолчанию: 0.7)
  c1           Когнитивный коэффициент (по умолчанию: 1.5)
  c2           Социальный коэффициент (по умолчанию: 1.5)

Обновление скорости (по каждому измерению компонента):
  v[i] = w·v[i] + c1·r1·(pbest[i]−x[i]) + c2·r2·(gbest[i]−x[i])
  x[i] = 1 если rand() < sigmoid(v[i]) иначе 0

Восстановление: если |S| ≠ k, случайно добавляются или убраются компоненты.
```

### 4.5 Муравьиный алгоритм (ACO)

```
Параметры:
  n_ants    Количество муравьёв (по умолчанию: 20)
  alpha     Вес феромона (по умолчанию: 1.0)
  beta      Вес эвристики (по умолчанию: 2.0)
  rho       Скорость испарения (по умолчанию: 0.1)
  q0        Вероятность эксплуатации (по умолчанию: 0.5)

Вероятность выбора:
  p(i,j) = [τ(j)]^α · [η(j)]^β / Σ [τ(l)]^α · [η(l)]^β

Обновление феромона:
  τ(j) ← (1-ρ)·τ(j) + ρ·Δτ(j)
  Δτ(j) = fitness(S) если j ∈ S иначе 0

η(j) = нормализованное среднее значение параметра компонента j.
```

### 4.6 Имитация отжига (SA)

```
Параметры:
  T0      Начальная температура (по умолчанию: 1.0)
  T_min   Конечная температура (по умолчанию: 1e-4)
  alpha   Скорость охлаждения (по умолчанию: 0.995)
  n_neighbors  Количество соседей за шаг (по умолчанию: 5)

Охлаждение: Tₜ = T₀ · alphaᵗ
Сосед: замена одного компонента на случайный не входящий в конфигурацию.
Принятие: P(принять худшее) = exp((новое - текущее) / T)
```

### 4.7 HybridDEGA

```
Фаза 1 (первые 60% от max_evals): DE/rand/1/bin
  — Глобальное исследование, популяционный подход
Фаза 2 (оставшиеся 40%): GA применяется к топ-50% популяции DE
  — Локальное уточнение через скрещивание/мутацию
```

### 4.8 ALGORITHM_REGISTRY

```python
ALGORITHM_REGISTRY: dict[str, type[BaseMetaheuristic]] = {
    "GA":         GeneticAlgorithm,
    "DE":         DifferentialEvolution,
    "PSO":        ParticleSwarmOptimization,
    "ACO":        AntColonyOptimization,
    "SA":         SimulatedAnnealing,
    "HybridDEGA": HybridDEGA,
}
```

---

## 5. Слой экспериментов

**Расположение:** `framework/experiment/runner.py`

```python
class ExperimentRunner:
    def add_problem(self, name: str, problem: BaseProblem): ...
    def add_algorithm(self, name: str, cls: type, params: dict): ...
    def run(self, n_runs: int, max_evals: int, base_seed: int) -> list[RunResult]: ...
    def get_summary(self) -> list[AlgorithmSummary]: ...
```

Поля `RunResult`:
- `algorithm: str`
- `run_id: int`
- `best_fitness: float`
- `best_config_ids: list[str]`
- `convergence: list[float]`
- `elapsed_sec: float`
- `objective_vector: list[float]`

Поля `AlgorithmSummary`:
- `algorithm: str`
- `best, mean, worst, std: float`
- `mean_time: float`

Прогресс передаётся клиенту через **Server-Sent Events**:
```
data: {"status": "running", "algorithm": "GA", "run": 3, "total": 20, "best": 7.234}
data: {"status": "done"}
```

---

## 6. Значения Шепли

**Расположение:** `shapley.py`

### Формула

$$\phi_i(v) = \sum_{S \subseteq \mathcal{A} \setminus \{i\}} \frac{|S|!\,(|\mathcal{A}| - |S| - 1)!}{|\mathcal{A}|!} \cdot \bigl[v(S \cup \{i\}) - v(S)\bigr]$$

Интерпретация: $\phi_i$ — средний предельный вклад компонента $i$ по всем возможным порядкам вступления в коалицию.

### Запрос к API

```http
POST /api/shapley
Content-Type: application/json

{
  "custom_problem_id": 42,
  "config_ids": ["3", "12", "57", "64", "80"],
  "method": "auto",
  "n_samples": 512
}
```

Ответ:
```json
{
  "shapley_values": [
    {"comp_id": "57", "name": "ML Engineer", "shapley": 15.956, "shapley_pct": 24.3, "rank": 1},
    ...
  ],
  "method_used": "exact",
  "n_players": 5
}
```

**Выбор метода:**
- `exact` — $2^n$ оценок коалиций, используется при $n \leq 12$
- `approx` — выборка из случайных перестановок, используется при $n > 12$
- `auto` — выбор на основе $n$ (по умолчанию)

---

## 7. Схема базы данных

```sql
CREATE TABLE catalogs (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    icon        TEXT DEFAULT '📦',
    created_at  REAL DEFAULT (unixepoch())
);

CREATE TABLE catalog_param_schema (
    id           SERIAL PRIMARY KEY,
    catalog_id   INTEGER REFERENCES catalogs(id) ON DELETE CASCADE,
    param_key    TEXT NOT NULL,
    param_label  TEXT NOT NULL,
    param_type   TEXT NOT NULL DEFAULT 'number',   -- 'number' | 'text'
    param_unit   TEXT DEFAULT '',
    param_options TEXT DEFAULT '[]',               -- JSON-массив для select-полей
    sort_order   INTEGER DEFAULT 0
);

CREATE TABLE components (
    id          SERIAL PRIMARY KEY,
    catalog_id  INTEGER REFERENCES catalogs(id) ON DELETE CASCADE,
    comp_id     TEXT NOT NULL,
    tags        TEXT DEFAULT '[]',    -- JSON-массив строк
    params      TEXT DEFAULT '{}',    -- JSON-объект
    note        TEXT DEFAULT '',
    active      INTEGER DEFAULT 1,
    created_at  REAL DEFAULT (unixepoch())
);

CREATE TABLE synergies (
    id          SERIAL PRIMARY KEY,
    catalog_id  INTEGER REFERENCES catalogs(id) ON DELETE CASCADE,
    comp_id_a   TEXT NOT NULL,
    comp_id_b   TEXT NOT NULL,
    value       REAL NOT NULL DEFAULT 0.0,
    note        TEXT DEFAULT '',
    UNIQUE(catalog_id, comp_id_a, comp_id_b)
);

CREATE TABLE custom_problems (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT DEFAULT '',
    catalog_id   INTEGER REFERENCES catalogs(id) ON DELETE CASCADE,
    config_size  INTEGER NOT NULL DEFAULT 5,
    constraints  TEXT DEFAULT '[]',   -- JSON-массив объектов ограничений
    objectives   TEXT DEFAULT '[]',   -- JSON-массив объектов целевых функций
    created_at   REAL DEFAULT (unixepoch())
);
```

### Схема объекта ограничения

```json
{
  "type": "param_agg | tag_required | tag_forbidden | count_tag",
  "param": "ключ_поля или значение_тега",
  "agg":   "sum | avg | max | min",
  "op":    "<= | >= | ==",
  "value": 400,
  "label": "необязательное описание"
}
```

### Схема объекта целевой функции

```json
{
  "param":  "ключ_поля | __synergy__ | __diversity__",
  "agg":    "avg | sum | max | min",
  "weight": 1.5,
  "label":  "отображаемое имя"
}
```

---

## 8. REST API

### Каталоги

```
GET    /api/catalogs                         Получить список всех каталогов (с количеством компонентов)
POST   /api/catalogs                         Создать каталог {name, description, icon}
GET    /api/catalogs/:id                     Получить каталог по ID
PUT    /api/catalogs/:id                     Обновить каталог
DELETE /api/catalogs/:id                     Удалить каталог и все данные

GET    /api/catalogs/:id/schema              Получить схему параметров
PUT    /api/catalogs/:id/schema              Заменить схему параметров (массив)
```

### Компоненты

```
GET    /api/catalogs/:id/components          Получить список активных компонентов
       ?active=0                             Включить неактивные компоненты
POST   /api/catalogs/:id/components          Создать компонент
POST   /api/catalogs/:id/components/batch    Массовый импорт (upsert по comp_id)
PUT    /api/components/:id                   Обновить компонент
DELETE /api/components/:id                   Удалить компонент
POST   /api/components/:id/toggle            Переключить активный/неактивный
```

### Синергии

```
GET    /api/catalogs/:id/synergies           Получить список синергий (отсортированы по убыванию значения)
POST   /api/catalogs/:id/synergies           Добавить синергии {comp_id_a, comp_id_b, value, note}
DELETE /api/synergies/:id                    Удалить синергию
```

### Пользовательские задачи

```
GET    /api/custom-problems                  Получить список всех задач
POST   /api/custom-problems                  Создать задачу
GET    /api/custom-problems/:id              Получить задачу с информацией о каталоге
PUT    /api/custom-problems/:id              Обновить задачу
DELETE /api/custom-problems/:id              Удалить задачу
GET    /api/custom-problems/:id/preview      Предпросмотр: количество компонентов, C(n,k), имена целевых функций
```

### Эксперименты

```
POST   /api/experiments                      Запустить эксперимент — возвращает {exp_id}
GET    /api/experiments/:id/stream           SSE поток событий прогресса
GET    /api/experiments/:id                  Полные результаты с best_config_detail
DELETE /api/experiments/:id                  Удалить из хранилища в памяти
```

**Тело запроса запуска эксперимента:**
```json
{
  "domain_id": "team_building",
  "config_size": 5,
  "custom_problem_id": 42,
  "algorithms": {
    "GA":  {"pop_size": 30, "cx_prob": 0.8, "mut_prob": 0.2},
    "SA":  {"T0": 1.0, "T_min": 1e-4, "alpha": 0.995}
  },
  "n_runs": 10,
  "max_evals": 2000
}
```

**Ответ с результатами (фрагмент):**
```json
{
  "status": "done",
  "summary": [
    {"algorithm": "GA", "best": 7.234, "mean": 7.190, "worst": 7.100, "std": 0.042, "mean_time": 0.07}
  ],
  "best_run": {
    "algorithm": "GA",
    "best_fitness": 7.234,
    "elapsed_sec": 0.063,
    "objective_vector": [85.0, 19.0, 88.0],
    "best_config_ids": ["12", "57", "2", "109", "65"]
  },
  "best_config_detail": [
    {"id": "57", "params": {"hero_name": "ML Engineer", "productivity": 93}, "tags": ["ML", "Senior"]}
  ],
  "problem_info": {
    "name": "Balanced Team",
    "objective_names": ["Collaboration", "Reliability", "Synergy"],
    "weights": [0.333, 0.333, 0.333]
  }
}
```

### Анализ

```
POST   /api/shapley       Вычисление значения Шепли {custom_problem_id|domain_id, config_ids, method, n_samples}
POST   /api/explain       Объяснение на основе правил {team, shapley_values, objective_names, weights, fitness, ...}
```

### Мета

```
GET    /api/domains        Получить список встроенных доменов с описаниями
GET    /api/algorithms     Получить список доступных алгоритмов со схемами параметров
```

---

## 9. Фронтенд

**Расположение:** `web/index.html` — одностраничное приложение, без зависимостей от фреймворков.

### Структура вкладок

| Вкладка | ID | Описание |
|---------|----|----------|
| Каталоги | `catalogs` | Управление компонентами, редактор схемы, матрица синергий |
| Задачи | `problems` | Конструктор пользовательских задач |
| Эксперимент | `experiment` | Запуск алгоритмов, просмотр результатов |
| Примеры | `examples` | Встроенные примеры доменов |

### Ключевые JavaScript-модули (в `index.html`)

| Функция | Описание |
|---------|----------|
| `init()` | Инициализация: загрузка каталогов, задач, алгоритмов |
| `selCat(id)` | Выбор каталога, рендер таблицы компонентов |
| `openImportModal(cid)` | Импорт CSV/Excel с маппингом столбцов |
| `openSyn(cid)` | Доступ к редактору синергий (3 вкладки: правка/тепловая карта/граф) |
| `openNewProb() / openEditProb(id)` | Конструктор задач |
| `saveProb()` | POST/PUT API |
| `runExp()` | Запуск эксперимента, SSE поток |
| `loadRes(id)` | Рендер результатов |
| `runShapley()` | POST `/api/shapley`, рендер полосы вкладов |
| `updateAlgRecommendation()` | Авторекомендация на основе характеристик задачи |
| `drawConv(results, canvas)` | График сходимости с подсказкой при наведении |
| `renderHeatmap()` | Тепловая карта синергий |
| `renderGraph()` | Граф синергий с силовой раскладкой |
| `toggleTheme()` | Переключение тёмной/светлой темы, сохранение в localStorage |

### Система тем

CSS-переменные с двумя темами:

```css
:root { --bg: #0d1117; --accent: #64b5f6; ... }            /* тёмная (по умолчанию) */
[data-theme="light"] { --bg: #f5f6f8; --accent: #1976d2; ... }  /* светлая */
```



---

## 10. Встроенные домены

**Расположение:** `framework/domains/adapters.py`

```python
DOMAIN_REGISTRY: dict[str, dict] = {
    "team_building": {
        "name": "Формирование IT-команды",
        "factory": create_team_building_problem,
        "default_config_size": 5,
        "description": "...",
    },
    ...
}

def create_problem(domain_id: str, config_size: int = None) -> BaseProblem:
    ...
```

---

## 11. Расширение фреймворка

### Добавление нового алгоритма

```python
# framework/algorithms/metaheuristics.py

class MyAlgorithm(BaseMetaheuristic):
    def __init__(self, problem: BaseProblem, max_evals: int, seed: int,
                 my_param: float = 0.5, **kwargs):
        super().__init__(problem, max_evals, seed)
        self.my_param = my_param

    def run(self) -> AlgorithmResult:
        start = time.time()
        best_cfg = self.problem.generate_random()
        best_fit = self.problem.evaluate(best_cfg)
        convergence = [best_fit]

        while self._evals < self.max_evals:
            # ... логика поиска ...
            candidate = ...
            fit = self.problem.evaluate(candidate)
            self._evals += 1
            if fit > best_fit:
                best_fit, best_cfg = fit, candidate
            convergence.append(best_fit)

        return AlgorithmResult(
            config=best_cfg, fitness=best_fit,
            convergence=convergence, eval_count=self._evals,
            elapsed_sec=time.time() - start,
        )

# Зарегистрировать
ALGORITHM_REGISTRY["MyAlgo"] = MyAlgorithm
```

### Добавление нового домена

```python
# framework/domains/adapters.py

def create_my_domain(config_size: int = 5) -> BaseProblem:
    components = [
        Component(id=str(i), params={...}, tags=[...])
        for i in range(20)
    ]
    synergies = InteractionMatrix({("1", "2"): 2.5, ("3", "4"): -1.0})

    def my_objective(cfg: Configuration) -> float:
        return sum(c.params["score"] for c in cfg.components) / len(cfg.components)

    def my_constraint(cfg: Configuration) -> bool:
        return sum(c.params["cost"] for c in cfg.components) <= 500

    return BaseProblem(
        name="Мой домен",
        components=components,
        interaction_matrix=synergies,
        objectives=[my_objective],
        objective_names=["Средний балл"],
        weights=[1.0],
        constraints=[my_constraint],
        constraint_names=["Бюджет ≤ 500"],
        config_size=config_size,
    )

DOMAIN_REGISTRY["my_domain"] = {
    "name": "Мой домен",
    "icon": "🔬",
    "factory": create_my_domain,
    "default_config_size": 5,
    "description": "Краткое описание для интерфейса.",
    "n_components": 20,
    "n_constraints": 1,
    "n_objectives": 1,
}
```

---