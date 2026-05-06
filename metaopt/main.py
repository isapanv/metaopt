import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.domains.adapters import create_problem, DOMAIN_REGISTRY
from framework.algorithms.metaheuristics import ALGORITHM_REGISTRY
from framework.experiment.runner import ExperimentRunner


def main():
    for domain_id, info in DOMAIN_REGISTRY.items():
        print(f"\n{'─'*50}")
        print(f"  {info['icon']}  {info['name']}")
        print(f"{'─'*50}")

        problem = create_problem(domain_id)
        print(f"  Компонентов: {len(problem.components)}, config_size={problem.config_size}")
        print(f"  Ограничения: {', '.join(problem.constraint_names)}")
        print(f"  Критерии: {', '.join(f'{n}(×{w})' for n,w in zip(problem.objective_names, problem.weights))}")

        runner = ExperimentRunner()
        runner.add_problem(problem.name, problem)

        runner.add_algorithm("GA",  ALGORITHM_REGISTRY["GA"],  {"pop_size": 20})
        runner.add_algorithm("PSO", ALGORITHM_REGISTRY["PSO"], {"pop_size": 20})
        runner.add_algorithm("DE",  ALGORITHM_REGISTRY["DE"],  {"pop_size": 20})

        def on_progress(info):
            print(f"  [{info['done']}/{info['total']}] {info['algorithm']:10s} "
                  f"run#{info['run_id']} → fitness={info['fitness']:.4f}")

        runner.set_progress_callback(on_progress)
        runner.run(n_runs=3, max_evals=300)
        runner.print_summary()

        best = max(runner._results, key=lambda r: r.best_fitness)
        print(f"\n Лучшее: {best.algorithm} fitness={best.best_fitness:.4f}")
        print(f"  Состав: {best.best_config_ids}")

    print("\n" + "=" * 65)
    print("  Готово! Запустите веб-интерфейс: python app.py")
    print("=" * 65)

    runner.export_csv("results_last_domain.csv")
    runner.export_json("results_last_domain.json")
    print("  Результаты сохранены в results_last_domain.csv / .json")


if __name__ == "__main__":
    main()
