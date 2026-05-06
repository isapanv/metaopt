import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, Response
import json, threading, time, uuid, traceback

from framework.domains.adapters import DOMAIN_REGISTRY, create_problem
from framework.algorithms.metaheuristics import ALGORITHM_REGISTRY, list_algorithms
from framework.experiment.runner import ExperimentRunner
import database as db
import problem_builder

db.init_db()

app = Flask(__name__, static_folder="web/static", template_folder="web/templates")
app.config["JSON_ENSURE_ASCII"] = False
_experiments: dict = {}
_lock = threading.Lock()

@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "web", "index.html"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/api/domains")
def get_domains():
    result = []
    for domain_id, info in DOMAIN_REGISTRY.items():
        p = info["factory"](config_size=info["default_config_size"])
        result.append({"id": domain_id, "name": info["name"], "description": info["description"],
            "icon": info["icon"], "default_config_size": info["default_config_size"],
            "min_config_size": info["min_config_size"], "max_config_size": info["max_config_size"],
            "n_components": len(p.components), "n_constraints": len(p.constraints),
            "constraint_names": p.constraint_names, "n_objectives": len(p.objectives),
            "objective_names": p.objective_names, "weights": p.weights})
    return jsonify(result)

@app.route("/api/domains/<domain_id>")
def get_domain_detail(domain_id):
    try:
        p = create_problem(domain_id, request.args.get("config_size", type=int))
        return jsonify(p.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

@app.route("/api/algorithms")
def get_algorithms():
    return jsonify(list_algorithms())


@app.route("/api/catalogs", methods=["GET"])
def api_get_catalogs():
    return jsonify(db.get_catalogs())

@app.route("/api/catalogs", methods=["POST"])
def api_create_catalog():
    body = request.get_json()
    if not body or not body.get("name"):
        return jsonify({"error": "Поле name обязательно"}), 400
    cid = db.create_catalog(body["name"], body.get("description",""), body.get("icon","📦"))
    if body.get("param_schema"):
        db.set_param_schema(cid, body["param_schema"])
    return jsonify({"id": cid, "ok": True}), 201

@app.route("/api/catalogs/<int:cid>", methods=["GET"])
def api_get_catalog(cid):
    cat = db.get_catalog(cid)
    if not cat: return jsonify({"error": "Не найдено"}), 404
    cat["param_schema"] = db.get_param_schema(cid)
    cat["n_components"] = len(db.get_components(cid))
    return jsonify(cat)

@app.route("/api/catalogs/<int:cid>", methods=["PUT"])
def api_update_catalog(cid):
    body = request.get_json()
    db.update_catalog(cid, body["name"], body.get("description",""), body.get("icon","📦"))
    if "param_schema" in body:
        db.set_param_schema(cid, body["param_schema"])
    return jsonify({"ok": True})

@app.route("/api/catalogs/<int:cid>", methods=["DELETE"])
def api_delete_catalog(cid):
    db.delete_catalog(cid); return jsonify({"ok": True})

@app.route("/api/catalogs/<int:cid>/schema", methods=["GET"])
def api_get_schema(cid):
    return jsonify(db.get_param_schema(cid))

@app.route("/api/catalogs/<int:cid>/schema", methods=["PUT"])
def api_set_schema(cid):
    db.set_param_schema(cid, request.get_json()); return jsonify({"ok": True})


@app.route("/api/catalogs/<int:cid>/components", methods=["GET"])
def api_get_components(cid):
    return jsonify(db.get_components(cid, active_only=request.args.get("active","1")=="1"))

@app.route("/api/catalogs/<int:cid>/components", methods=["POST"])
def api_create_component(cid):
    body = request.get_json()
    if not body or not body.get("comp_id"):
        return jsonify({"error": "Поле comp_id обязательно"}), 400
    new_id = db.create_component(cid, body["comp_id"], body.get("params",{}), body.get("tags",[]), body.get("note",""))
    return jsonify({"id": new_id, "ok": True}), 201

@app.route("/api/catalogs/<int:cid>/components/batch", methods=["POST"])
def api_batch_components(cid):
    body = request.get_json()
    if not isinstance(body, list): return jsonify({"error": "Ожидается список"}), 400
    ids = [db.create_component(cid, item["comp_id"], item.get("params",{}), item.get("tags",[]), item.get("note",""))
           for item in body if item.get("comp_id")]
    return jsonify({"created": len(ids), "ids": ids})

@app.route("/api/components/<int:comp_id>", methods=["GET"])
def api_get_component(comp_id):
    c = db.get_component(comp_id)
    return jsonify(c) if c else (jsonify({"error": "Не найдено"}), 404)

@app.route("/api/components/<int:comp_id>", methods=["PUT"])
def api_update_component(comp_id):
    body = request.get_json()
    db.update_component(comp_id, body["comp_id"], body.get("params",{}), body.get("tags",[]), body.get("note",""))
    return jsonify({"ok": True})

@app.route("/api/components/<int:comp_id>", methods=["DELETE"])
def api_delete_component(comp_id):
    db.delete_component(comp_id); return jsonify({"ok": True})

@app.route("/api/components/<int:comp_id>/toggle", methods=["POST"])
def api_toggle_component(comp_id):
    body = request.get_json()
    db.toggle_component(comp_id, body.get("active", True)); return jsonify({"ok": True})

@app.route("/api/catalogs/<int:cid>/synergies", methods=["GET"])
def api_get_synergies(cid):
    return jsonify(db.get_synergies(cid))

@app.route("/api/catalogs/<int:cid>/synergies", methods=["POST"])
def api_set_synergy(cid):
    body = request.get_json()
    db.set_synergy(cid, body["comp_id_a"], body["comp_id_b"], float(body["value"]), body.get("note",""))
    return jsonify({"ok": True})

@app.route("/api/synergies/<int:syn_id>", methods=["DELETE"])
def api_delete_synergy(syn_id):
    db.delete_synergy(syn_id); return jsonify({"ok": True})

@app.route("/api/custom-problems", methods=["GET"])
def api_get_custom_problems():
    return jsonify(db.get_custom_problems())

@app.route("/api/custom-problems", methods=["POST"])
def api_create_custom_problem():
    body = request.get_json()
    for f in ["name","catalog_id","config_size","objectives"]:
        if f not in body: return jsonify({"error": f"Поле {f} обязательно"}), 400
    pid = db.create_custom_problem(body["name"], body.get("description",""), int(body["catalog_id"]),
                                    int(body["config_size"]), body.get("constraints",[]), body["objectives"])
    return jsonify({"id": pid, "ok": True}), 201

@app.route("/api/custom-problems/<int:pid>", methods=["GET"])
def api_get_custom_problem(pid):
    p = db.get_custom_problem(pid)
    return jsonify(p) if p else (jsonify({"error": "Не найдено"}), 404)

@app.route("/api/custom-problems/<int:pid>", methods=["PUT"])
def api_update_custom_problem(pid):
    body = request.get_json()
    db.update_custom_problem(pid, body["name"], body.get("description",""),
                              int(body["config_size"]), body.get("constraints",[]), body["objectives"])
    return jsonify({"ok": True})

@app.route("/api/custom-problems/<int:pid>", methods=["DELETE"])
def api_delete_custom_problem(pid):
    db.delete_custom_problem(pid); return jsonify({"ok": True})

@app.route("/api/custom-problems/<int:pid>/preview", methods=["GET"])
def api_preview_custom_problem(pid):
    try:
        return jsonify(problem_builder.build_problem_from_db(pid).to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/experiments", methods=["POST"])
def create_experiment():
    body = request.get_json()
    if not body: return jsonify({"error": "Тело запроса отсутствует"}), 400
    exp_id = str(uuid.uuid4())[:8]
    with _lock:
        _experiments[exp_id] = {"id": exp_id, "status": "pending", "progress": [],
                                  "results": None, "error": None, "config": body, "created_at": time.time()}
    threading.Thread(target=_run_experiment, args=(exp_id, body), daemon=True).start()
    return jsonify({"experiment_id": exp_id, "status": "pending"})

def _run_experiment(exp_id, config):
    try:
        if "custom_problem_id" in config:
            problem = problem_builder.build_problem_from_db(int(config["custom_problem_id"]))
        else:
            problem = create_problem(config["domain_id"], config.get("config_size"))

        runner = ExperimentRunner()
        runner.add_problem(problem.name, problem)
        for alg_cfg in config.get("algorithms", []):
            alg_id = alg_cfg["id"]
            schema = ALGORITHM_REGISTRY[alg_id].PARAMS_SCHEMA
            typed = {}
            for k, v in alg_cfg.get("params", {}).items():
                t = schema.get(k, {}).get("type", "float")
                typed[k] = int(v) if t == "int" else (float(v) if t == "float" else v)
            runner.add_algorithm(alg_id, ALGORITHM_REGISTRY[alg_id], typed)

        with _lock: _experiments[exp_id]["status"] = "running"
        prog = []
        def cb(info):
            prog.append(info)
            with _lock: _experiments[exp_id]["progress"] = list(prog)
        runner.set_progress_callback(cb)
        results = runner.run(n_runs=config.get("n_runs",5), max_evals=config.get("max_evals",500), base_seed=config.get("base_seed",0))
        summary = runner.get_summary()
        best_run = max(results, key=lambda r: r.best_fitness) if results else None
        comp_map = {c.id: c.to_dict() for c in problem.components}
        best_detail = [comp_map[cid] for cid in (best_run.best_config_ids if best_run else []) if cid in comp_map]
        with _lock:
            _experiments[exp_id].update({"status":"done","results":[r.to_dict() for r in results],
                "summary":[s.to_dict() for s in summary],"best_run":best_run.to_dict() if best_run else None,
                "best_config_detail":best_detail,"problem_info":problem.to_dict(),"finished_at":time.time()})
    except Exception as e:
        with _lock: _experiments[exp_id].update({"status":"error","error":str(e)+"\n"+traceback.format_exc()})

@app.route("/api/experiments/<exp_id>")
def get_experiment(exp_id):
    with _lock: exp = _experiments.get(exp_id)
    return jsonify(exp) if exp else (jsonify({"error":"Не найдено"}),404)

@app.route("/api/experiments/<exp_id>/stream")
def stream_experiment(exp_id):
    def gen():
        n = 0
        while True:
            with _lock: exp = _experiments.get(exp_id)
            if not exp: yield f"data: {json.dumps({'error':'not found'})}\n\n"; break
            prog = exp.get("progress",[])
            for item in prog[n:]: yield f"data: {json.dumps(item)}\n\n"
            n = len(prog)
            s = exp.get("status")
            if s == "done": yield f"data: {json.dumps({'status':'done'})}\n\n"; break
            elif s == "error": yield f"data: {json.dumps({'status':'error','error':exp.get('error','')})}\n\n"; break
            time.sleep(0.3)
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/experiments")
def list_experiments():
    with _lock:
        return jsonify([{"id":e["id"],"status":e["status"],"config":e["config"],"created_at":e["created_at"]}
                        for e in sorted(_experiments.values(),key=lambda x:x["created_at"],reverse=True)])

@app.route("/api/experiments/<exp_id>", methods=["DELETE"])
def delete_experiment(exp_id):
    with _lock:
        if exp_id in _experiments: del _experiments[exp_id]; return jsonify({"ok":True})
    return jsonify({"error":"Не найдено"}),404



@app.route("/api/explain", methods=["POST"])
def explain_solution():
    import metaopt.explainer as explainer
    body = request.get_json()
    if not body:
        return jsonify({"error": "Тело запроса отсутствует"}), 400
    try:
        result = explainer.generate_explanation(
            team=body.get("team", []),
            shapley_values=body.get("shapley_values", []),
            objective_names=body.get("objective_names", []),
            weights=body.get("weights", []),
            fitness=float(body.get("fitness", 0)),
            algorithm=body.get("algorithm", ""),
            objective_vector=body.get("objective_vector"),
            problem_name=body.get("problem_name", ""),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/shapley", methods=["POST"])
def compute_shapley_route():
    import shapley as shp
    body = request.get_json()
    if not body:
        return jsonify({"error": "Тело запроса отсутствует"}), 400
    config_ids = body.get("config_ids", [])
    if not config_ids:
        return jsonify({"error": "Поле config_ids обязательно"}), 400
    try:
        if "custom_problem_id" in body:
            problem = problem_builder.build_problem_from_db(int(body["custom_problem_id"]))
        else:
            problem = create_problem(body["domain_id"], body.get("config_size"))
        result = shp.compute_shapley(
            problem=problem,
            selected_ids=config_ids,
            method=body.get("method", "auto"),
            n_samples=int(body.get("n_samples", 512)),
            seed=int(body.get("seed", 42)),
        )
        return jsonify({
            "shapley_values": result,
            "method_used": "exact" if len(config_ids) <= 12 else "approx",
            "n_players": len(config_ids),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("="*60)
    print("  MetaOpt Framework — http://localhost:5000")
    print("="*60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)