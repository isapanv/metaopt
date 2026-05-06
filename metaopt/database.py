import psycopg2
import json
import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://metaopt:secret@localhost:5432/metaopt_db"
)

def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn

def _fetchall(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def _fetchone(cur):
    if cur.description is None: return None
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS catalogs (
        id SERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE,
        description TEXT DEFAULT '', icon TEXT DEFAULT '📦',
        created_at REAL DEFAULT extract(epoch from now())
    );
    CREATE TABLE IF NOT EXISTS catalog_param_schema (
        id SERIAL PRIMARY KEY, catalog_id INTEGER NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
        param_key TEXT NOT NULL, param_label TEXT NOT NULL,
        param_type TEXT NOT NULL DEFAULT 'number', param_unit TEXT DEFAULT '',
        param_options TEXT DEFAULT '[]', sort_order INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS components (
        id SERIAL PRIMARY KEY, catalog_id INTEGER NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
        comp_id TEXT NOT NULL, tags TEXT DEFAULT '[]', params TEXT DEFAULT '{}',
        note TEXT DEFAULT '', active INTEGER DEFAULT 1,
        created_at REAL DEFAULT extract(epoch from now())
    );
    CREATE TABLE IF NOT EXISTS synergies (
        id SERIAL PRIMARY KEY, catalog_id INTEGER NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
        comp_id_a TEXT NOT NULL, comp_id_b TEXT NOT NULL,
        value REAL NOT NULL DEFAULT 0.0, note TEXT DEFAULT '',
        UNIQUE(catalog_id, comp_id_a, comp_id_b)
    );
    CREATE TABLE IF NOT EXISTS custom_problems (
        id SERIAL PRIMARY KEY, name TEXT NOT NULL, description TEXT DEFAULT '',
        catalog_id INTEGER NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
        config_size INTEGER NOT NULL DEFAULT 5,
        constraints TEXT DEFAULT '[]', objectives TEXT DEFAULT '[]',
        created_at REAL DEFAULT extract(epoch from now())
    );
    """)
    conn.commit(); cur.close(); conn.close()

def get_catalogs():
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""SELECT c.*, COUNT(comp.id) as n_components FROM catalogs c
        LEFT JOIN components comp ON comp.catalog_id = c.id AND comp.active = 1
        GROUP BY c.id ORDER BY c.created_at DESC""")
    rows = _fetchall(cur); cur.close(); conn.close(); return rows

def get_catalog(catalog_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM catalogs WHERE id=%s", (catalog_id,))
    row = _fetchone(cur); cur.close(); conn.close(); return row

def create_catalog(name, description='', icon='📦'):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("INSERT INTO catalogs (name,description,icon) VALUES (%s,%s,%s) RETURNING id",
                (name, description, icon))
    new_id = cur.fetchone()[0]; conn.commit(); cur.close(); conn.close(); return new_id

def update_catalog(catalog_id, name, description, icon):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("UPDATE catalogs SET name=%s,description=%s,icon=%s WHERE id=%s",
                (name, description, icon, catalog_id))
    conn.commit(); cur.close(); conn.close()

def delete_catalog(catalog_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM catalogs WHERE id=%s", (catalog_id,))
    conn.commit(); cur.close(); conn.close()

def get_param_schema(catalog_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM catalog_param_schema WHERE catalog_id=%s ORDER BY sort_order", (catalog_id,))
    rows = _fetchall(cur); cur.close(); conn.close()
    for r in rows: r['param_options'] = json.loads(r.get('param_options') or '[]')
    return rows

def set_param_schema(catalog_id, schema):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM catalog_param_schema WHERE catalog_id=%s", (catalog_id,))
    for i, item in enumerate(schema):
        cur.execute("""INSERT INTO catalog_param_schema
            (catalog_id,param_key,param_label,param_type,param_unit,param_options,sort_order)
            VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (catalog_id, item['param_key'], item['param_label'],
             item.get('param_type','number'), item.get('param_unit',''),
             json.dumps(item.get('param_options',[])), i))
    conn.commit(); cur.close(); conn.close()

def get_components(catalog_id, active_only=True):
    conn = get_connection(); cur = conn.cursor()
    q = "SELECT * FROM components WHERE catalog_id=%s"
    if active_only: q += " AND active=1"
    q += " ORDER BY CASE WHEN comp_id ~ '^[0-9]+$' THEN comp_id::integer ELSE 999999 END, comp_id"
    cur.execute(q, (catalog_id,)); rows = _fetchall(cur); cur.close(); conn.close()
    for r in rows:
        r['params'] = json.loads(r.get('params') or '{}')
        r['tags'] = json.loads(r.get('tags') or '[]')
    return rows

def get_component(comp_db_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM components WHERE id=%s", (comp_db_id,))
    r = _fetchone(cur); cur.close(); conn.close()
    if r:
        r['params'] = json.loads(r.get('params') or '{}')
        r['tags'] = json.loads(r.get('tags') or '[]')
    return r

def create_component(catalog_id, comp_id, params, tags=None, note=''):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("INSERT INTO components (catalog_id,comp_id,params,tags,note) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                (catalog_id, comp_id, json.dumps(params), json.dumps(tags or []), note))
    new_id = cur.fetchone()[0]; conn.commit(); cur.close(); conn.close(); return new_id

def update_component(comp_db_id, comp_id, params, tags, note):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("UPDATE components SET comp_id=%s,params=%s,tags=%s,note=%s WHERE id=%s",
                (comp_id, json.dumps(params), json.dumps(tags), note, comp_db_id))
    conn.commit(); cur.close(); conn.close()

def delete_component(comp_db_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM components WHERE id=%s", (comp_db_id,))
    conn.commit(); cur.close(); conn.close()

def toggle_component(comp_db_id, active):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("UPDATE components SET active=%s WHERE id=%s", (1 if active else 0, comp_db_id))
    conn.commit(); cur.close(); conn.close()

def get_synergies(catalog_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM synergies WHERE catalog_id=%s ORDER BY comp_id_a, comp_id_b", (catalog_id,))
    rows = _fetchall(cur); cur.close(); conn.close(); return rows

def set_synergy(catalog_id, comp_id_a, comp_id_b, value, note=''):
    if comp_id_a > comp_id_b: comp_id_a, comp_id_b = comp_id_b, comp_id_a
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""INSERT INTO synergies (catalog_id,comp_id_a,comp_id_b,value,note) VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT (catalog_id,comp_id_a,comp_id_b) DO UPDATE SET value=EXCLUDED.value,note=EXCLUDED.note""",
        (catalog_id, comp_id_a, comp_id_b, value, note))
    conn.commit(); cur.close(); conn.close()

def delete_synergy(synergy_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM synergies WHERE id=%s", (synergy_id,))
    conn.commit(); cur.close(); conn.close()

def get_custom_problems():
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""SELECT cp.*, c.name as catalog_name, c.icon as catalog_icon
        FROM custom_problems cp JOIN catalogs c ON c.id=cp.catalog_id ORDER BY cp.created_at DESC""")
    rows = _fetchall(cur); cur.close(); conn.close()
    for r in rows:
        r['constraints'] = json.loads(r.get('constraints') or '[]')
        r['objectives'] = json.loads(r.get('objectives') or '[]')
    return rows

def get_custom_problem(problem_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM custom_problems WHERE id=%s", (problem_id,))
    r = _fetchone(cur); cur.close(); conn.close()
    if r:
        r['constraints'] = json.loads(r.get('constraints') or '[]')
        r['objectives'] = json.loads(r.get('objectives') or '[]')
    return r

def create_custom_problem(name, description, catalog_id, config_size, constraints, objectives):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""INSERT INTO custom_problems (name,description,catalog_id,config_size,constraints,objectives)
        VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
        (name, description, catalog_id, config_size, json.dumps(constraints), json.dumps(objectives)))
    new_id = cur.fetchone()[0]; conn.commit(); cur.close(); conn.close(); return new_id

def update_custom_problem(problem_id, name, description, config_size, constraints, objectives):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""UPDATE custom_problems SET name=%s,description=%s,config_size=%s,constraints=%s,objectives=%s
        WHERE id=%s""",
        (name, description, config_size, json.dumps(constraints), json.dumps(objectives), problem_id))
    conn.commit(); cur.close(); conn.close()

def delete_custom_problem(problem_id):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM custom_problems WHERE id=%s", (problem_id,))
    conn.commit(); cur.close(); conn.close()
