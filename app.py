# region Imports
import os
import re
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import datetime as _dt

import streamlit as st
import pandas as pd
from PIL import Image

# Cognite CDF
from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthClientCredentials
from cognite.client.data_classes import data_modeling as dm

# SQLAlchemy / MySQL
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
# endregion Imports


# region App Meta & Branding
# ------------------------------------------------------------
# Version & app identity shown in the UI
# ------------------------------------------------------------
APP_VERSION = "1.3.0 - dev"  # bump to 1.3.0 on release
APP_TITLE = "Convergix DataMosaix View Explorer"
APP_ICON_PATH = "assets/convergix_logo.png"

_icon = None
try:
    _icon = Image.open(APP_ICON_PATH)
except Exception:
    pass

# Streamlit page config (must be set before any UI rendering)
st.set_page_config(
    page_title=f"{APP_TITLE} · v{APP_VERSION}",
    page_icon=_icon if _icon else "🧭",
    layout="wide",
)

# App header with logo + title
left, right = st.columns([1, 8])
with left:
    if _icon:
        st.image(_icon, width=60)
with right:
    st.title(APP_TITLE)
    st.caption(f"Version {APP_VERSION}")
# endregion App Meta & Branding


# region Utilities, Environment, Logging
# ------------------------------------------------------------
# Helpers for environment loading, persistent JSON profiles,
# and application logging (with timestamps).
# ------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    """Get environment variable with fallback and strip whitespace."""
    return (os.getenv(name, default) or "").strip()

def _ts() -> str:
    """Timestamp for log entries."""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    """Append a timestamped line to the session log buffer."""
    st.session_state.setdefault("logs", [])
    st.session_state.logs.append(f"[{_ts()}] {msg}")

# persistent profile stores (bind-mounted volume at /data)
PROFILE_STORE_PATH_CDF = _env("PROFILE_STORE_PATH", "/data/profiles.json")
PROFILE_STORE_PATH_MYSQL = _env("MYSQL_PROFILE_STORE_PATH", "/data/mysql_profiles.json")

def _path(p: str) -> Path:
    """Ensure the parent folder exists and return Path."""
    pp = Path(p)
    pp.parent.mkdir(parents=True, exist_ok=True)
    return pp

def load_json_profiles(path: str) -> Dict[str, Dict[str, str]]:
    """Load a dict of named connection profiles from a JSON file."""
    f = _path(path)
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass
    return {}

def save_json_profile(path: str, name: str, profile: Dict[str, str]):
    """Upsert a single profile in the JSON store."""
    data = load_json_profiles(path)
    data[name] = profile
    _path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def delete_json_profile(path: str, name: str):
    """Delete a named profile from the JSON store."""
    data = load_json_profiles(path)
    if name in data:
        del data[name]
    _path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def env_profiles(prefix_list_var: str = "DATAMOSAIX_PROFILES", var_prefix: str = "DMX_") -> Dict[str, Dict[str, str]]:
    """
    Load CDF profiles from environment variables.

    Example:
    DATAMOSAIX_PROFILES=prod,dev
    DMX_PROD_HOST=...
    DMX_PROD_PROJECT=...
    ...
    """
    names = [n.strip() for n in _env(prefix_list_var, "").split(",") if n.strip()]
    profiles: Dict[str, Dict[str, str]] = {}
    for n in names:
        key = n.upper().replace("-", "_")
        def gv(k): return _env(f"{var_prefix}{key}_{k}")
        prof = {
            "host": gv("HOST"),
            "project": gv("PROJECT"),
            "token_url": gv("TOKENURL"),
            "client_id": gv("CLIENTID"),
            "client_secret": gv("CLIENTSECRET"),
            "scopes": gv("SCOPES") or "user_impersonation",
        }
        if prof["host"] and prof["project"]:
            profiles[n] = prof
    return profiles

def mysql_env_profiles(prefix_list_var: str = "MYSQL_PROFILES", var_prefix: str = "MYSQL_") -> Dict[str, Dict[str, str]]:
    """
    Load MySQL profiles from environment variables.

    Example:
    MYSQL_PROFILES=local,stage
    MYSQL_LOCAL_HOST=...
    MYSQL_LOCAL_DB=...
    ...
    """
    names = [n.strip() for n in _env(prefix_list_var, "").split(",") if n.strip()]
    profiles: Dict[str, Dict[str, str]] = {}
    for n in names:
        key = n.upper().replace("-", "_")
        def gv(k): return _env(f"{var_prefix}{key}_{k}")
        prof = {
            "host": gv("HOST") or _env("MYSQL_HOST"),
            "port": gv("PORT") or _env("MYSQL_PORT") or "3306",
            "user": gv("USER") or _env("MYSQL_USER"),
            "password": gv("PASSWORD") or _env("MYSQL_PASSWORD"),
            "db": gv("DB") or _env("MYSQL_DB"),
        }
        if prof["host"] and prof["user"] and prof["db"]:
            profiles[n] = prof
    return profiles
# endregion Utilities, Environment, Logging


# region DataMosaix (CDF) client and helpers
# ------------------------------------------------------------
# Build authenticated CDF client and cache model/view lookups
# ------------------------------------------------------------
def build_client(host_url: str, project: str, token_url: str, client_id: str, client_secret: str, scopes_csv: str) -> CogniteClient:
    """Build an OAuth client-credentialed Cognite client."""
    def need(n, v):
        v = (v or "").strip()
        if not v:
            raise RuntimeError(f"Missing required field: {n}")
        return v
    host_url = need("Host", host_url).rstrip("/")
    project = need("Project", project)
    token_url = need("Token URL", token_url)
    client_id = need("Client ID", client_id)
    client_secret = need("Client Secret", client_secret)
    scopes = [s.strip() for s in (scopes_csv or "user_impersonation").split(",") if s.strip()]
    creds = OAuthClientCredentials(token_url=token_url, client_id=client_id, client_secret=client_secret, scopes=scopes)
    cfg = ClientConfig(client_name=project, project=project, credentials=creds, base_url=host_url)
    return CogniteClient(cfg)

@st.cache_data(show_spinner=False)
def list_spaces(_client: CogniteClient, conn_key: str) -> List[str]:
    """Return all spaces containing any data models."""
    models = _client.data_modeling.data_models.list(limit=None, all_versions=False)
    return sorted({m.space for m in models})

@st.cache_data(show_spinner=False)
def list_models_latest(_client: CogniteClient, conn_key: str, space: str) -> List[Tuple[str, str]]:
    """Return latest versions of all Data Models in a space."""
    models = _client.data_modeling.data_models.list(limit=None, all_versions=False)
    return [(m.external_id, str(m.version)) for m in models if m.space == space]

@st.cache_data(show_spinner=False)
def list_views_for_model(_client: CogniteClient, conn_key: str, space: str, dm_eid: str, dm_ver: str) -> List[Tuple[str, str]]:
    """Return views (eid, version) contained in a Data Model."""
    dmid = dm.DataModelId(space=space, external_id=dm_eid, version=str(dm_ver))
    details = _client.data_modeling.data_models.retrieve(dmid, inline_views=True)
    if not details:
        return []
    model = details[0]
    if not getattr(model, "views", None):
        return []
    return [(v.external_id, str(v.version)) for v in model.views]

def flatten_properties_from_instance(inst) -> Dict[str, Any]:
    """Safely flatten instance properties across sources."""
    props_obj = getattr(inst, "properties", None)
    if props_obj:
        if isinstance(props_obj, dict):
            return props_obj
        if hasattr(props_obj, "items"):
            merged: Dict[str, Any] = {}
            for _, v in props_obj.items():
                if isinstance(v, dict):
                    merged.update(v)
                elif hasattr(v, "dump"):
                    dv = v.dump(camel_case=False)
                    if isinstance(dv, dict):
                        merged.update(dv)
            if merged:
                return merged
    try:
        d = inst.dump(camel_case=False)
        p = d.get("properties")
        if isinstance(p, dict):
            merged = {}
            for v in p.values():
                if isinstance(v, dict):
                    merged.update(v)
            if merged:
                return merged
        s = d.get("sources")
        if isinstance(s, list):
            merged = {}
            for src in s:
                pv = src.get("properties", {})
                if isinstance(pv, dict):
                    merged.update(pv)
            if merged:
                return merged
    except Exception:
        pass
    return {}

@st.cache_data(show_spinner=True)
def fetch_view_dataframe(
    _client: CogniteClient,
    conn_key: str,
    space: str,
    dm_eid: str,
    dm_ver: str,
    view_eid: str,
    view_ver: str,
    max_rows: Optional[int],
) -> pd.DataFrame:
    """Download nodes from a View into a flattened DataFrame."""
    view_id = dm.ViewId(space=space, external_id=view_eid, version=str(view_ver))
    view_def_list = _client.data_modeling.views.retrieve(view_id, include_inherited_properties=True)
    if not view_def_list:
        raise RuntimeError("View not found.")
    props = list(view_def_list[0].properties.keys())
    columns = ["space", "external_id", *props]
    rows = []
    for i, inst in enumerate(_client.data_modeling.instances.list(instance_type="node", sources=[view_id], limit=None)):
        flat = flatten_properties_from_instance(inst)
        row = [getattr(inst, "space", ""), getattr(inst, "external_id", "")]
        for col in props:
            val = flat.get(col, "")
            if isinstance(val, (dict, list)):
                val = str(val)
            row.append(val)
        rows.append(row)
        if max_rows and (i + 1) >= max_rows:
            break
    df = pd.DataFrame(rows, columns=columns)
    df["__view__"] = view_eid
    df["__view_version__"] = str(view_ver)
    return df
# endregion DataMosaix (CDF) client and helpers


# region DataFrame filtering widgets
# ------------------------------------------------------------
# Reusable, richer filtering UI for displayed tables
# ------------------------------------------------------------
def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_datetime_series(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _as_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def filter_dataframe(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Filter UI with stable keys and operators for text/number/datetime.
    key_prefix must remain stable per (DB, table) to persist widget state.
    """
    with st.expander("Filters", expanded=False):
        # global fuzzy search
        q = st.text_input("Search across all columns", placeholder="Type to filter…", key=f"{key_prefix}_q")
        filtered = df
        if q:
            mask = pd.Series(False, index=filtered.index)
            for col in filtered.columns:
                try:
                    mask |= filtered[col].astype(str).str.contains(q, case=False, na=False)
                except Exception:
                    pass
            filtered = filtered[mask]

        # per-column filters
        cols = st.multiselect("Add per-column filters", sorted(list(df.columns)), key=f"{key_prefix}_cols")

        for c in cols:
            s = filtered[c]

            # DATETIME
            if _is_datetime_series(s) or (not _is_numeric_series(s) and pd.to_datetime(s, errors="coerce").notna().sum() > 0 and s.dropna().shape[0] > 0):
                s2 = _as_dates(s)
                op = st.selectbox(f"{c} operator", ["Between", "On date", "Not on date"], key=f"{key_prefix}_{c}_op_dt")
                min_d, max_d = s2.min(), s2.max()
                if pd.isna(min_d) or pd.isna(max_d):
                    min_d = pd.Timestamp("1970-01-01")
                    max_d = pd.Timestamp.today()

                if op == "Between":
                    a, b = st.date_input(f"{c} between", value=(min_d.date(), max_d.date()), key=f"{key_prefix}_{c}_between")
                    if a:
                        filtered = filtered[s2 >= pd.to_datetime(a)]
                    if b:
                        filtered = filtered[s2 <= pd.to_datetime(b)]

                elif op == "On date":
                    d = st.date_input(f"{c} on", value=min_d.date(), key=f"{key_prefix}_{c}_on")
                    if isinstance(d, tuple):
                        d = d[0] if d else None
                    if d:
                        d0 = pd.to_datetime(d); d1 = d0 + pd.Timedelta(days=1)
                        filtered = filtered[(s2 >= d0) & (s2 < d1)]

                elif op == "Not on date":
                    d = st.date_input(f"{c} not on", value=min_d.date(), key=f"{key_prefix}_{c}_noton")
                    if isinstance(d, tuple):
                        d = d[0] if d else None
                    if d:
                        d0 = pd.to_datetime(d); d1 = d0 + pd.Timedelta(days=1)
                        filtered = filtered[(s2 < d0) | (s2 >= d1)]

            # NUMERIC
            elif _is_numeric_series(s):
                op = st.selectbox(f"{c} operator", ["Equals", "Not Equals", "Range"], key=f"{key_prefix}_{c}_op_num")
                if op == "Equals":
                    val = st.number_input(f"{c} =", value=float(s.dropna().median()) if s.dropna().size else 0.0, key=f"{key_prefix}_{c}_eq")
                    filtered = filtered[s.astype(float) == float(val)]
                elif op == "Not Equals":
                    val = st.number_input(f"{c} ≠", value=float(s.dropna().median()) if s.dropna().size else 0.0, key=f"{key_prefix}_{c}_neq")
                    filtered = filtered[s.astype(float) != float(val)]
                elif op == "Range":
                    mn, mx = float(s.min()) if s.size else 0.0, float(s.max()) if s.size else 1.0
                    a, b = st.slider(f"{c} range", mn, mx, (mn, mx), key=f"{key_prefix}_{c}_range")
                    filtered = filtered[(s.astype(float) >= a) & (s.astype(float) <= b)]

            # TEXT / CATEGORICAL
            else:
                op = st.selectbox(f"{c} operator", ["Contains", "Not Contains", "Equals", "Not Equals"], key=f"{key_prefix}_{c}_op_txt")
                if op in ("Contains", "Not Contains"):
                    val = st.text_input(f"{c} value", key=f"{key_prefix}_{c}_txt_contains")
                    if val:
                        contains_mask = s.astype(str).str.contains(val, case=False, na=False)
                        filtered = filtered[contains_mask] if op == "Contains" else filtered[~contains_mask]
                else:
                    uniques = s.dropna().astype(str).unique()
                    if len(uniques) <= 200:
                        val = st.selectbox(f"{c} value", ["(pick)"] + sorted(map(str, uniques)), key=f"{key_prefix}_{c}_pick_eq")
                        if val and val != "(pick)":
                            mask = s.astype(str) == val
                            filtered = filtered[mask] if op == "Equals" else filtered[~mask]
                    else:
                        val = st.text_input(f"{c} value", key=f"{key_prefix}_{c}_txt_eq")
                        if val:
                            mask = s.astype(str) == val
                            filtered = filtered[mask] if op == "Equals" else filtered[~mask]

    return filtered
# endregion DataFrame filtering widgets


# region MySQL helpers
# ------------------------------------------------------------
# Engine construction, simple reads/writes, and table listing
# ------------------------------------------------------------
def _mysql_env_default() -> Dict[str, str]:
    """Default MySQL connection values (via env or sensible defaults)."""
    return {
        "host": _env("MYSQL_HOST", "host.docker.internal"),
        "port": _env("MYSQL_PORT", "3306"),
        "user": _env("MYSQL_USER", ""),
        "password": _env("MYSQL_PASSWORD", ""),
        "db": _env("MYSQL_DB", ""),
    }

def _sanitize_table_name(name: str) -> str:
    """Sanitize arbitrary strings for use as a MySQL table name."""
    s = (name or "cdf_view").strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    return s or "cdf_view"

def _mysql_engine_from_profile(prof: Dict[str, str]):
    """Create a SQLAlchemy engine from a profile dict."""
    url = URL.create(
        "mysql+pymysql",
        username=prof.get("user") or "",
        password=prof.get("password") or "",
        host=prof.get("host") or "localhost",
        port=int(prof.get("port") or 3306),
        database=prof.get("db") or "",
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)

def test_mysql_connection(prof: Dict[str, str]) -> Tuple[bool, str]:
    """Simple connection test (SELECT 1)."""
    try:
        engine = _mysql_engine_from_profile(prof)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True, "Connection OK."
    except OperationalError as oe:
        return False, f"OperationalError: {oe.orig}"
    except Exception as e:
        return False, f"Error: {e}"

def commit_dataframe_to_mysql(df: pd.DataFrame, table_name: str, prof: Dict[str, str]) -> int:
    """Replace table contents with the given DataFrame."""
    if df is None:
        raise RuntimeError("No dataframe to commit.")
    engine = _mysql_engine_from_profile(prof)
    with engine.begin() as conn:
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
    return len(df)

def list_mysql_tables(prof: Dict[str, str]) -> List[str]:
    """List tables for a given database/schema."""
    engine = _mysql_engine_from_profile(prof)
    q = text("SELECT table_name FROM information_schema.tables WHERE table_schema = :db ORDER BY table_name")
    with engine.connect() as conn:
        rows = conn.execute(q, {"db": prof.get("db") or ""}).fetchall()
    return [r[0] for r in rows]

def read_mysql_table(prof: Dict[str, str], table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read entire table (or up to LIMIT) as DataFrame."""
    engine = _mysql_engine_from_profile(prof)
    safe = re.sub(r"[^\w]+", "", table_name)
    sql = f"SELECT * FROM `{safe}`"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql(sql, engine)
# endregion MySQL helpers


# region AI Demo Data transforms (shared)
# ------------------------------------------------------------
# Light-weight transformation helpers used in the demo tab
# ------------------------------------------------------------
AI_MODEL_DEFAULT = _env("OPENAI_MODEL", "gpt-4.1-mini")
RANDOM_SEED = 42

def _water_like(industry: str) -> bool:
    s = industry.lower()
    return any(k in s for k in ["water", "wastewater", "wwtp", "utilities", "sewer"])

def _automotive_like(industry: str) -> bool:
    s = industry.lower()
    return any(k in s for k in ["auto", "vehicle", "oem", "tier"])

def _pharma_like(industry: str) -> bool:
    s = industry.lower()
    return any(k in s for k in ["pharma", "biotech", "life science", "medic"])

def _generic_segments() -> List[str]:
    return ["Assembly","Operations","Maintenance","Logistics","Quality","Powertrain","Body & Chassis","Automation"]

CITIES_NA = [
    "Toronto","Montreal","Vancouver","Calgary","Ottawa","Edmonton","Mississauga","Winnipeg","Hamilton","Quebec City",
    "New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","San Diego","Dallas","San Jose",
    "Austin","Jacksonville","Fort Worth","Columbus","Charlotte","San Francisco","Indianapolis","Seattle","Denver","Washington",
    "Boston","El Paso","Nashville","Detroit","Oklahoma City","Portland","Las Vegas","Memphis","Louisville","Milwaukee",
    "Baltimore","Albuquerque","Tucson","Fresno","Sacramento","Mesa","Kansas City","Atlanta","Omaha","Colorado Springs",
    "Miami","Minneapolis","Arlington","New Orleans","Wichita","Cleveland","Tampa","Bakersfield","Aurora","Anaheim",
    "Honolulu","Buffalo","Plano","Lincoln","Henderson","Chandler","Riverside","Irvine","Orlando","St. Louis"
]

COMPANY_PREFIX = {
    "water": ["Hydro","Aqua","Clear","Blue","River","Lake","Flow","Hydra","Aquam","Stream"],
    "auto": ["Auto","Moto","Drive","Motion","Gear","Torque","Axel","MotoX","Vector","Velo"],
    "pharma": ["Pharma","Bio","Medi","Gen","Thera","Nova","Viva","Helix","Cura","Vita"],
    "generic": ["Omni","Neo","Prime","North","United","Pioneer","Summit","Crown","Global","Vertex"],
}
COMPANY_SUFFIX = {
    "water": ["via","max","works","source","pure","logic","grid","line","core","flow"],
    "auto": ["cor","tron","motors","dyno","tech","labs","forge","line","grid","drive"],
    "pharma": ["can","genix","nova","thera","medica","vita","pharm","zyme","logic","cura"],
    "generic": ["corp","labs","works","group","systems","nex","logic","core","point","grid"],
}

def _company_bucket(industry: str) -> str:
    if _water_like(industry): return "water"
    if _automotive_like(industry): return "auto"
    if _pharma_like(industry): return "pharma"
    return "generic"

def gen_company_name(industry: str, use_ai: bool = True) -> str:
    """Optionally uses OpenAI to generate a fake company name, else heuristic."""
    bucket = _company_bucket(industry)
    if use_ai:
        try:
            from openai import OpenAI  # type: ignore
            api_key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
            prompt = f"Invent a concise fictional company name for the {industry} industry. One or two words max, no real trademarks, avoid 'Inc'/'LLC'. Return just the name."
            try:
                resp = client.responses.create(model=_env("OPENAI_MODEL", AI_MODEL_DEFAULT), input=prompt)
                name = (getattr(resp, "output_text", "") or "").strip()
            except Exception:
                resp = client.chat.completions.create(model=_env("OPENAI_MODEL", "gpt-4o-mini"),
                                                      messages=[{"role":"user","content":prompt}], temperature=0.8)
                name = (resp.choices[0].message.content or "").strip()
            name = re.sub(r"[^A-Za-z0-9 ]", "", name)
            if 3 <= len(name) <= 24:
                return name
        except Exception:
            pass
    p = random.choice(COMPANY_PREFIX[bucket]); s = random.choice(COMPANY_SUFFIX[bucket])
    return (p + s).title()

def get_names_with_company(industry: str, plant_count: int, region_count: int, segment_count: int, use_ai: bool = True) -> Dict[str, List[str]]:
    """Derive consistent names for plants/regions/segments for demo data."""
    random.seed(RANDOM_SEED)
    company = gen_company_name(industry, use_ai=use_ai)
    regions = ["Northern District","Southern District","Eastern District","Western District","Central District","Coastal Operations","Inland Region","Metro West","Great Lakes Region"]
    segments = ["Powertrain","Body & Chassis","Stamping","Final Assembly","Paint","Logistics","Quality"]
    cities = CITIES_NA.copy(); random.shuffle(cities)
    if plant_count > len(cities):
        extras = [f"{c} {i}" for i, c in enumerate(cities * ((plant_count // len(cities)) + 1), start=1)]
        cities = extras
    plants = [f"{company} {city}" for city in cities[:plant_count]]
    def take_cycle(lst, n, add_suffix_if_repeats=True):
        if n <= 0: return []
        out = []; i = 0
        while len(out) < n:
            out.append(lst[i % len(lst)] if lst else f"Group {i+1}")
            i += 1
        if add_suffix_if_repeats and len(set(out)) < len(out):
            out = [f"{v} ({i+1})" for i, v in enumerate(out)]
        return out
    return {"company": company,
            "plants": take_cycle(plants, plant_count, add_suffix_if_repeats=True),
            "regions": take_cycle(regions, max(region_count,1), add_suffix_if_repeats=True),
            "segments": take_cycle(_generic_segments(), max(segment_count,1), add_suffix_if_repeats=False)}

_ALNUM_RUN = re.compile(r"[A-Za-z]+|\d+|[^A-Za-z0-9]+")
def preserve_format_scramble(s: str) -> str:
    """Scramble alphanumeric strings while preserving structure/length."""
    if not isinstance(s, str) or s == "": return s
    parts = _ALNUM_RUN.findall(s)
    out = []
    for p in parts:
        if p.isdigit():
            if len(p) == 1:
                out.append(str(random.randint(0 if p == "0" else 1, 9)))
            else:
                first_min = 1 if p[0] != "0" else 0
                new_first = str(random.randint(first_min, 9))
                rest = "".join(str(random.randint(0, 9)) for _ in range(len(p)-1))
                out.append(new_first + rest)
        elif p.isalpha():
            def rot(c):
                base = ord('A') if c.isupper() else ord('a')
                return chr(base + (ord(c) - base + random.randint(1, 25)) % 26)
            out.append("".join(rot(c) for c in p))
        else:
            out.append(p)
    return "".join(out)

def _plant_code4(plant: str) -> str:
    """Derive a 4-letter code from a plant name."""
    if not isinstance(plant, str) or not plant.strip(): return "PLNT"
    words = re.findall(r"[A-Za-z]+", plant)
    if not words: return "PLNT"
    chars = [w[0].upper() for w in words if w]
    i = 1
    while len(chars) < 4:
        progressed = False
        for w in words:
            if len(w) > i:
                chars.append(w[i].upper()); progressed = True
                if len(chars) == 4: break
        if len(chars) >= 4: break
        if not progressed:
            i += 1
            if i > 6: break
    code4 = "".join(chars)[:4]
    if len(code4) < 4: code4 = (code4 + "XXXX")[:4]
    return code4

def server_name_for_plant(plant: str, seed: int = RANDOM_SEED) -> str:
    """Deterministic server name per plant."""
    code = _plant_code4(plant)
    rnd = random.Random(seed + hash(plant))
    num = rnd.randint(100, 999)
    return f"MS{code}{num}MP"

def _fake_ip_for_plant(plant: str) -> str:
    """Kept for completeness; not used by mutate_path_minimal anymore."""
    rnd = random.Random(RANDOM_SEED + hash(plant))
    choice = rnd.choice(["10", "172", "192"])
    if choice == "10":
        return f"10.{rnd.randint(0,255)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}"
    if choice == "172":
        return f"172.{rnd.randint(16,31)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}"
    return f"192.168.{rnd.randint(0,255)}.{rnd.randint(1,254)}"

def mutate_path_minimal(path: str, plant: str, server: Optional[str], per_row_server: Optional[str]) -> str:
    """
    Replace ONLY the server segment (first path segment) and keep the remainder intact.
    Specifically:
      New path = <ServerName>\<ORIGINAL_FROM_FIRST_IP_OR_SECOND_SEGMENT_ONWARD>
    Example:
      MSDUNL227MP!Dunlop10-4-10\10.4.10.207\Backplane\4
      -> <NewServerName>\10.4.10.207\Backplane\4
    """
    if not isinstance(path, str):
        path = ""

    sep = "\\" if ("\\" in path or "/" not in path) else "/"
    parts = [p for p in re.split(r"[\\/]+", path) if p]

    server_name = per_row_server or server or "SERVER01"
    if not parts:
        return server_name

    ip_re = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    ip_idx = None
    for i, seg in enumerate(parts):
        if ip_re.match(seg):
            ip_idx = i
            break

    if ip_idx is not None:
        rest = sep.join(parts[ip_idx:])
    else:
        rest = sep.join(parts[1:])

    return server_name + (sep + rest if rest else "")

def mutate_device_address(addr: str) -> str:
    """Randomize device addresses while preserving structure."""
    if not isinstance(addr, str) or addr == "": return addr
    if ":" in addr and "." in addr:
        host, port = addr.split(":", 1)
        host = ".".join(str(random.randint(1, 254)) for _ in range(4))
        try:
            port_int = int(re.sub(r"\D", "", port) or "502")
        except:
            port_int = 502
        port_int = random.choice([port_int, 44818, 2222, 1025, random.randint(1024, 65535)])
        return f"{host}:{port_int}"
    if "." in addr:
        return ".".join(str(random.randint(1, 254)) for _ in range(4))
    return preserve_format_scramble(addr)

def mutate_raai_filename(orig: str, plant: str) -> str:
    """Rewrite filename to include plant name, preserving extension."""
    if not isinstance(orig, str) or orig == "":
        return f"{plant.replace(' ', '_')}_RAAI.json"
    m = re.search(r"\.([A-Za-z0-9]{1,6})$", orig)
    ext = m.group(0) if m else ".json"
    return f"{plant.replace(' ', '_')}_RAAI{ext}"

def choose_version_for_plant(plant: str, seed_offset: int = 0) -> str:
    rnd = random.Random(RANDOM_SEED + hash(plant) + seed_offset)
    return f"{rnd.randint(1, 4)}.{rnd.randint(0, 9)}.{rnd.randint(0, 9)}"

SHUFFLE_BLOCK = [
    "ProductTypeCode","ProductCode","ProductType","VendorId","Revision","UpdatedRevision","ProductName",
    "LastUpdateDate","FirmwareLifecycleStatus","ProductNotices","ReplaceProduct","ProductServiceAdvisories",
    "FirmwareName","ReleaseNotes","DiscontinuedDate","SeriesNums","ProductDesc","CIPkeys",
    "LifecycleStatus","ReplaceCategory","ProductID"
]

def shuffle_block(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Shuffle a set of columns together to break direct correlations."""
    present = [c for c in columns if c in df.columns]
    if not present: return df
    block = df[present].copy()
    idx = list(block.index)
    random.shuffle(idx)
    block = block.reindex(idx).reset_index(drop=True)
    df = df.copy()
    df[present] = block.values
    return df

def apply_common_changes(df: pd.DataFrame, industry: str, maps: Dict, server_override: Optional[str], add_server_col: bool = False) -> pd.DataFrame:
    """Apply renaming/masking/server path changes to a DataFrame."""
    df = df.copy()
    plant_map = maps.get("plant_map", {}); region_map = maps.get("region_map", {}); segment_map = maps.get("segment_map", {})
    ai_names = {"plants": maps.get("plants", []), "regions": maps.get("regions", []), "segments": maps.get("segments", [])}
    def map_value(col: str, old: str) -> str:
        if col == "Plant": return plant_map.get(old, (ai_names["plants"][0] if ai_names["plants"] else old))
        if col == "Region": return region_map.get(old, (ai_names["regions"][0] if ai_names["regions"] else old))
        if col == "Segment": return segment_map.get(old, (ai_names["segments"][0] if ai_names["segments"] else old))
        return old
    for col in ["Plant","Region","Segment"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda v: map_value(col, v))
    new_plants = set(df["Plant"].astype(str).unique()) if "Plant" in df.columns else set()
    plant_server_map = {p: server_name_for_plant(p) for p in new_plants if p}
    if add_server_col and "ServerName" not in df.columns:
        df.insert(0, "ServerName", "")
    # Only replace server segment; preserve IP and remainder
    for col in [c for c in ["Path","AssetPath","Location"] if c in df.columns]:
        df[col] = df.apply(lambda r: mutate_path_minimal(
            str(r[col]), str(r.get("Plant","")), server_override,
            (None if server_override else plant_server_map.get(str(r.get("Plant","")), "SERVER01"))
        ), axis=1)
        if add_server_col:
            df["ServerName"] = df.apply(lambda r: (plant_server_map.get(str(r.get("Plant","")), "SERVER01") if not server_override else server_override), axis=1)
    for col in ["SerialNumber","ProgramName"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(preserve_format_scramble)
    if "DeviceAddress" in df.columns:
        df["DeviceAddress"] = df["DeviceAddress"].astype(str).apply(mutate_device_address)
    if "RAAIFileName" in df.columns:
        df["RAAIFileName"] = df.apply(lambda r: mutate_raai_filename(str(r["RAAIFileName"]), str(r.get("Plant","Plant"))), axis=1)
    if "external_id" in df.columns and plant_map:
        def repl_external(s: str) -> str:
            if not isinstance(s, str): return s
            out = s
            for old, new in plant_map.items():
                out = re.sub(re.escape(old), new, out)
            return out
        df["external_id"] = df["external_id"].astype(str).apply(repl_external)
    return df

def build_name_maps_from_mysql_tables(engine, db: str, table_list: List[str], industry: str, use_ai: bool) -> Dict[str, Any]:
    """Scan selected MySQL tables to build plant/region/segment rename maps."""
    plants, regions, segments = set(), set(), set()
    for t in table_list:
        safe = re.sub(r"[^\w]+", "", t)
        try:
            df_head = pd.read_sql(text(f"SELECT * FROM `{safe}` LIMIT 5000"), engine)
        except Exception:
            continue
        if "Plant" in df_head.columns: plants.update(df_head["Plant"].dropna().astype(str).unique().tolist())
        if "Region" in df_head.columns: regions.update(df_head["Region"].dropna().astype(str).unique().tolist())
        if "Segment" in df_head.columns: segments.update(df_head["Segment"].dropna().astype(str).unique().tolist())
    names = get_names_with_company(industry, max(len(plants), 6), max(len(regions), 4), max(len(segments), 4), use_ai=use_ai)
    names["plant_map"] = {old: names["plants"][i % len(names["plants"])] for i, old in enumerate(sorted(plants))} if plants else {}
    names["region_map"] = {old: names["regions"][i % len(names["regions"])] for i, old in enumerate(sorted(regions))} if regions else {}
    names["segment_map"] = {old: names["segments"][i % len(names["segments"])] for i, old in enumerate(sorted(segments))} if segments else {}
    return names
# endregion AI Demo Data transforms (shared)


# region Session State
# ------------------------------------------------------------
# Centralized initialization of state variables
# ------------------------------------------------------------
ss = st.session_state
# CDF
ss.setdefault("client", None)
ss.setdefault("conn_key", "")
ss.setdefault("spaces", [])
ss.setdefault("models", [])
ss.setdefault("views", [])
ss.setdefault("dfs_by_viewkey", {})        # eid@ver -> DataFrame
ss.setdefault("combined_df", None)
ss.setdefault("current_viewkey", "__ALL__")
ss.setdefault("table_names_by_viewkey", {})
ss.setdefault("dm_ctx", {})  # space, dm_eid, dm_ver
# Logs / profiles
ss.setdefault("logs", [])
ss.setdefault("prefill", {})
# AI demo tab state
ss.setdefault("mysql_source_prof", {})
ss.setdefault("mysql_dest_prof", {})
ss.setdefault("source_tables", [])
ss.setdefault("selected_source_tables", [])
ss.setdefault("demo_table_names", {})       # src_table -> dest_table
ss.setdefault("preview_table", None)
# Viewer (Data Explorer) state
ss.setdefault("viewer_mysql_prof", {})
ss.setdefault("viewer_tables", [])
ss.setdefault("viewer_selected_tables", [])
ss.setdefault("viewer_current_table", None)
# endregion Session State


# region Tabs
tabs = st.tabs([
    "🔌 Connect & Download",
    "📊 Data Explorer",
    "🤖 AI Demo Data",
    "⬆️ DataMosaix Upload",
    "🪵 Logs",
    "ℹ️ About",
])
# endregion Tabs


# region Connect & Download
# ------------------------ Connect & Download ------------------------
with tabs[0]:
    st.header("Connect")
    st.caption("Pick a profile (env/saved) or fill in fields, then click **Connect**.")

    env_defaults = {
        "host": _env("CARGILL_FTDM_HOST"),
        "project": _env("CARGILL_FTDM_PROJECT"),
        "token_url": _env("CARGILL_FTDM_TOKENURL"),
        "client_id": _env("CARGILL_FTDM_CLIENTID"),
        "client_secret": _env("CARGILL_FTDM_CLIENTSECRET"),
        "scopes": _env("CARGILL_FTDM_SCOPES", "user_impersonation"),
    }
    env_profs = env_profiles()
    saved_profs = load_json_profiles(PROFILE_STORE_PATH_CDF)

    colp1, colp2, colp3 = st.columns([2, 1, 1])
    with colp1:
        options = ["(none)"] + [f"[env] {n}" for n in sorted(env_profs.keys())] + [f"[saved] {n}" for n in sorted(saved_profs.keys())]
        picked = st.selectbox("Load CDF profile", options, key="cdf_prof_pick")
    with colp2:
        if st.button("Load", use_container_width=True, key="cdf_prof_load"):
            if picked.startswith("[env] "):
                name = picked.replace("[env] ", "", 1)
                ss.prefill = env_profs.get(name, {})
                log(f"Loaded env profile '{name}'.")
            elif picked.startswith("[saved] "):
                name = picked.replace("[saved] ", "", 1)
                ss.prefill = saved_profs.get(name, {})
                log(f"Loaded saved profile '{name}'.")
            else:
                ss.prefill = {}
    with colp3:
        can_delete = picked.startswith("[saved] ")
        if st.button("Delete", use_container_width=True, disabled=not can_delete, key="cdf_prof_delete"):
            name = picked.replace("[saved] ", "", 1)
            delete_json_profile(PROFILE_STORE_PATH_CDF, name)
            st.success(f"Deleted profile '{name}'.")
            log(f"Deleted profile '{name}'.")

    def get_val(name, fallback):
        return ss.prefill.get(name) or env_defaults.get(name) or fallback

    with st.form("conn"):
        c1, c2 = st.columns(2)
        with c1:
            host = st.text_input("Host", value=get_val("host", ""), placeholder="https://api.cognitedata.com", key="cdf_host").strip()
            project = st.text_input("Project", value=get_val("project", ""), placeholder="your-cdf-project", key="cdf_project").strip()
            token_url = st.text_input("Token URL", value=get_val("token_url", ""),
                                      placeholder="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token", key="cdf_token").strip()
            scopes = st.text_input("Scopes (comma-separated)", value=get_val("scopes", "user_impersonation"), key="cdf_scopes").strip()
        with c2:
            client_id = st.text_input("Client ID", value=get_val("client_id", ""), key="cdf_client_id").strip()
            show = st.toggle("Show client secret", value=False, key="cdf_show_secret")
            client_secret = st.text_input("Client Secret", value=get_val("client_secret", ""),
                                          type="default" if show else "password", key="cdf_client_secret").strip()
            max_rows = st.number_input("Max rows to fetch (per view)", min_value=1, value=50000, step=1000, key="cdf_max_rows")

        sp1, sp2 = st.columns([3, 1])
        with sp1:
            prof_name = st.text_input("Save as CDF profile (persists to /data)", placeholder="e.g., prod-cdf", key="cdf_prof_name")
        with sp2:
            if st.form_submit_button("Save", use_container_width=True) and prof_name:
                try:
                    save_json_profile(PROFILE_STORE_PATH_CDF, prof_name, {
                        "host": host, "project": project, "token_url": token_url,
                        "client_id": client_id, "client_secret": client_secret, "scopes": scopes,
                    })
                    st.success(f"Saved profile '{prof_name}'.")
                    log(f"Saved profile '{prof_name}'.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
                    log(f"ERROR save profile: {e}")

        errs = []
        if not host.startswith("http"): errs.append("Host must start with http(s).")
        if not project: errs.append("Project is required.")
        if not token_url.startswith("http"): errs.append("Token URL must start with http(s).")
        if not client_id: errs.append("Client ID is required.")
        if not client_secret: errs.append("Client Secret is required.")

        connect = st.form_submit_button("Connect", type="primary", disabled=bool(errs), use_container_width=True)
    if errs:
        st.warning(" • " + "\n • ".join(errs))

    if connect and not errs:
        try:
            ss.client = build_client(host, project, token_url, client_id, client_secret, scopes)
            ss.conn_key = f"{host}|{project}|{token_url}|{client_id}|{scopes}"
            _ = list_spaces(ss.client, ss.conn_key)
            st.success(f"Connected to '{project}'.")
            log("Connected.")
            ss.spaces, ss.models, ss.views = [], [], []
            ss.dfs_by_viewkey = {}
            ss.combined_df = None
            ss.current_viewkey = "__ALL__"
            ss.table_names_by_viewkey = {}
            ss.dm_ctx = {}
        except Exception as e:
            st.error(str(e)); log(f"ERROR connect: {e}")

    st.divider()
    st.header("Download")
    st.caption("Flow: **Load Spaces** → choose Space → **Load Models** → choose Model → **Load Views** → choose one or many → **Fetch data**.")

    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Load Spaces", use_container_width=True, disabled=ss.client is None, key="cdf_load_spaces"):
            try:
                ss.spaces = list_spaces(ss.client, ss.conn_key)
                log(f"Loaded {len(ss.spaces)} spaces.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR load spaces: {e}")
    with cols[1]:
        st.caption("Choose your Space here →")
        space = st.selectbox("Space", options=ss.spaces or ["(none)"], label_visibility="collapsed", key="cdf_space_pick")

    cols = st.columns([1, 3])
    with cols[0]:
        load_models_disabled = (not space) or space == "(none)"
        if st.button("Load Models", use_container_width=True, disabled=load_models_disabled, key="cdf_load_models"):
            try:
                ss.models = list_models_latest(ss.client, ss.conn_key, space)
                log(f"Loaded {len(ss.models)} models for space '{space}'.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR load models: {e}")
    with cols[1]:
        st.caption("Choose your Model here →")
        dm_display = [f"{eid} v{ver}" for (eid, ver) in ss.models] or ["(none)"]
        dm_choice = st.selectbox("Model", options=dm_display, label_visibility="collapsed", key="cdf_model_pick")
        dm_eid, dm_ver = ("", "")
        if " v" in dm_choice:
            dm_eid, dm_ver = dm_choice.rsplit(" v", 1)

    cols = st.columns([1, 3])
    with cols[0]:
        load_views_disabled = not (dm_eid and dm_ver and dm_choice != "(none)")
        if st.button("Load Views", use_container_width=True, disabled=load_views_disabled, key="cdf_load_views"):
            try:
                ss.views = list_views_for_model(ss.client, ss.conn_key, space, dm_eid, dm_ver)
                log(f"Loaded {len(ss.views)} views for model '{dm_eid}' v{dm_ver}.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR load views: {e}")
    with cols[1]:
        st.caption("Choose your Views here → (multi-select)")
        view_display = [f"{eid} v{ver}" for (eid, ver) in ss.views] or []
        selected_views = st.multiselect("Views", options=view_display, default=view_display[:1], key="cdf_view_multiselect")

    cols = st.columns(2)
    with cols[0]:
        fetch_enabled = bool(ss.client and space and dm_eid and dm_ver and selected_views)
        if st.button("Fetch data", type="primary", use_container_width=True, disabled=not fetch_enabled, key="cdf_fetch"):
            try:
                frames = []
                fetched_keys = []
                with st.spinner("Fetching instances…"):
                    for choice in selected_views:
                        v_eid, v_ver = choice.rsplit(" v", 1)
                        key = f"{v_eid}@{v_ver}"
                        df_part = fetch_view_dataframe(ss.client, ss.conn_key, space, dm_eid, dm_ver, v_eid, v_ver, max_rows=max_rows)
                        ss.dfs_by_viewkey[key] = df_part
                        frames.append(df_part)
                        fetched_keys.append(key)
                        ss.table_names_by_viewkey.setdefault(key, _sanitize_table_name(v_eid))
                if frames:
                    ss.combined_df = pd.concat(frames, ignore_index=True, sort=False)
                    ss.current_viewkey = "__ALL__"
                    ss.dm_ctx = {"space": space, "dm_eid": dm_eid, "dm_ver": dm_ver}
                    st.success(f"Loaded {len(ss.combined_df):,} rows from {len(frames)} view(s).")
                    log(f"Fetched {len(ss.combined_df):,} rows from {len(frames)} views: {', '.join(fetched_keys)}")
                    st.download_button(
                        "Download CSV (all views)",
                        ss.combined_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dm_eid}_views.csv",
                        mime="text/csv",
                        key="cdf_download_all"
                    )
                else:
                    st.warning("No views selected.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR fetch: {e}")
    with cols[1]:
        if st.button("Clear data", use_container_width=True, key="cdf_clear"):
            ss.dfs_by_viewkey = {}
            ss.combined_df = None
            ss.current_viewkey = "__ALL__"
            ss.table_names_by_viewkey = {}
            ss.dm_ctx = {}
            log("Cleared data.")

    if ss.dfs_by_viewkey:
        st.info(f"Cached views: {len(ss.dfs_by_viewkey)} • Rows (combined): {len(ss.combined_df) if isinstance(ss.combined_df, pd.DataFrame) else 0:,}")

        # ------ Commit fetched CDF data to MySQL ------
        with st.expander("Commit fetched CDF data to MySQL", expanded=False):
            mysql_envs = mysql_env_profiles()
            mysql_saved = load_json_profiles(PROFILE_STORE_PATH_MYSQL)
            mysql_defaults = _mysql_env_default()

            p1, p2, p3, _ = st.columns([2,1,1,1])
            with p1:
                dest_opts = ["(none)"] + [f"[env] {n}" for n in sorted(mysql_envs.keys())] + [f"[saved] {n}" for n in sorted(mysql_saved.keys())]
                dest_pick = st.selectbox("Load destination profile", dest_opts, key="cdf_dest_pick")
            with p2:
                if st.button("Load Dest", use_container_width=True, key="cdf_dest_load"):
                    if dest_pick.startswith("[env] "):
                        name = dest_pick.replace("[env] ", "", 1)
                        ss.mysql_dest_prof = mysql_envs.get(name, {})
                        log(f"Loaded MySQL dest env profile '{name}'.")
                    elif dest_pick.startswith("[saved] "):
                        name = dest_pick.replace("[saved] ", "", 1)
                        ss.mysql_dest_prof = mysql_saved.get(name, {})
                        log(f"Loaded MySQL dest saved profile '{name}'.")
                    else:
                        ss.mysql_dest_prof = {}
            with p3:
                if st.button("Test Dest", use_container_width=True, key="cdf_dest_test"):
                    ok, msg = test_mysql_connection(ss.mysql_dest_prof or mysql_defaults)
                    (st.success if ok else st.error)(msg)

            d1, d2, d3, d4, d5 = st.columns([2,1,1,1,1])
            with d1:
                dest_host = st.text_input("Dest Host", value=(ss.mysql_dest_prof or mysql_defaults).get("host",""), key="cdf_dest_host")
            with d2:
                dest_port = st.text_input("Dest Port", value=str((ss.mysql_dest_prof or mysql_defaults).get("port","3306")), key="cdf_dest_port")
            with d3:
                dest_user = st.text_input("Dest User", value=(ss.mysql_dest_prof or mysql_defaults).get("user",""), key="cdf_dest_user")
            with d4:
                showp_dest = st.toggle("Show password", value=False, key="cdf_show_dest_pwd")
                dest_password = st.text_input("Dest Password", value=(ss.mysql_dest_prof or mysql_defaults).get("password",""),
                                              type="default" if showp_dest else "password", key="cdf_dest_password")
            with d5:
                dest_db = st.text_input("Dest Database", value=(ss.mysql_dest_prof or mysql_defaults).get("db",""), key="cdf_dest_db")

            sv1, sv2 = st.columns([3,1])
            with sv1:
                dest_save_name = st.text_input("Save destination profile as", placeholder="e.g., reporting-db", key="cdf_dest_save_name")
            with sv2:
                if st.button("Save Dest Profile", use_container_width=True, key="cdf_dest_save_btn") and dest_save_name:
                    save_json_profile(PROFILE_STORE_PATH_MYSQL, dest_save_name, {
                        "host": dest_host, "port": dest_port, "user": dest_user, "password": dest_password, "db": dest_db
                    })
                    st.success(f"Saved MySQL destination profile '{dest_save_name}'"); log(f"Saved dest profile '{dest_save_name}'")

            dest_prof_current = {"host": dest_host, "port": dest_port, "user": dest_user, "password": dest_password, "db": dest_db}

            st.subheader("Table mapping")
            # Combined table name
            default_combined = ss.get("combined_table_name") or _sanitize_table_name(f"{ss.dm_ctx.get('dm_eid','combined')}_all_views")
            ss["combined_table_name"] = st.text_input("Destination table for ALL (combined)", value=default_combined, key="cdf_combined_tbl")

            # Per-view edit area
            with st.expander("Edit per-view table names (for 'Commit EACH view')", expanded=False):
                cols_map = st.columns(2)
                for i, k in enumerate(sorted(ss.dfs_by_viewkey.keys())):
                    with cols_map[i % 2]:
                        default_out = _sanitize_table_name(k.split("@",1)[0])
                        ss.table_names_by_viewkey[k] = st.text_input(f"{k}", value=ss.table_names_by_viewkey.get(k, default_out), key=f"cdf_map_{k}")

            # Commit buttons
            cmt1, cmt2 = st.columns(2)
            with cmt1:
                if st.button("Commit ALL (combined) to MySQL", type="primary", use_container_width=True, key="cdf_commit_combined"):
                    try:
                        if ss.combined_df is None or ss.combined_df.empty:
                            st.error("No combined data to commit.")
                        else:
                            rows = commit_dataframe_to_mysql(ss.combined_df, ss.get("combined_table_name") or "combined_views", dest_prof_current)
                            st.success(f"Wrote {rows:,} rows to `{dest_db}.{ss.get('combined_table_name')}`")
                            log(f"CDF Commit combined -> {dest_db}.{ss.get('combined_table_name')} ({rows})")
                    except OperationalError as oe:
                        st.error(f"MySQL error: {oe.orig}")
                        log(f"ERROR commit combined: {oe.orig}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        log(f"ERROR commit combined: {e}")
            with cmt2:
                if st.button("Commit EACH view separately", use_container_width=True, key="cdf_commit_each"):
                    try:
                        results = []
                        for k, dfk in ss.dfs_by_viewkey.items():
                            if dfk is None or dfk.empty:
                                continue
                            tname = ss.table_names_by_viewkey.get(k, _sanitize_table_name(k.split("@",1)[0]))
                            rows = commit_dataframe_to_mysql(dfk, tname, dest_prof_current)
                            results.append((k, tname, rows))
                        if results:
                            lines = [f"- `{k}` → `{dest_db}.{t}`: {r:,} rows" for (k,t,r) in results]
                            st.success("Committed:\n" + "\n".join(lines))
                            log("CDF Commit EACH: " + "; ".join([f"{k}->{t}:{r}" for (k,t,r) in results]))
                        else:
                            st.info("No non-empty views to commit.")
                    except OperationalError as oe:
                        st.error(f"MySQL error: {oe.orig}")
                        log(f"ERROR commit each: {oe.orig}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        log(f"ERROR commit each: {e}")
# endregion Connect & Download


# region Data Explorer
# ------------------------ Data Explorer (Generic MySQL viewer) ------------------------
with tabs[1]:
    st.header("MySQL Table Viewer")
    st.caption("Connect to a MySQL database, pick one or more tables, and explore with filters.")

    mysql_envs = mysql_env_profiles()
    mysql_saved = load_json_profiles(PROFILE_STORE_PATH_MYSQL)
    mysql_defaults = _mysql_env_default()

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        viewer_pick = st.selectbox("Load viewer profile", ["(none)"] + [f"[env] {n}" for n in sorted(mysql_envs.keys())] + [f"[saved] {n}" for n in sorted(mysql_saved.keys())], key="viewer_prof_pick")
    with c2:
        if st.button("Load Viewer", use_container_width=True, key="viewer_prof_load"):
            if viewer_pick.startswith("[env] "):
                name = viewer_pick.replace("[env] ", "", 1)
                ss.viewer_mysql_prof = mysql_envs.get(name, {})
                log(f"Loaded viewer env profile '{name}'.")
            elif viewer_pick.startswith("[saved] "):
                name = viewer_pick.replace("[saved] ", "", 1)
                ss.viewer_mysql_prof = mysql_saved.get(name, {})
                log(f"Loaded viewer saved profile '{name}'.")
            else:
                ss.viewer_mysql_prof = {}
    with c3:
        if st.button("Test Viewer", use_container_width=True, key="viewer_test"):
            ok, msg = test_mysql_connection(ss.viewer_mysql_prof or mysql_defaults)
            (st.success if ok else st.error)(msg)

    v1, v2, v3, v4, v5 = st.columns([2,1,1,1,1])
    with v1:
        v_host = st.text_input("Host", value=(ss.viewer_mysql_prof or mysql_defaults).get("host",""), key="viewer_host")
    with v2:
        v_port = st.text_input("Port", value=str((ss.viewer_mysql_prof or mysql_defaults).get("port","3306")), key="viewer_port")
    with v3:
        v_user = st.text_input("User", value=(ss.viewer_mysql_prof or mysql_defaults).get("user",""), key="viewer_user")
    with v4:
        v_show = st.toggle("Show password", value=False, key="viewer_show_pwd")
        v_password = st.text_input("Password", value=(ss.viewer_mysql_prof or mysql_defaults).get("password",""),
                                   type="default" if v_show else "password", key="viewer_password")
    with v5:
        v_db = st.text_input("Database", value=(ss.viewer_mysql_prof or mysql_defaults).get("db",""), key="viewer_db")

    sv1, sv2 = st.columns([3,1])
    with sv1:
        v_save_name = st.text_input("Save viewer profile as", placeholder="e.g., ops-db", key="viewer_save_name")
    with sv2:
        if st.button("Save Viewer Profile", use_container_width=True, key="viewer_save_btn") and v_save_name:
            save_json_profile(PROFILE_STORE_PATH_MYSQL, v_save_name, {
                "host": v_host, "port": v_port, "user": v_user, "password": v_password, "db": v_db
            })
            st.success(f"Saved MySQL viewer profile '{v_save_name}'"); log(f"Saved viewer profile '{v_save_name}'")

    viewer_prof_current = {"host": v_host, "port": v_port, "user": v_user, "password": v_password, "db": v_db}

    if st.button("Load tables", key="viewer_load_tables"):
        try:
            ss.viewer_tables = list_mysql_tables(viewer_prof_current)
            st.success(f"Found {len(ss.viewer_tables)} tables.")
            log(f"Viewer tables: {len(ss.viewer_tables)}")
        except Exception as e:
            st.error(f"Failed to list tables: {e}")
            log(f"ERROR viewer list tables: {e}")

    if ss.viewer_tables:
        st.caption("Pick tables to explore (choose one to display at a time below)")
        ss.viewer_selected_tables = st.multiselect("Tables", options=ss.viewer_tables, default=ss.viewer_tables[:1], key="viewer_table_multiselect")
        lim1, lim2 = st.columns([2,1])
        with lim1:
            current_table = st.selectbox("Current table to display", options=ss.viewer_selected_tables or ["(none)"], key="viewer_current_table_select")
        with lim2:
            max_rows_view = st.number_input("Max rows", min_value=10, max_value=200000, value=5000, step=1000, key="viewer_max_rows")

        if current_table and current_table != "(none)":
            try:
                df_view = read_mysql_table(viewer_prof_current, current_table, limit=int(max_rows_view))
                st.caption(f"Showing up to {len(df_view):,} rows from `{v_db}.{current_table}`")
                filtered = filter_dataframe(df_view, key_prefix=f"viewer_{v_db}_{current_table}")
                st.dataframe(filtered, use_container_width=True, height=520)

                # ---------- Quick stats (under the table) ----------
                with st.expander("Quick stats", expanded=False):
                    n_rows = len(filtered)
                    n_cols = len(filtered.columns)
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Rows (filtered)", f"{n_rows:,}")
                    with c2: st.metric("Columns", f"{n_cols:,}")
                    with c3:
                        approx_mem = filtered.memory_usage(deep=True).sum()
                        st.metric("Approx. memory", f"{approx_mem/1024/1024:.2f} MB")

                    if n_rows > 0:
                        rows = []
                        for col in filtered.columns:
                            s = filtered[col]
                            dtype = str(s.dtype)
                            non_null = int(s.notna().sum())
                            missing = int(n_rows - non_null)
                            missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0
                            unique = int(s.nunique(dropna=True))
                            samp = ", ".join(map(str, s.dropna().astype(str).unique()[:3]))
                            rows.append({
                                "column": col,
                                "dtype": dtype,
                                "non_null": non_null,
                                "missing_pct": round(missing_pct, 2),
                                "unique": unique,
                                "sample_values": samp
                            })
                        summary_df = pd.DataFrame(rows).set_index("column").sort_values("unique", ascending=False)
                        st.markdown("**Per-column summary**")
                        st.dataframe(summary_df, use_container_width=True, height=260)

                        num_df = filtered.select_dtypes(include="number")
                        if not num_df.empty:
                            st.markdown("**Numeric columns (describe)**")
                            st.dataframe(num_df.describe().T, use_container_width=True, height=260)

                        dt_rows = []
                        for col in filtered.columns:
                            s = filtered[col]
                            s_dt = pd.to_datetime(s, errors="coerce")
                            if s_dt.notna().any():
                                dt_rows.append({
                                    "column": col,
                                    "min": s_dt.min(),
                                    "max": s_dt.max(),
                                    "non_null": int(s_dt.notna().sum())
                                })
                        if dt_rows:
                            st.markdown("**Datetime columns**")
                            st.dataframe(pd.DataFrame(dt_rows).set_index("column"), use_container_width=True, height=220)

                        if "Plant" in filtered.columns:
                            st.markdown("**Counts by Plant (top 20)**")
                            plant_counts = filtered["Plant"].astype(str).value_counts().head(20)
                            pc1, pc2 = st.columns([1,1])
                            with pc1:
                                st.bar_chart(plant_counts)
                            with pc2:
                                st.dataframe(plant_counts.rename_axis("Plant").to_frame("count"), use_container_width=True, height=260)

                st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{current_table}_filtered.csv", mime="text/csv", key="viewer_dl")
            except Exception as e:
                st.error(f"Failed to read table: {e}")
# endregion Data Explorer


# region AI Demo Data
# ------------------------ AI Demo Data (MySQL → MySQL) ------------------------
with tabs[2]:
    st.header("AI Demo Data (MySQL → MySQL)")
    st.caption("Select a **source** DB + tables, choose an **industry**, (optionally) use OpenAI naming, then write transformed tables to a **destination** DB.")

    # Source profile
    st.subheader("Source MySQL")
    mysql_envs = mysql_env_profiles()
    mysql_saved = load_json_profiles(PROFILE_STORE_PATH_MYSQL)
    mysql_defaults = _mysql_env_default()

    c1, c2, c3, _ = st.columns([2,1,1,1])
    with c1:
        src_opts = ["(none)"] + [f"[env] {n}" for n in sorted(mysql_envs.keys())] + [f"[saved] {n}" for n in sorted(mysql_saved.keys())]
        src_pick = st.selectbox("Load source profile", src_opts, key="demo_src_pick")
    with c2:
        if st.button("Load Source", use_container_width=True, key="demo_src_load"):
            if src_pick.startswith("[env] "):
                name = src_pick.replace("[env] ", "", 1)
                st.session_state.mysql_source_prof = mysql_envs.get(name, {})
                log(f"Loaded MySQL source env profile '{name}'.")
            elif src_pick.startswith("[saved] "):
                name = src_pick.replace("[saved] ", "", 1)
                st.session_state.mysql_source_prof = mysql_saved.get(name, {})
                log(f"Loaded MySQL source saved profile '{name}'.")
            else:
                st.session_state.mysql_source_prof = {}
    with c3:
        if st.button("Test Source", use_container_width=True, key="demo_src_test"):
            ok, msg = test_mysql_connection(st.session_state.mysql_source_prof or mysql_defaults)
            (st.success if ok else st.error)(msg)

    # Inline editable fields for source
    s1, s2, s3, s4, s5 = st.columns([2,1,1,1,1])
    with s1:
        src_host = st.text_input("Source Host", value=(st.session_state.mysql_source_prof or mysql_defaults).get("host",""), key="demo_src_host")
    with s2:
        src_port = st.text_input("Source Port", value=str((st.session_state.mysql_source_prof or mysql_defaults).get("port","3306")), key="demo_src_port")
    with s3:
        src_user = st.text_input("Source User", value=(st.session_state.mysql_source_prof or mysql_defaults).get("user",""), key="demo_src_user")
    with s4:
        showp_src = st.toggle("Show password (src)", value=False, key="demo_src_show_pwd")
        src_password = st.text_input("Source Password", value=(st.session_state.mysql_source_prof or mysql_defaults).get("password",""),
                                     type="default" if showp_src else "password", key="demo_src_password")
    with s5:
        src_db = st.text_input("Source Database", value=(st.session_state.mysql_source_prof or mysql_defaults).get("db",""), key="demo_src_db")

    # Save source profile
    s6, s7 = st.columns([3,1])
    with s6:
        src_save_name = st.text_input("Save source profile as", placeholder="e.g., prod-source", key="demo_src_save_name")
    with s7:
        if st.button("Save Source Profile", use_container_width=True, key="demo_src_save_btn") and src_save_name:
            save_json_profile(PROFILE_STORE_PATH_MYSQL, src_save_name, {
                "host": src_host, "port": src_port, "user": src_user, "password": src_password, "db": src_db
            })
            st.success(f"Saved MySQL source profile '{src_save_name}'"); log(f"Saved src profile '{src_save_name}'")

    # List tables
    src_prof_current = {"host": src_host, "port": src_port, "user": src_user, "password": src_password, "db": src_db}
    if st.button("Load tables from source DB", key="demo_src_load_tables"):
        try:
            st.session_state.source_tables = list_mysql_tables(src_prof_current)
            st.success(f"Found {len(st.session_state.source_tables)} tables.")
            log(f"Source tables: {len(st.session_state.source_tables)}")
        except Exception as e:
            st.error(f"Failed to list source tables: {e}")
            log(f"ERROR list tables: {e}")

    if st.session_state.source_tables:
        st.caption("Pick one or more source tables to transform")
        st.session_state.selected_source_tables = st.multiselect("Source tables", options=st.session_state.source_tables, default=st.session_state.source_tables[:3], key="demo_src_table_multiselect")

    st.divider()
    st.subheader("Transformation settings")

    colx1, colx2, colx3 = st.columns([2,1,1])
    with colx1:
        industry = st.text_input("Target industry", value="Water & Wastewater",
                                 help="Used to generate consistent Regions/Segments/Plant names; you can type anything.", key="demo_industry")
    with colx2:
        use_openai = st.checkbox("Use OpenAI for naming", value=True,
                                 help="Reads API key from OPENAI_API or OPENAI_API_KEY in .env. Falls back to heuristics if missing.", key="demo_use_openai")
    with colx3:
        add_server_col = st.checkbox("Add ServerName column (if missing)", value=False, key="demo_add_server")

    if use_openai and not (os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")):
        st.warning("OPENAI_API (or OPENAI_API_KEY) not set. Will fall back to heuristic naming.")

    # Destination profile
    st.subheader("Destination MySQL")
    d1, d2, d3, d4, d5 = st.columns([2,1,1,1,1])
    dest_prof_default = _mysql_env_default()
    with d1:
        dest_host = st.text_input("Dest Host", value=(st.session_state.mysql_dest_prof or dest_prof_default).get("host",""), key="demo_dest_host")
    with d2:
        dest_port = st.text_input("Dest Port", value=str((st.session_state.mysql_dest_prof or dest_prof_default).get("port","3306")), key="demo_dest_port")
    with d3:
        dest_user = st.text_input("Dest User", value=(st.session_state.mysql_dest_prof or dest_prof_default).get("user",""), key="demo_dest_user")
    with d4:
        showp_dest2 = st.toggle("Show password (dest)", value=False, key="demo_dest_show_pwd")
        dest_password = st.text_input("Dest Password", value=(st.session_state.mysql_dest_prof or dest_prof_default).get("password",""),
                                      type="default" if showp_dest2 else "password", key="demo_dest_password")
    with d5:
        dest_db = st.text_input("Dest Database", value=(st.session_state.mysql_dest_prof or dest_prof_default).get("db",""), key="demo_dest_db")

    dd1, dd2 = st.columns([3,1])
    with dd1:
        dest_save_name = st.text_input("Save destination profile as", placeholder="e.g., demo-dest", key="demo_dest_save_name")
    with dd2:
        if st.button("Save Destination Profile", use_container_width=True, key="demo_dest_save_btn") and dest_save_name:
            save_json_profile(PROFILE_STORE_PATH_MYSQL, dest_save_name, {
                "host": dest_host, "port": dest_port, "user": dest_user,
                "password": dest_password, "db": dest_db
            })
            st.success(f"Saved MySQL destination profile '{dest_save_name}'"); log(f"Saved dest profile '{dest_save_name}'")

    # Table name mapping UI
    if st.session_state.selected_source_tables:
        st.markdown("**Output table names (one per selected source table)**")
        cols_map = st.columns(2)
        for i, t in enumerate(sorted(st.session_state.selected_source_tables)):
            with cols_map[i % 2]:
                default_out = _sanitize_table_name(f"{t}_demo")
                st.session_state.demo_table_names[t] = st.text_input(f"{t}", value=st.session_state.demo_table_names.get(t, default_out), key=f"demo_tbl_{t}")

    # Preview
    pv1, pv2 = st.columns([2,1])
    with pv1:
        preview_rows = st.number_input("Preview rows (per table)", min_value=5, max_value=2000, value=50, step=5, key="demo_preview_rows")
    with pv2:
        do_shuffle = st.checkbox("Shuffle product-ish fields (if present)", value=False, key="demo_shuffle")

    if st.session_state.selected_source_tables:
        if st.button("Preview first table transform", key="demo_preview_btn"):
            try:
                engine_src = _mysql_engine_from_profile(src_prof_current)
                names_map = build_name_maps_from_mysql_tables(engine_src, src_db, st.session_state.selected_source_tables, industry, use_ai=use_openai)
                first = st.session_state.selected_source_tables[0]
                df_src = read_mysql_table(src_prof_current, first).head(int(preview_rows))
                df_tx = apply_common_changes(df_src, industry, names_map, server_override=None, add_server_col=add_server_col)
                if do_shuffle:
                    df_tx = shuffle_block(df_tx, SHUFFLE_BLOCK)
                st.success(f"Preview: {first}")
                st.dataframe(df_tx, use_container_width=True, height=420)
                st.session_state.preview_table = first
            except Exception as e:
                st.error(f"Preview failed: {e}")
                log(f"ERROR preview: {e}")

    # Run transform & write
    if st.session_state.selected_source_tables and st.button("Transform & write ALL selected to destination", type="primary", key="demo_write_all"):
        try:
            engine_src = _mysql_engine_from_profile(src_prof_current)
            names_map = build_name_maps_from_mysql_tables(engine_src, src_db, st.session_state.selected_source_tables, industry, use_ai=use_openai)
            dest_prof_current2 = {"host": dest_host, "port": dest_port, "user": dest_user, "password": dest_password, "db": dest_db}
            results = []
            with st.spinner("Transforming and writing tables…"):
                for t in st.session_state.selected_source_tables:
                    out_name = st.session_state.demo_table_names.get(t, _sanitize_table_name(f"{t}_demo"))
                    df_src_full = read_mysql_table(src_prof_current, t)
                    df_tx_full = apply_common_changes(df_src_full, industry, names_map, server_override=None, add_server_col=add_server_col)
                    if do_shuffle:
                        df_tx_full = shuffle_block(df_tx_full, SHUFFLE_BLOCK)
                    rows = commit_dataframe_to_mysql(df_tx_full, out_name, dest_prof_current2)
                    results.append((t, out_name, rows))
            lines = [f"- `{t}` → `{dest_db}.{o}`: {r:,} rows" for (t, o, r) in results]
            st.success("Wrote:\n" + "\n".join(lines))
            log("AI Demo write: " + "; ".join([f"{t}->{o}:{r}" for (t,o,r) in results]))
        except OperationalError as oe:
            st.error(f"MySQL error: {oe.orig}")
            log(f"ERROR write demo: {oe.orig}")
        except Exception as e:
            st.error(f"Failed: {e}")
            log(f"ERROR write demo: {e}")
# endregion AI Demo Data


# region DataMosaix Upload
# ------------------------ DataMosaix Upload ------------------------
with tabs[3]:
    # --- local helpers (scoped to this tab) ---
    def _get_view_properties(_client: CogniteClient, space: str, view_eid: str, view_ver: str) -> List[str]:
        """Return property names for a view (including inherited)."""
        v = dm.ViewId(space=space, external_id=view_eid, version=str(view_ver))
        vd = _client.data_modeling.views.retrieve(v, include_inherited_properties=True)
        if not vd:
            return []
        return list(vd[0].properties.keys())

    def _auto_map_columns(df: pd.DataFrame, props: List[str]) -> Dict[str, str]:
        """Best-effort auto-map by case-insensitive exact name."""
        col_by_lower = {c.lower(): c for c in df.columns}
        out = {}
        for p in props:
            if p.lower() in col_by_lower:
                out[p] = col_by_lower[p.lower()]
        return out

    def _save_mapped_csv(df: pd.DataFrame, out_path: Path) -> None:
        """Persist mapped payload under /data/uploads for reuse."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")

    st.header("DataMosaix Upload")
    st.caption("Prepare rows to upload into **one or many** target Views. Choose a source (CSV/MySQL/Combined), map or auto-map columns → properties, dry-run (single), or batch upload (multiple).")

    # --- guard: need a CDF client (connect in tab 1) ---
    client = st.session_state.get("client")
    conn_key = st.session_state.get("conn_key", "")
    if not client:
        st.warning("Please connect to DataMosaix in **Connect & Download** first.")
    else:
        # ---------- Target selection: Space → Model → View(s) ----------
        st.subheader("Target (where to upload)")
        t1, t2 = st.columns([1, 3])
        with t1:
            if st.button("Load Spaces (target)", use_container_width=True, key="upload_load_spaces"):
                try:
                    st.session_state.upload_spaces = list_spaces(client, conn_key)
                    st.success(f"Loaded {len(st.session_state.upload_spaces)} spaces.")
                    log(f"[Upload] Loaded spaces: count={len(st.session_state.upload_spaces)}")
                except Exception as e:
                    st.error(str(e)); log(f"[Upload][ERROR] Load spaces failed: {e}")
        with t2:
            space_u = st.selectbox(
                "Space",
                options=(st.session_state.get("upload_spaces") or st.session_state.get("spaces") or ["(none)"]),
                key="upload_space_pick",
            )

        um1, um2 = st.columns([1, 3])
        with um1:
            can_load_models = bool(space_u and space_u != "(none)")
            if st.button("Load Models (target)", use_container_width=True, disabled=not can_load_models, key="upload_load_models"):
                try:
                    st.session_state.upload_models = list_models_latest(client, conn_key, space_u)
                    st.success(f"Loaded {len(st.session_state.upload_models)} models.")
                    log(f"[Upload] Loaded models for space='{space_u}': count={len(st.session_state.upload_models)}")
                except Exception as e:
                    st.error(str(e)); log(f"[Upload][ERROR] Load models failed: {e}")
        with um2:
            dm_display_u = [f"{eid} v{ver}" for (eid, ver) in (st.session_state.get("upload_models") or [])] or ["(none)"]
            dm_choice_u = st.selectbox("Model", options=dm_display_u, key="upload_model_pick")
            dm_eid_u, dm_ver_u = ("", "")
            if " v" in dm_choice_u:
                dm_eid_u, dm_ver_u = dm_choice_u.rsplit(" v", 1)

        uv1, uv2 = st.columns([1, 3])
        with uv1:
            can_load_views = bool(dm_eid_u and dm_ver_u and dm_choice_u != "(none)")
            if st.button("Load Views (target)", use_container_width=True, disabled=not can_load_views, key="upload_load_views"):
                try:
                    st.session_state.upload_views = list_views_for_model(client, conn_key, space_u, dm_eid_u, dm_ver_u)
                    st.success(f"Loaded {len(st.session_state.upload_views)} views.")
                    log(f"[Upload] Loaded views for model='{dm_eid_u}' v{dm_ver_u}: count={len(st.session_state.upload_views)}")
                except Exception as e:
                    st.error(str(e)); log(f"[Upload][ERROR] Load views failed: {e}")
        with uv2:
            view_display_u = [f"{eid} v{ver}" for (eid, ver) in (st.session_state.get("upload_views") or [])]
            selected_views_u = st.multiselect("Target View(s)", options=(view_display_u or []), default=view_display_u[:1] if view_display_u else [], key="upload_views_multiselect")

        have_any_views = bool(selected_views_u)
        have_single_view = bool(have_any_views and len(selected_views_u) == 1)

        st.divider()

        # ---------- Source selection ----------
        st.subheader("Source (what to upload)")
        src_kind = st.radio("Pick a source", ["CSV upload", "MySQL table(s)", "Combined (from Connect & Download)"],
                            horizontal=True, key="upload_src_kind_v2")

        df_source: Optional[pd.DataFrame] = None
        log(f"[Upload] Source selected: {src_kind}")

        mysql_prof_for_upload = None
        mysql_tables_loaded: List[str] = []

        if src_kind == "CSV upload":
            file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="upload_csv_file_v2")
            if file is not None:
                try:
                    df_source = pd.read_csv(file)
                    st.success(f"Loaded CSV with {len(df_source):,} rows and {len(df_source.columns)} columns.")
                    st.dataframe(df_source.head(50), use_container_width=True, height=300)
                    log(f"[Upload] CSV loaded: rows={len(df_source)}, cols={len(df_source.columns)}, name='{getattr(file, 'name', 'uploaded.csv')}'")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}"); log(f"[Upload][ERROR] CSV read failed: {e}")

        elif src_kind == "MySQL table(s)":
            # reuse helpers and UX to load profile & tables
            mysql_envs_u = mysql_env_profiles()
            mysql_saved_u = load_json_profiles(PROFILE_STORE_PATH_MYSQL)
            mysql_defaults_u = _mysql_env_default()

            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                pick = st.selectbox("Load MySQL profile", ["(none)"] + [f"[env] {n}" for n in sorted(mysql_envs_u.keys())] +
                                    [f"[saved] {n}" for n in sorted(mysql_saved_u.keys())], key="upload_mysql_pick_v2")
            with c2:
                if st.button("Load", use_container_width=True, key="upload_mysql_load_v2"):
                    if pick.startswith("[env] "):
                        name = pick.replace("[env] ", "", 1)
                        st.session_state.upload_mysql_prof = mysql_envs_u.get(name, {})
                        log(f"[Upload] Loaded MySQL env profile '{name}'")
                    elif pick.startswith("[saved] "):
                        name = pick.replace("[saved] ", "", 1)
                        st.session_state.upload_mysql_prof = mysql_saved_u.get(name, {})
                        log(f"[Upload] Loaded MySQL saved profile '{name}'")
                    else:
                        st.session_state.upload_mysql_prof = {}; log(f"[Upload] Cleared MySQL profile selection")
            with c3:
                if st.button("Test", use_container_width=True, key="upload_mysql_test_v2"):
                    ok, msg = test_mysql_connection(st.session_state.get("upload_mysql_prof") or mysql_defaults_u)
                    (st.success if ok else st.error)(msg)
                    log(f"[Upload] MySQL test result: ok={ok}, msg='{msg}'")

            p = st.session_state.get("upload_mysql_prof") or mysql_defaults_u
            m1, m2, m3, m4, m5 = st.columns([2,1,1,1,1])
            with m1:
                u_host = st.text_input("Host", value=p.get("host",""), key="upload_mysql_host_v2")
            with m2:
                u_port = st.text_input("Port", value=str(p.get("port","3306")), key="upload_mysql_port_v2")
            with m3:
                u_user = st.text_input("User", value=p.get("user",""), key="upload_mysql_user_v2")
            with m4:
                sh = st.toggle("Show password", value=False, key="upload_mysql_showpwd_v2")
                u_pwd = st.text_input("Password", value=p.get("password",""), type="default" if sh else "password", key="upload_mysql_pwd_v2")
            with m5:
                u_db = st.text_input("Database", value=p.get("db",""), key="upload_mysql_db_v2")

            mysql_prof_for_upload = {"host": u_host, "port": u_port, "user": u_user, "password": u_pwd, "db": u_db}

            # DEFAULT max rows now 30,000
            lim1, lim2 = st.columns([2,1])
            with lim1:
                st.caption("You can load one table for single-mapping, or multiselect for batch.")
                if st.button("List tables", key="upload_mysql_list_tables_v2"):
                    try:
                        st.session_state.upload_mysql_tables = list_mysql_tables(mysql_prof_for_upload)
                        st.success(f"Found {len(st.session_state.upload_mysql_tables)} tables.")
                        log(f"[Upload] MySQL list tables: db='{u_db}', count={len(st.session_state.upload_mysql_tables)}")
                    except Exception as e:
                        st.error(f"Failed to list tables: {e}"); log(f"[Upload][ERROR] MySQL list tables failed: {e}")
            with lim2:
                lim_rows = st.number_input("Max rows", min_value=10, max_value=500000, value=30000, step=1000, key="upload_mysql_limit_v2")

            mysql_tables_loaded = st.session_state.get("upload_mysql_tables") or []
            if mysql_tables_loaded:
                # Multiselect tables for batch OR pick one for single
                st.markdown("**Select table(s)**")
                picked_tables = st.multiselect("MySQL table(s)", options=mysql_tables_loaded,
                                               default=mysql_tables_loaded[:1], key="upload_mysql_tables_pick_v2")

                # If exactly one table is chosen, show its preview for single-mapping flow
                if picked_tables and len(picked_tables) == 1:
                    tbl = picked_tables[0]
                    try:
                        df_source = read_mysql_table(mysql_prof_for_upload, tbl, limit=int(lim_rows))
                        st.caption(f"Showing up to {len(df_source):,} rows from `{u_db}.{tbl}`")
                        st.dataframe(df_source.head(50), use_container_width=True, height=300)
                        log(f"[Upload] Read MySQL table (single preview): {u_db}.{tbl}, rows={len(df_source)}, limit={int(lim_rows)}")
                    except Exception as e:
                        st.error(f"Failed to read table: {e}"); log(f"[Upload][ERROR] Read MySQL table failed: {e}")

        else:  # Combined (from previous tab)
            if isinstance(st.session_state.get("combined_df"), pd.DataFrame) and not st.session_state.combined_df.empty:
                df_source = st.session_state.combined_df.copy()
                st.info(f"Using combined dataset from Connect & Download: {len(df_source):,} rows, {len(df_source.columns)} columns.")
                st.dataframe(df_source.head(50), use_container_width=True, height=300)
                log(f"[Upload] Using combined dataset: rows={len(df_source)}, cols={len(df_source.columns)}")
            else:
                st.warning("No combined data found. Fetch data in **Connect & Download** first.")
                log("[Upload] Combined dataset not available")

        st.divider()

        # ---------- SINGLE mapping UI (manual mapping + dry-run + upload) ----------
        st.subheader("Single mapping")
        st.caption("Use this to configure one source payload → one target View with manual mapping and dry-run.")

        # For single mapping, require exactly one target view selected
        single_view_ok = have_single_view
        if not single_view_ok:
            st.info("Pick exactly one **Target View** above to use single-mapping.")
        else:
            view_choice_single = selected_views_u[0]
            view_eid_u, view_ver_u = view_choice_single.rsplit(" v", 1)

            # Build df_source and props if we have a source
            if df_source is None or df_source.empty:
                st.info("Choose/Load a source first.")
            else:
                # read view properties
                try:
                    props = _get_view_properties(client, space_u, view_eid_u, view_ver_u)
                    log(f"[Upload] (single) Loaded view properties: view='{view_eid_u}' v{view_ver_u}, props={len(props)}")
                except Exception as e:
                    st.error(f"Failed to read view schema: {e}"); log(f"[Upload][ERROR] (single) Read view schema failed: {e}")
                    props = []

                if props:
                    # external_id handling (DEFAULT: use external_id column if present)
                    st.markdown("**Node identifier (external_id)**")
                    default_mode = "Use a column" if "external_id" in [c.lower() for c in df_source.columns] else "Generate"
                    id_mode = st.radio("How to set `external_id`?",
                                       ["Use a column", "Generate"],
                                       horizontal=True, key="upload_id_mode_single",
                                       index=(0 if default_mode == "Use a column" else 1))
                    external_id_series = None
                    if id_mode == "Use a column":
                        # preselect the real column named (case-insensitive) external_id if it exists
                        cols_ci = {c.lower(): c for c in df_source.columns}
                        preselect_col = cols_ci.get("external_id")
                        id_col = st.selectbox("Column for external_id", options=list(df_source.columns),
                                              index=(list(df_source.columns).index(preselect_col) if preselect_col in df_source.columns else 0),
                                              key="upload_id_col_single")
                        external_id_series = df_source[id_col].astype(str)
                    else:
                        prefix = st.text_input("Generated external_id prefix", value=f"{view_eid_u}_", key="upload_id_prefix_single")
                        external_id_series = pd.Series([f"{prefix}{i+1}" for i in range(len(df_source))], index=df_source.index, dtype="string")

                    # property mapping table
                    auto_map = _auto_map_columns(df_source, props)
                    st.markdown("**Property mappings** (skip any you don't want to set)")
                    mapped: Dict[str, Optional[str]] = {}
                    cols_map = st.columns(2)
                    for i, p in enumerate(sorted(props)):
                        with cols_map[i % 2]:
                            options = ["(skip)"] + list(df_source.columns)
                            default_idx = 0
                            if p in auto_map:
                                try:
                                    default_idx = 1 + list(df_source.columns).index(auto_map[p])
                                except ValueError:
                                    default_idx = 0
                            mapped[p] = st.selectbox(f"{p}", options=options, index=default_idx, key=f"upload_map_single_{p}")

                    # preview / dry-run
                    if st.button("Dry-run mapping", type="primary", key="upload_dryrun_single"):
                        included_props = [p for p, c in mapped.items() if c and c != "(skip)"]
                        if not included_props:
                            st.warning("No properties selected to map."); log("[Upload] (single) Dry-run aborted: 0 properties mapped")
                        else:
                            out_df = pd.DataFrame(index=df_source.index)
                            out_df["external_id"] = external_id_series.astype(str)
                            for p in included_props:
                                out_df[p] = df_source[mapped[p]]

                            # ✨ persist for the upload step
                            st.session_state["upload_out_df"] = out_df
                            st.session_state["upload_target"] = {"space": space_u, "view_eid": view_eid_u, "view_ver": str(view_ver_u)}

                            empties = out_df["external_id"].astype(str).str.len() == 0
                            if empties.any():
                                st.warning(f"{int(empties.sum())} row(s) have empty external_id and will fail to upsert.")

                            st.success(f"Prepared {len(out_df):,} rows with {len(included_props)} mapped properties.")
                            st.dataframe(out_df.head(50), use_container_width=True, height=320)
                            log(f"[Upload] (single) Dry-run prepared: rows={len(out_df)}, props_mapped={len(included_props)}, "
                                f"id_mode='{id_mode}', empty_ids={int(empties.sum()) if 'empties' in locals() else 0}, "
                                f"target={space_u}:{view_eid_u} v{view_ver_u}")

                            # Save-to-volume & download
                            fname = f"cdf_upload_{space_u}_{view_eid_u}_v{view_ver_u}.csv".replace("/", "-")
                            out_path = Path("/data/uploads") / fname
                            col_save, col_dl = st.columns([1,1])
                            with col_save:
                                if st.button("Save mapped CSV to /data/uploads", key="upload_save_csv_single"):
                                    try:
                                        _save_mapped_csv(out_df, out_path)
                                        st.success(f"Saved → {out_path}"); log(f"[Upload] (single) Saved mapped CSV: {out_path}")
                                    except Exception as e:
                                        st.error(f"Failed to save: {e}"); log(f"[Upload][ERROR] (single) Save CSV failed: {e}")
                            with col_dl:
                                st.download_button(
                                    "Download mapped CSV",
                                    out_df.to_csv(index=False).encode("utf-8"),
                                    file_name=f"{fname}",
                                    mime="text/csv",
                                    key="upload_download_csv_single",
                                )

                # --- Upload (single, uses last dry-run result) ---
                if st.session_state.get("upload_out_df") is not None and st.session_state.get("upload_target"):
                    st.caption("Upload uses the mapped payload from the dry-run above.")
                    sync_delete_missing_single = st.checkbox(
                        "Sync mode: delete existing nodes in the target View that are **not** present in this payload (by `external_id`)",
                        value=False,
                        key="upload_sync_single",
                        help="Dangerous: will remove nodes not included here before uploading."
                    )
                    batch_size_single = st.number_input("Batch size (single)", min_value=100, max_value=5000, value=1000, step=100, key="upload_batch_single")
                    confirm_single = st.checkbox("I understand this will write data into CDF (single).", value=False, key="upload_confirm_single")
                    if st.button("Upload mapped rows to CDF (single)", type="primary", disabled=not confirm_single, key="upload_go_single"):
                        out_df = st.session_state["upload_out_df"].copy()
                        tgt = st.session_state["upload_target"]
                        space_s = tgt["space"]; view_eid_s = tgt["view_eid"]; view_ver_s = tgt["view_ver"]
                        try:
                            view_id = dm.ViewId(space=space_s, external_id=view_eid_s, version=str(view_ver_s))

                            # optional sync delete
                            if sync_delete_missing_single:
                                log(f"[Upload] (single) Sync mode: fetching existing external_ids for {space_s}:{view_eid_s} v{view_ver_s}")
                                existing_ids: set[str] = set()
                                for inst in client.data_modeling.instances.list(instance_type="node", sources=[view_id], limit=None):
                                    eid = getattr(inst, "external_id", None)
                                    if eid:
                                        existing_ids.add(str(eid))
                                payload_ids = set(out_df["external_id"].astype(str).tolist())
                                to_delete = sorted(existing_ids - payload_ids)
                                log(f"[Upload] (single) Sync diff: existing={len(existing_ids)}, payload={len(payload_ids)}, to_delete={len(to_delete)}")
                                if to_delete:
                                    del_prog = st.progress(0, text="Deleting nodes not in payload…")
                                    for i in range(0, len(to_delete), 1000):
                                        chunk = to_delete[i:i+1000]
                                        try:
                                            client.data_modeling.instances.delete(
                                                nodes=[dm.NodeId(space=space_s, external_id=e) for e in chunk]
                                            )
                                            log(f"[Upload] (single) Deleted nodes {i+1}-{min(i+1000, len(to_delete))}")
                                        except Exception as e:
                                            log(f"[Upload][ERROR] (single) Delete batch {i+1}-{min(i+1000, len(to_delete))} failed: {e}")
                                        finally:
                                            del_prog.progress(min(1.0, (i+1000)/max(len(to_delete),1)))

                            # build NodeApply list
                            nodes = []
                            prop_cols = [c for c in out_df.columns if c != "external_id"]
                            for _, row in out_df.iterrows():
                                ext_id = str(row["external_id"])
                                props = {c: (None if pd.isna(row[c]) else row[c]) for c in prop_cols}
                                nodes.append(
                                    dm.NodeApply(
                                        space=space_s,
                                        external_id=ext_id,
                                        sources=[dm.NodeOrEdgeData(source=view_id, properties=props)],
                                    )
                                )

                            total = len(nodes)
                            log(f"[Upload] (single) Start upload: total={total}, batch={int(batch_size_single)}, target={space_s}:{view_eid_s} v{view_ver_s}")
                            prog = st.progress(0, text="Uploading…")
                            errs: List[str] = []
                            for i in range(0, total, int(batch_size_single)):
                                start = i + 1
                                end = min(i + int(batch_size_single), total)
                                batch = nodes[i:end]
                                try:
                                    client.data_modeling.instances.apply(nodes=batch)
                                    log(f"[Upload] (single) Batch applied: rows {start}-{end}")
                                except Exception as e:
                                    msg = f"Rows {start}-{end}: {e}"
                                    errs.append(msg); log(f"[Upload][ERROR] (single) {msg}")
                                finally:
                                    prog.progress(min(1.0, end / max(total, 1)))

                            if errs:
                                st.error("Finished with errors:\n" + "\n".join(errs[:10]) + ("\n… (more)" if len(errs) > 10 else ""))
                                log(f"[Upload] (single) Completed with errors: batches_with_errors={len(errs)}")
                            else:
                                st.success(f"Uploaded {total} node(s) to {space_s}:{view_eid_s} v{view_ver_s}.")
                                log(f"[Upload] (single) Completed successfully: total={total}")
                        except Exception as e:
                            st.error(f"Upload failed: {e}"); log(f"[Upload][ERROR] (single) Upload failed: {e}")

        st.divider()

        # ---------- BATCH mode (multi: MySQL tables ↔ multiple Views, auto-mapped) ----------
        st.subheader("Batch upload (auto-map)")
        st.caption("Select **multiple MySQL tables** and **multiple Views**. We’ll auto-map by exact name (case-insensitive) and use `external_id` column by default if present, else generate.")

        can_batch = (src_kind == "MySQL table(s)") and bool(mysql_prof_for_upload) and bool(st.session_state.get("upload_mysql_tables_pick_v2"))
        if not have_any_views:
            st.info("Pick one or more **Target View(s)** above to use batch upload.")
        elif not (src_kind == "MySQL table(s)"):
            st.info("Batch mode currently supports **MySQL table(s)** sources.")
        elif not can_batch:
            st.info("Select at least one MySQL table above (in the MySQL section).")
        else:
            picked_tables = st.session_state.get("upload_mysql_tables_pick_v2") or []
            pairing_mode = "Zip tables ↔ views" if len(picked_tables) == len(selected_views_u) and len(picked_tables) > 1 else "All tables → first selected view"
            pairing = st.selectbox("Pairing strategy", ["Zip tables ↔ views", "All tables → first selected view"], index=(0 if pairing_mode=="Zip tables ↔ views" else 1), key="upload_batch_pairing")

            sync_delete_missing_batch = st.checkbox(
                "Sync mode for each mapping: delete existing nodes not in the table payload (by `external_id`)",
                value=False,
                key="upload_sync_batch",
                help="Dangerous: before upload, will remove nodes not present in that table's payload for the target View."
            )
            batch_size_multi = st.number_input("Batch size (each mapping)", min_value=100, max_value=5000, value=1000, step=100, key="upload_batch_multi")
            confirm_multi = st.checkbox("I understand this will write data into CDF (batch).", value=False, key="upload_confirm_multi")

            if st.button("Run batch upload (auto-map)", type="primary", disabled=not confirm_multi, key="upload_go_batch"):
                try:
                    # Build mapping pairs (table -> view)
                    if pairing == "Zip tables ↔ views":
                        if len(picked_tables) != len(selected_views_u):
                            st.error("Zip requires the same number of tables and views."); log("[Upload][batch] Zip failed: count mismatch")
                            st.stop()
                        pairs = list(zip(picked_tables, selected_views_u))
                    else:
                        # all tables to first selected view
                        if not selected_views_u:
                            st.error("No target view selected."); log("[Upload][batch] No views selected")
                            st.stop()
                        pairs = [(t, selected_views_u[0]) for t in picked_tables]

                    total_pairs = len(pairs)
                    st.info(f"Batch mappings: {total_pairs}")
                    overall_prog = st.progress(0, text="Batch progress…")
                    errs_global: List[str] = []
                    engine_msg = f"{mysql_prof_for_upload.get('host')}:{mysql_prof_for_upload.get('port')}/{mysql_prof_for_upload.get('db')}"
                    log(f"[Upload][batch] Start: pairs={total_pairs}, mysql={engine_msg}, max_rows={int(st.session_state.get('upload_mysql_limit_v2', 30000))}")

                    # Iterate pairs
                    for idx, (tbl, view_choice) in enumerate(pairs, start=1):
                        try:
                            # read table
                            limit_rows = int(st.session_state.get("upload_mysql_limit_v2", 30000))
                            df_tbl = read_mysql_table(mysql_prof_for_upload, tbl, limit=limit_rows)

                            # extract target view parts
                            v_eid, v_ver = view_choice.rsplit(" v", 1)
                            v_props = _get_view_properties(client, space_u, v_eid, v_ver)

                            # external_id default: use 'external_id' column if present
                            cols_ci = {c.lower(): c for c in df_tbl.columns}
                            if "external_id" in cols_ci:
                                ext_ids = df_tbl[cols_ci["external_id"]].astype(str)
                            else:
                                prefix = f"{v_eid}_"
                                ext_ids = pd.Series([f"{prefix}{i+1}" for i in range(len(df_tbl))], index=df_tbl.index, dtype="string")

                            # auto map properties present in table
                            amap = _auto_map_columns(df_tbl, v_props)
                            included_props = list(amap.keys())
                            out_df = pd.DataFrame(index=df_tbl.index)
                            out_df["external_id"] = ext_ids.astype(str)
                            for p in included_props:
                                out_df[p] = df_tbl[amap[p]]

                            # optional sync delete
                            view_id = dm.ViewId(space=space_u, external_id=v_eid, version=str(v_ver))
                            if sync_delete_missing_batch:
                                existing_ids: set[str] = set()
                                for inst in client.data_modeling.instances.list(instance_type="node", sources=[view_id], limit=None):
                                    eid = getattr(inst, "external_id", None)
                                    if eid:
                                        existing_ids.add(str(eid))
                                payload_ids = set(out_df["external_id"].astype(str).tolist())
                                to_delete = sorted(existing_ids - payload_ids)
                                if to_delete:
                                    log(f"[Upload][batch] {idx}/{total_pairs} Sync delete: {len(to_delete)} nodes for {space_u}:{v_eid} v{v_ver}")
                                    for j in range(0, len(to_delete), 1000):
                                        chunk = to_delete[j:j+1000]
                                        try:
                                            client.data_modeling.instances.delete(
                                                nodes=[dm.NodeId(space=space_u, external_id=e) for e in chunk]
                                            )
                                        except Exception as e:
                                            errs_global.append(f"[{tbl} → {v_eid}] delete {j+1}-{min(j+1000,len(to_delete))}: {e}")

                            # build nodes and upload in batches
                            nodes = []
                            prop_cols = [c for c in out_df.columns if c != "external_id"]
                            for _, row in out_df.iterrows():
                                ext_id = str(row["external_id"])
                                props = {c: (None if pd.isna(row[c]) else row[c]) for c in prop_cols}
                                nodes.append(
                                    dm.NodeApply(
                                        space=space_u,
                                        external_id=ext_id,
                                        sources=[dm.NodeOrEdgeData(source=view_id, properties=props)],
                                    )
                                )

                            total = len(nodes)
                            log(f"[Upload][batch] {idx}/{total_pairs} Upload start: table='{tbl}', rows={total}, view='{v_eid}' v{v_ver}")
                            for k in range(0, total, int(batch_size_multi)):
                                start = k + 1
                                end = min(k + int(batch_size_multi), total)
                                batch = nodes[k:end]
                                try:
                                    client.data_modeling.instances.apply(nodes=batch)
                                except Exception as e:
                                    errs_global.append(f"[{tbl} → {v_eid}] rows {start}-{end}: {e}")

                            log(f"[Upload][batch] {idx}/{total_pairs} Upload done: table='{tbl}' → view='{v_eid}' ({total} rows)")
                        except Exception as e:
                            errs_global.append(f"[{tbl}] failed: {e}")
                        finally:
                            overall_prog.progress(min(1.0, idx / max(total_pairs, 1)))

                    if errs_global:
                        st.error("Batch finished with errors:\n" + "\n".join(errs_global[:12]) + ("\n… (more)" if len(errs_global) > 12 else ""))
                        log(f"[Upload][batch] Completed with errors: {len(errs_global)} issue(s)")
                    else:
                        st.success(f"Batch complete: {total_pairs} mapping(s) executed.")
                        log(f"[Upload][batch] Completed successfully: pairs={total_pairs}")
                except Exception as e:
                    st.error(f"Batch failed: {e}"); log(f"[Upload][ERROR][batch] {e}")
# endregion DataMosaix Upload


# region Logs
# ------------------------ Logs ------------------------
with tabs[4]:
    st.header("Logs")
    log_text = "\n".join(st.session_state.logs[-500:]) if st.session_state.logs else "No logs yet."
    st.text_area("Activity", value=log_text, height=240, label_visibility="collapsed")
# endregion Logs


# region About
# ------------------------ About ------------------------
with tabs[5]:
    st.header("About")
    st.caption(f"Version {APP_VERSION}")
    st.subheader("Changelog")
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("Changelog not bundled in this image. Add `COPY CHANGELOG.md ./` to your Dockerfile or bind-mount the project.")
# endregion About
