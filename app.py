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

# ============================================================
# region Version / Branding
# ============================================================
APP_VERSION = "1.3.0"  # bump to 1.3.0 on release
APP_TITLE = "Convergix DataMosaix View Explorer"
APP_ICON_PATH = "assets/convergix_logo.png"

# Upload/Apply parameters
UPLOAD_CHUNK_SIZE = int(os.getenv("CDF_UPLOAD_CHUNK_SIZE", "1000"))  # batch size for node apply

_icon = None
try:
    _icon = Image.open(APP_ICON_PATH)
except Exception:
    pass

st.set_page_config(page_title=f"{APP_TITLE} · v{APP_VERSION}", page_icon=_icon if _icon else "🧭", layout="wide")

left, right = st.columns([1, 8])
with left:
    if _icon:
        st.image(_icon, width=60)
with right:
    st.title(APP_TITLE)
    st.caption(f"Version {APP_VERSION}")
# endregion Version / Branding
# ============================================================


# ============================================================
# region Utilities / Env / Logging
# ============================================================
def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()

def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    st.session_state.setdefault("logs", [])
    st.session_state.logs.append(f"{_ts()} {msg}")

# persistent profile stores (bind-mounted volume at /data)
PROFILE_STORE_PATH_CDF = _env("PROFILE_STORE_PATH", "/data/profiles.json")
PROFILE_STORE_PATH_MYSQL = _env("MYSQL_PROFILE_STORE_PATH", "/data/mysql_profiles.json")

def _path(p: str) -> Path:
    pp = Path(p)
    pp.parent.mkdir(parents=True, exist_ok=True)
    return pp

def load_json_profiles(path: str) -> Dict[str, Dict[str, str]]:
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
    data = load_json_profiles(path)
    data[name] = profile
    _path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def delete_json_profile(path: str, name: str):
    data = load_json_profiles(path)
    if name in data:
        del data[name]
        _path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def env_profiles(prefix_list_var: str = "DATAMOSAIX_PROFILES", var_prefix: str = "DMX_") -> Dict[str, Dict[str, str]]:
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
# endregion Utilities / Env / Logging
# ============================================================


# ============================================================
# region Cognite CDF client + caching
# ============================================================
def build_client(host_url: str, project: str, token_url: str, client_id: str, client_secret: str, scopes_csv: str) -> CogniteClient:
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
    models = _client.data_modeling.data_models.list(limit=None, all_versions=False)
    return sorted({m.space for m in models})

@st.cache_data(show_spinner=False)
def list_models_latest(_client: CogniteClient, conn_key: str, space: str) -> List[Tuple[str, str]]:
    models = _client.data_modeling.data_models.list(limit=None, all_versions=False)
    return [(m.external_id, str(m.version)) for m in models if m.space == space]

@st.cache_data(show_spinner=False)
def list_views_for_model(_client: CogniteClient, conn_key: str, space: str, dm_eid: str, dm_ver: str) -> List[Tuple[str, str]]:
    dmid = dm.DataModelId(space=space, external_id=dm_eid, version=str(dm_ver))
    details = _client.data_modeling.data_models.retrieve(dmid, inline_views=True)
    if not details:
        return []
    model = details[0]
    if not getattr(model, "views", None):
        return []
    return [(v.external_id, str(v.version)) for v in model.views]

def flatten_properties_from_instance(inst) -> Dict[str, Any]:
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
def fetch_view_dataframe(_client: CogniteClient, conn_key: str, space: str, dm_eid: str, dm_ver: str, view_eid: str, view_ver: str, max_rows: Optional[int]) -> pd.DataFrame:
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
# endregion Cognite CDF client + caching
# ============================================================


# ============================================================
# region Filter helpers (stable keys + richer operators)
# ============================================================
def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_datetime_series(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)

def _as_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def filter_dataframe(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Stable, richer filtering UI.
    - key_prefix must be stable across reruns for the same table/view.
    - Offers Equals / Not Equals / Contains / Not Contains for text.
    - Numeric: Equals / Not Equals / Range.
    - Datetime: On date / Not on date / Between.
    """
    with st.expander("Filters", expanded=False):
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
# endregion Filter helpers
# ============================================================


# ============================================================
# region MySQL helpers
# ============================================================
def _mysql_env_default() -> Dict[str, str]:
    return {
        "host": _env("MYSQL_HOST", "host.docker.internal"),
        "port": _env("MYSQL_PORT", "3306"),
        "user": _env("MYSQL_USER", ""),
        "password": _env("MYSQL_PASSWORD", ""),
        "db": _env("MYSQL_DB", ""),
    }

def _sanitize_table_name(name: str) -> str:
    s = (name or "cdf_view").strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    return s or "cdf_view"

def _mysql_engine_from_profile(prof: Dict[str, str]):
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
    engine = _mysql_engine_from_profile(prof)
    q = text("SELECT table_name FROM information_schema.tables WHERE table_schema = :db ORDER BY table_name")
    with engine.connect() as conn:
        rows = conn.execute(q, {"db": prof.get("db") or ""}).fetchall()
    return [r[0] for r in rows]

def read_mysql_table(prof: Dict[str, str], table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    engine = _mysql_engine_from_profile(prof)
    safe = re.sub(r"[^\w]+", "", table_name)
    sql = f"SELECT * FROM `{safe}`"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql(sql, engine)
# endregion MySQL helpers
# ============================================================


# ============================================================
# region Industry-aware naming & taxonomy (improved with AI flag)
# ============================================================
AI_MODEL_DEFAULT = _env("OPENAI_MODEL", "gpt-4.1-mini")
RANDOM_SEED = 42

CITIES_NA = [
    "Toronto","Montreal","Vancouver","Calgary","Ottawa","Edmonton","Mississauga","Winnipeg","Hamilton","Quebec City",
    "New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","San Diego","Dallas","San Jose",
    "Austin","Jacksonville","Fort Worth","Columbus","Charlotte","San Francisco","Indianapolis","Seattle","Denver","Washington",
    "Boston","El Paso","Nashville","Detroit","Oklahoma City","Portland","Las Vegas","Memphis","Louisville","Milwaukee",
    "Baltimore","Albuquerque","Tucson","Fresno","Sacramento","Mesa","Kansas City","Atlanta","Omaha","Colorado Springs",
    "Miami","Minneapolis","Arlington","New Orleans","Wichita","Cleveland","Tampa","Bakersfield","Aurora","Anaheim",
    "Honolulu","Buffalo","Plano","Lincoln","Henderson","Chandler","Riverside","Irvine","Orlando","St. Louis"
]

# --- Company name helpers (kept for non-municipal industries) ---
COMPANY_PREFIX = {
    "water":   ["Hydro","Aqua","Clear","Blue","River","Lake","Flow","Hydra","Aquam","Stream"],
    "auto":    ["Auto","Moto","Drive","Motion","Gear","Torque","Axel","Vector","Velo","Forge"],
    "pharma":  ["Pharma","Bio","Medi","Gen","Thera","Nova","Viva","Helix","Cura","Vita"],
    "generic": ["Omni","Neo","Prime","North","United","Pioneer","Summit","Crown","Global","Vertex"],
}
COMPANY_SUFFIX = {
    "water":   ["via","max","works","source","pure","logic","grid","line","core","flow"],
    "auto":    ["cor","tron","motors","dyno","tech","labs","forge","line","grid","drive"],
    "pharma":  ["can","genix","nova","thera","medica","vita","pharm","zyme","logic","cura"],
    "generic": ["corp","labs","works","group","systems","nex","logic","core","point","grid"],
}

def _company_bucket(industry: str) -> str:
    s = (industry or "").lower()
    if any(k in s for k in ["water", "wastewater", "wwtp", "utilities", "sewer"]): return "water"
    if any(k in s for k in ["auto", "vehicle", "oem", "tier"]): return "auto"
    if any(k in s for k in ["pharma", "biotech", "life science", "medic"]): return "pharma"
    return "generic"

def gen_company_name(industry: str, use_ai: bool = True) -> Tuple[str, bool]:
    """
    Returns (company_name, ai_used)
    """
    used_ai = False
    bucket = _company_bucket(industry)
    if use_ai:
        try:
            from openai import OpenAI  # type: ignore
            api_key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
            prompt = (
                f"Invent a concise fictional company/operator name for the {industry} industry. "
                "One or two words max, no real trademarks, avoid 'Inc'/'LLC'. Return just the name."
            )
            try:
                resp = client.responses.create(model=_env("OPENAI_MODEL", AI_MODEL_DEFAULT), input=prompt)
                name = (getattr(resp, "output_text", "") or "").strip()
                used_ai = bool(name)
            except Exception:
                resp = client.chat.completions.create(
                    model=_env("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                name = (resp.choices[0].message.content or "").strip()
                used_ai = bool(name)
            name = re.sub(r"[^A-Za-z0-9 ]", "", name)
            if 3 <= len(name) <= 24:
                return name, used_ai
        except Exception:
            pass
    return (random.choice(COMPANY_PREFIX[bucket]) + random.choice(COMPANY_SUFFIX[bucket])).title(), False

_WATER_REGIONS = [
    "Northern Basin","Southern Basin","Eastern Basin","Western Basin","Great Lakes District",
    "Coastal Operations","River Valleys","Inland Basin","Metro Water Region"
]
_WATER_SEGMENTS = [
    "Pump Stations","Water Treatment","Wastewater Treatment","Distribution",
    "Lift Stations","SCADA","Reservoirs","Intake"
]
_AUTO_REGIONS = [
    "North Assembly Zone","South Assembly Zone","Powertrain Region","Body & Chassis Region","Logistics Hub"
]
_AUTO_SEGMENTS = [
    "Powertrain","Body & Chassis","Stamping","Final Assembly","Paint","Logistics","Quality"
]
_PHARMA_REGIONS = [
    "Northern Labs","Southern Labs","Bioprocess Region","Fill-Finish Region","Utilities Region"
]
_PHARMA_SEGMENTS = [
    "Formulation","Fill-Finish","Packaging","Utilities","Quality","Labs","Bioreactors","Cleanroom"
]
GENERIC_REGIONS = [
    "Northern District","Southern District","Eastern District","Western District","Central District",
    "Coastal Operations","Inland Region","Metro West","Great Lakes Region"
]
GENERIC_SEGMENTS = [
    "Assembly","Operations","Maintenance","Logistics","Quality","Automation","R&D"
]

WATER_FACILITY_TYPES = [
    "Pump Station","Wastewater Treatment Plant","Water Treatment Plant","Lift Station",
    "Reservoir","Intake","Groundwater Well","Storage Tank","Booster Station"
]

def _ai_suggest_list(industry: str, kind: str, fallback: List[str], desired: int) -> Tuple[List[str], bool]:
    """
    Returns (items, ai_used)
    """
    if desired <= 0:
        return [], False
    api_key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback[:desired], False
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        prompt = (
            f"Propose concise {kind} names for a {industry} organization relevant to industrial automation. "
            f"Return a comma-separated list of up to {max(desired,4)} items, no explanations."
        )
        try:
            resp = client.responses.create(model=_env("OPENAI_MODEL", AI_MODEL_DEFAULT), input=prompt)
            text = (getattr(resp, "output_text", "") or "")
            used = bool(text.strip())
        except Exception:
            resp = client.chat.completions.create(
                model=_env("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            text = resp.choices[0].message.content or ""
            used = bool(text.strip())
        items = [re.sub(r"\s+", " ", x).strip() for x in text.split(",") if x.strip()]
        seen, uniq = set(), []
        for x in items:
            k = x.lower()
            if k not in seen:
                uniq.append(x)
                seen.add(k)
        return (uniq or fallback)[:desired], used
    except Exception:
        return fallback[:desired], False

def _choose_regions_segments(industry: str, n_regions: int, n_segments: int, use_ai: bool) -> Tuple[List[str], List[str], bool]:
    """
    Returns (regions, segments, ai_used)
    """
    bucket = _company_bucket(industry)
    if bucket == "water":
        regions_fallback = _WATER_REGIONS
        segments_fallback = _WATER_SEGMENTS
    elif bucket == "auto":
        regions_fallback = _AUTO_REGIONS
        segments_fallback = _AUTO_SEGMENTS
    elif bucket == "pharma":
        regions_fallback = _PHARMA_REGIONS
        segments_fallback = _PHARMA_SEGMENTS
    else:
        regions_fallback = GENERIC_REGIONS
        segments_fallback = GENERIC_SEGMENTS

    ai_used_any = False
    if use_ai:
        regions, used_r = _ai_suggest_list(industry, "regional group", regions_fallback, max(n_regions, 1))
        segments, used_s = _ai_suggest_list(industry, "operational segment", segments_fallback, max(n_segments, 1))
        ai_used_any = used_r or used_s
    else:
        regions = regions_fallback[:max(n_regions, 1)]
        segments = segments_fallback[:max(n_segments, 1)]

    def take_cycle(lst: List[str], n: int) -> List[str]:
        if n <= 0: return []
        out, i = [], 0
        while len(out) < n:
            out.append(lst[i % len(lst)] if lst else f"Group {i+1}")
            i += 1
        if len(set(out)) < len(out):
            out = [f"{v} ({i+1})" for i, v in enumerate(out)]
        return out

    return take_cycle(regions, n_regions), take_cycle(segments, n_segments), ai_used_any

def _water_plant_names(count: int, rng: random.Random) -> List[str]:
    cities = CITIES_NA.copy(); rng.shuffle(cities)
    fac = WATER_FACILITY_TYPES.copy(); rng.shuffle(fac)
    names = []
    for i in range(count):
        city = cities[i % len(cities)]
        ftype = fac[i % len(fac)]
        ordinal = (i % 9) + 1
        names.append(f"City of {city} {ftype} {ordinal}")
    return names

def _generic_plant_names(company: str, count: int, rng: random.Random) -> List[str]:
    cities = CITIES_NA.copy(); rng.shuffle(cities)
    if count > len(cities):
        extras = [f"{c} {i+1}" for i, c in enumerate(cities * ((count // len(cities)) + 1))]
        cities = extras
    return [f"{company} {cities[i]}" for i in range(count)]

def get_names_with_company(industry: str, plant_count: int, region_count: int, segment_count: int, use_ai: bool = True) -> Dict[str, Any]:
    """
    Returns dict including:
      - company, plants, regions, segments
      - ai_used: True iff OpenAI produced any of the naming (company or lists)
    """
    rng = random.Random(RANDOM_SEED)
    company, used_c = gen_company_name(industry, use_ai=use_ai)
    regions, segments, used_rs = _choose_regions_segments(industry, max(region_count, 1), max(segment_count, 1), use_ai=use_ai)
    if _company_bucket(industry) == "water":
        plants = _water_plant_names(max(plant_count, 1), rng)
    else:
        plants = _generic_plant_names(company, max(plant_count, 1), rng)
    return {
        "company": company,
        "plants": plants[:plant_count],
        "regions": regions[:region_count],
        "segments": segments[:segment_count],
        "ai_used": bool(used_c or used_rs),
    }
# endregion Industry-aware naming & taxonomy (improved with AI flag)
# ============================================================


# ============================================================
# region Mutators & transforms (paths, devices, shuffle, apply)
# ============================================================
_ALNUM_RUN = re.compile(r"[A-Za-z]+|\d+|[^A-Za-z0-9]+")
def preserve_format_scramble(s: str) -> str:
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
    code = _plant_code4(plant)
    rnd = random.Random(seed + hash(plant))
    num = rnd.randint(100, 999)
    return f"MS{code}{num}MP"

def mutate_path_minimal(path: str, plant: str, server: Optional[str], per_row_server: Optional[str]) -> str:
    """
    Replace ONLY the server segment (first path segment) and keep the remainder intact.
    New path = <ServerName>\<ORIGINAL_FROM_FIRST_IP_OR_SECOND_SEGMENT_ONWARD>
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
    # Kept for completeness, but NOT used anymore.
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
    """
    Apply plant/region/segment mapping and path/device tweaks.
    IMPORTANT: does NOT modify 'external_id' or 'DeviceAddress'.
    """
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

    # Replace only first path segment with server; keep IP/remainder
    for col in [c for c in ["Path","AssetPath","Location"] if c in df.columns]:
        df[col] = df.apply(lambda r: mutate_path_minimal(
            str(r[col]), str(r.get("Plant","")), server_override,
            (None if server_override else plant_server_map.get(str(r.get("Plant","")), "SERVER01"))
        ), axis=1)
        if add_server_col:
            df["ServerName"] = df.apply(lambda r: (plant_server_map.get(str(r.get("Plant","")), "SERVER01") if not server_override else server_override), axis=1)

    # scramble a few product-ish fields for demo realism
    for col in ["SerialNumber","ProgramName"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(preserve_format_scramble)

    # Intentionally do NOT touch DeviceAddress anymore.

    if "RAAIFileName" in df.columns:
        df["RAAIFileName"] = df.apply(lambda r: mutate_raai_filename(str(r["RAAIFileName"]), str(r.get("Plant","Plant"))), axis=1)

    # DO NOT touch external_id
    return df

def build_name_maps_from_mysql_tables(engine, db: str, table_list: List[str], industry: str, use_ai: bool) -> Dict[str, Any]:
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
    # include AI usage flag
    names["ai_used"] = bool(names.get("ai_used"))
    return names
# endregion Mutators & transforms
# ============================================================


# ============================================================
# region Session state
# ============================================================
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
# Quick Change state
ss.setdefault("qc_spaces", [])
ss.setdefault("qc_models", [])
ss.setdefault("qc_views", [])
# endregion Session state
# ============================================================


# ============================================================
# region Tabs setup
# ============================================================
tabs = st.tabs(["🔌 Connect & Download", "📊 Data Explorer", "🤖 AI Demo Data", "⬆️ DataMosaix Upload", "🪵 Logs", "⚡ Quick Change", "ℹ️ About"])
# endregion Tabs setup
# ============================================================


# ============================================================
# region Connect & Download
# ============================================================
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
                log(f"[Connect] Loaded env profile '{name}'.")
            elif picked.startswith("[saved] "):
                name = picked.replace("[saved] ", "", 1)
                ss.prefill = saved_profs.get(name, {})
                log(f"[Connect] Loaded saved profile '{name}'.")
            else:
                ss.prefill = {}
    with colp3:
        can_delete = picked.startswith("[saved] ")
        if st.button("Delete", use_container_width=True, disabled=not can_delete, key="cdf_prof_delete"):
            name = picked.replace("[saved] ", "", 1)
            delete_json_profile(PROFILE_STORE_PATH_CDF, name)
            st.success(f"Deleted profile '{name}'.")
            log(f"[Connect] Deleted profile '{name}'.")

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
                    log(f"[Connect] Saved profile '{prof_name}'.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
                    log(f"[Connect][ERROR] save profile: {e}")

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
            log("[Connect] Connected.")
            ss.spaces, ss.models, ss.views = [], [], []
            ss.dfs_by_viewkey = {}
            ss.combined_df = None
            ss.current_viewkey = "__ALL__"
            ss.table_names_by_viewkey = {}
            ss.dm_ctx = {}
        except Exception as e:
            st.error(str(e)); log(f"[Connect][ERROR] {e}")

    st.divider()
    st.header("Download")
    st.caption("Flow: **Load Spaces** → choose Space → **Load Models** → choose Model → **Load Views** → choose one or many → **Fetch data**.")

    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Load Spaces", use_container_width=True, disabled=ss.client is None, key="cdf_load_spaces"):
            try:
                ss.spaces = list_spaces(ss.client, ss.conn_key)
                log(f"[Download] Loaded {len(ss.spaces)} spaces.")
            except Exception as e:
                st.error(str(e)); log(f"[Download][ERROR] load spaces: {e}")
    with cols[1]:
        st.caption("Choose your Space here →")
        space = st.selectbox("Space", options=ss.spaces or ["(none)"], label_visibility="collapsed", key="cdf_space_pick")

    cols = st.columns([1, 3])
    with cols[0]:
        load_models_disabled = (not space) or space == "(none)"
        if st.button("Load Models", use_container_width=True, disabled=load_models_disabled, key="cdf_load_models"):
            try:
                ss.models = list_models_latest(ss.client, ss.conn_key, space)
                log(f"[Download] Loaded {len(ss.models)} models for space '{space}'.")
            except Exception as e:
                st.error(str(e)); log(f"[Download][ERROR] load models: {e}")
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
                log(f"[Download] Loaded {len(ss.views)} views for model '{dm_eid}' v{dm_ver}.")
            except Exception as e:
                st.error(str(e)); log(f"[Download][ERROR] load views: {e}")
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
                    log(f"[Download] Fetched {len(ss.combined_df):,} rows from {len(frames)} views: {', '.join(fetched_keys)}")
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
                st.error(str(e)); log(f"[Download][ERROR] fetch: {e}")
    with cols[1]:
        if st.button("Clear data", use_container_width=True, key="cdf_clear"):
            ss.dfs_by_viewkey = {}
            ss.combined_df = None
            ss.current_viewkey = "__ALL__"
            ss.table_names_by_viewkey = {}
            ss.dm_ctx = {}
            log("[Download] Cleared data.")

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
                        log(f"[Commit] Loaded MySQL dest env profile '{name}'.")
                    elif dest_pick.startswith("[saved] "):
                        name = dest_pick.replace("[saved] ", "", 1)
                        ss.mysql_dest_prof = mysql_saved.get(name, {})
                        log(f"[Commit] Loaded MySQL dest saved profile '{name}'.")
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
                    st.success(f"Saved MySQL destination profile '{dest_save_name}'"); log(f"[Commit] Saved dest profile '{dest_save_name}'")

            dest_prof_current = {"host": dest_host, "port": dest_port, "user": dest_user, "password": dest_password, "db": dest_db}

            st.subheader("Table mapping")
            default_combined = ss.get("combined_table_name") or _sanitize_table_name(f"{ss.dm_ctx.get('dm_eid','combined')}_all_views")
            ss["combined_table_name"] = st.text_input("Destination table for ALL (combined)", value=default_combined, key="cdf_combined_tbl")

            # ---- FIX: replace the broken nested expander with a simple header + inputs ----
            st.markdown("**Per-view table names (for 'Commit EACH view')**")
            cols_map = st.columns(2)
            for i, k in enumerate(sorted(ss.dfs_by_viewkey.keys())):
                with cols_map[i % 2]:
                    default_out = _sanitize_table_name(k.split("@",1)[0])
                    ss.table_names_by_viewkey[k] = st.text_input(f"{k}", value=ss.table_names_by_viewkey.get(k, default_out), key=f"cdf_map_{k}")

            cmt1, cmt2 = st.columns(2)
            with cmt1:
                if st.button("Commit ALL (combined) to MySQL", type="primary", use_container_width=True, key="cdf_commit_combined"):
                    try:
                        if ss.combined_df is None or ss.combined_df.empty:
                            st.error("No combined data to commit.")
                        else:
                            rows = commit_dataframe_to_mysql(ss.combined_df, ss.get("combined_table_name") or "combined_views", dest_prof_current)
                            st.success(f"Wrote {rows:,} rows to `{dest_db}.{ss.get('combined_table_name')}`")
                            log(f"[Commit] Combined -> {dest_db}.{ss.get('combined_table_name')} ({rows})")
                    except OperationalError as oe:
                        st.error(f"MySQL error: {oe.orig}")
                        log(f"[Commit][ERROR] combined: {oe.orig}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        log(f"[Commit][ERROR] combined: {e}")
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
                            log("[Commit] EACH: " + "; ".join([f"{k}->{t}:{r}" for (k,t,r) in results]))
                        else:
                            st.info("No non-empty views to commit.")
                    except OperationalError as oe:
                        st.error(f"MySQL error: {oe.orig}")
                        log(f"[Commit][ERROR] each: {oe.orig}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        log(f"[Commit][ERROR] each: {e}")
# endregion Connect & Download
# ============================================================


# ============================================================
# region Data Explorer (Generic MySQL viewer)
# ============================================================
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
                log(f"[Explorer] Loaded viewer env profile '{name}'.")
            elif viewer_pick.startswith("[saved] "):
                name = viewer_pick.replace("[saved] ", "", 1)
                ss.viewer_mysql_prof = mysql_saved.get(name, {})
                log(f"[Explorer] Loaded viewer saved profile '{name}'.")
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
            st.success(f"Saved MySQL viewer profile '{v_save_name}'"); log(f"[Explorer] Saved viewer profile '{v_save_name}'")

    viewer_prof_current = {"host": v_host, "port": v_port, "user": v_user, "password": v_password, "db": v_db}

    if st.button("Load tables", key="viewer_load_tables"):
        try:
            ss.viewer_tables = list_mysql_tables(viewer_prof_current)
            st.success(f"Found {len(ss.viewer_tables)} tables.")
            log(f"[Explorer] tables: {len(ss.viewer_tables)}")
        except Exception as e:
            st.error(f"Failed to list tables: {e}")
            log(f"[Explorer][ERROR] list tables: {e}")

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
# ============================================================


# ============================================================
# region AI Demo Data (MySQL → MySQL)
# ============================================================
with tabs[2]:
    st.header("AI Demo Data (MySQL → MySQL)")
    st.caption("Select a **source** DB + tables, choose an **industry**, (optionally) use OpenAI naming, then write transformed tables to a **destination** DB.")

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
                log(f"[DemoData] Loaded MySQL source env profile '{name}'.")
            elif src_pick.startswith("[saved] "):
                name = src_pick.replace("[saved] ", "", 1)
                st.session_state.mysql_source_prof = mysql_saved.get(name, {})
                log(f"[DemoData] Loaded MySQL source saved profile '{name}'.")
            else:
                st.session_state.mysql_source_prof = {}
    with c3:
        if st.button("Test Source", use_container_width=True, key="demo_src_test"):
            ok, msg = test_mysql_connection(st.session_state.mysql_source_prof or mysql_defaults)
            (st.success if ok else st.error)(msg)

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

    s6, s7 = st.columns([3,1])
    with s6:
        src_save_name = st.text_input("Save source profile as", placeholder="e.g., prod-source", key="demo_src_save_name")
    with s7:
        if st.button("Save Source Profile", use_container_width=True, key="demo_src_save_btn") and src_save_name:
            save_json_profile(PROFILE_STORE_PATH_MYSQL, src_save_name, {
                "host": src_host, "port": src_port, "user": src_user, "password": src_password, "db": src_db
            })
            st.success(f"Saved MySQL source profile '{src_save_name}'"); log(f"[DemoData] Saved src profile '{src_save_name}'")

    src_prof_current = {"host": src_host, "port": src_port, "user": src_user, "password": src_password, "db": src_db}
    if st.button("Load tables from source DB", key="demo_src_load_tables"):
        try:
            st.session_state.source_tables = list_mysql_tables(src_prof_current)
            st.success(f"Found {len(st.session_state.source_tables)} tables.")
            log(f"[DemoData] source tables: {len(st.session_state.source_tables)}")
        except Exception as e:
            st.error(f"Failed to list source tables: {e}")
            log(f"[DemoData][ERROR] list tables: {e}")

    if st.session_state.source_tables:
        st.caption("Pick one or more source tables to transform")
        st.session_state.selected_source_tables = st.multiselect("Source tables", options=st.session_state.source_tables, default=st.session_state.source_tables[:3], key="demo_src_table_multiselect")

    st.divider()
    st.subheader("Transformation settings")

    colx1, colx2, colx3 = st.columns([2,1,1])
    with colx1:
        industry = st.text_input("Target industry", value="Water & Wastewater",
                                 help="Used to generate consistent Regions/Segments/Plant names.", key="demo_industry")
    with colx2:
        use_openai = st.checkbox("Use OpenAI for naming", value=True,
                                 help="Reads API key from OPENAI_API or OPENAI_API_KEY in .env.", key="demo_use_openai")
    with colx3:
        add_server_col = st.checkbox("Add ServerName column (if missing)", value=False, key="demo_add_server")

    if use_openai and not (os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")):
        st.warning("OPENAI_API (or OPENAI_API_KEY) not set. Will fall back to heuristic naming.")

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
            st.success(f"Saved MySQL destination profile '{dest_save_name}'"); log(f"[DemoData] Saved dest profile '{dest_save_name}'")

    # ---- Preview ----
    if st.session_state.selected_source_tables and st.button("Preview first table transform", key="demo_preview_btn"):
        try:
            engine_src = _mysql_engine_from_profile(src_prof_current)
            names_map = build_name_maps_from_mysql_tables(engine_src, src_db, st.session_state.selected_source_tables, industry, use_ai=use_openai)

            # Show whether AI was used for naming
            ai_flag = bool(names_map.get("ai_used"))
            st.info(f"Naming source: {'OpenAI (AI)' if ai_flag else 'Fallback heuristics'} · Company: {names_map.get('company','')}")
            log(f"[DemoData] naming source: {'AI' if ai_flag else 'fallback'}; company={names_map.get('company','')}")

            first = st.session_state.selected_source_tables[0]
            df_src = read_mysql_table(src_prof_current, first).head(50)
            df_tx = apply_common_changes(df_src, industry, names_map, server_override=None, add_server_col=add_server_col)
            # Preserve external_id exactly
            if "external_id" in df_src.columns:
                df_tx["external_id"] = df_src["external_id"]
            st.success(f"Preview: {first}")
            st.dataframe(df_tx, use_container_width=True, height=420)
            st.session_state.preview_table = first
        except Exception as e:
            st.error(f"Preview failed: {e}")
            log(f"[DemoData][ERROR] preview: {e}")

    # ---- Transform & write all ----
    if st.session_state.selected_source_tables and st.button("Transform & write ALL selected to destination", type="primary", key="demo_write_all"):
        try:
            engine_src = _mysql_engine_from_profile(src_prof_current)
            names_map = build_name_maps_from_mysql_tables(engine_src, src_db, st.session_state.selected_source_tables, industry, use_ai=use_openai)

            # Announce naming source for this run
            ai_flag = bool(names_map.get("ai_used"))
            st.toast(f"Naming source: {'OpenAI (AI)' if ai_flag else 'Fallback heuristics'}", icon="🤖" if ai_flag else "🧩")
            log(f"[DemoData] naming source: {'AI' if ai_flag else 'fallback'}; company={names_map.get('company','')}")

            dest_prof_current2 = {"host": dest_host, "port": dest_port, "user": dest_user, "password": dest_password, "db": dest_db}
            results = []
            with st.spinner("Transforming and writing tables…"):
                for t in st.session_state.selected_source_tables:
                    out_name = _sanitize_table_name(f"{t}_demo")
                    df_src_full = read_mysql_table(src_prof_current, t)
                    df_tx_full = apply_common_changes(df_src_full, industry, names_map, server_override=None, add_server_col=add_server_col)
                    if "external_id" in df_src_full.columns:
                        df_tx_full["external_id"] = df_src_full["external_id"]
                    rows = commit_dataframe_to_mysql(df_tx_full, out_name, dest_prof_current2)
                    results.append((t, out_name, rows))
            lines = [f"- `{t}` → `{dest_db}.{o}`: {r:,} rows" for (t, o, r) in results]
            st.success("Wrote:\n" + "\n".join(lines))
            log("[DemoData] write: " + "; ".join([f"{t}->{o}:{r}" for (t,o,r) in results]))
        except OperationalError as oe:
            st.error(f"MySQL error: {oe.orig}")
            log(f"[DemoData][ERROR] write: {oe.orig}")
        except Exception as e:
            st.error(f"Failed: {e}")
            log(f"[DemoData][ERROR] write: {e}")
# endregion AI Demo Data
# ============================================================


# ============================================================
# region Helpers: CDF Upload (shared by Upload & Quick Change)
# ============================================================
def upload_df_to_view(client: CogniteClient, space: str, view_eid: str, view_ver: str,
                      df: pd.DataFrame, max_rows: int, label: str) -> List[str]:
    """
    Upload df rows (up to max_rows) into the given view.
    - external_id is used as-is (required).
    - Only columns that exist as view properties are applied.
    Returns list of error strings.
    """
    errs: List[str] = []
    if df is None or df.empty:
        return errs
    if "external_id" not in df.columns:
        errs.append(f"[{label}] skipped: no 'external_id' column.")
        return errs

    view_id = dm.ViewId(space=space, external_id=view_eid, version=str(view_ver))
    vdef_list = client.data_modeling.views.retrieve(view_id, include_inherited_properties=True)
    if not vdef_list:
        errs.append(f"[{label}] view not found.")
        return errs
    view_props = set(vdef_list[0].properties.keys())

    def row_to_apply(row: pd.Series) -> dm.NodeApply:
        props: Dict[str, Any] = {}
        for col in df.columns:
            if col in ("space", "external_id"):
                continue
            if col in view_props:
                val = row[col]
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                props[col] = val
        return dm.NodeApply(
            space=space,
            external_id=str(row["external_id"]),
            sources=[dm.NodeOrEdgeData(source=view_id, properties=props)],
        )

    out = df.head(int(max_rows)).copy()
    total = len(out)
    if total == 0:
        return errs

    bs = UPLOAD_CHUNK_SIZE
    prog = st.progress(0.0, text=f"{label}: uploading…")
    for i in range(0, total, bs):
        chunk = out.iloc[i:i+bs]
        try:
            nodes = [row_to_apply(chunk.iloc[j]) for j in range(len(chunk))]
            client.data_modeling.instances.apply(nodes=nodes)
        except Exception as e:
            errs.append(f"[{label}] rows {i+1}-{min(i+bs,total)}: {e}")
        prog.progress(min(1.0, (i+bs)/total), text=f"{label}: {min(i+bs,total)}/{total}")
    return errs
# endregion Helpers: CDF Upload
# ============================================================


# ============================================================
# region DataMosaix Upload
# ============================================================
with tabs[3]:
    st.header("DataMosaix Upload")
    st.caption("Upload **CSV** or **MySQL tables** into selected Data Model Views in CDF. external_id is required and used as-is.")

    # Build CDF client from a chosen/saved profile or env (reuse Connect tab env by default)
    st.subheader("1) Connect to CDF (target)")
    env_defaults = {
        "host": _env("CARGILL_FTDM_HOST"),
        "project": _env("CARGILL_FTDM_PROJECT"),
        "token_url": _env("CARGILL_FTDM_TOKENURL"),
        "client_id": _env("CARGILL_FTDM_CLIENTID"),
        "client_secret": _env("CARGILL_FTDM_CLIENTSECRET"),
        "scopes": _env("CARGILL_FTDM_SCOPES", "user_impersonation"),
    }
    with st.form("upload_cdf_conn"):
        c1, c2 = st.columns(2)
        with c1:
            u_host = st.text_input("Host", value=env_defaults["host"], key="upload_host")
            u_project = st.text_input("Project", value=env_defaults["project"], key="upload_project")
            u_token = st.text_input("Token URL", value=env_defaults["token_url"], key="upload_token")
            u_scopes = st.text_input("Scopes", value=env_defaults["scopes"], key="upload_scopes")
        with c2:
            u_client = st.text_input("Client ID", value=env_defaults["client_id"], key="upload_client")
            u_show = st.toggle("Show client secret", value=False, key="upload_show_secret")
            u_secret = st.text_input("Client Secret", value=env_defaults["client_secret"], type=("default" if u_show else "password"), key="upload_secret")
            u_max_rows = st.number_input("Default Max Rows per upload", min_value=1, max_value=1_000_000, value=30000, step=1000, key="upload_max_rows")

        connect_upload = st.form_submit_button("Connect to CDF")
    if connect_upload:
        try:
            ss.upload_client = build_client(u_host, u_project, u_token, u_client, u_secret, u_scopes)
            ss.upload_conn_key = f"{u_host}|{u_project}|{u_token}|{u_client}|{u_scopes}"
            st.success("Connected.")
            log("[Upload] Connected.")
        except Exception as e:
            st.error(f"Failed: {e}")
            log(f"[Upload][ERROR] connect: {e}")

    if ss.get("upload_client"):
        st.subheader("2) Select Space / Model / Views (target)")
        s1, s2 = st.columns([1,3])
        with s1:
            if st.button("Load Spaces", key="upload_load_spaces"):
                try:
                    ss.upload_spaces = list_spaces(ss.upload_client, ss.upload_conn_key)
                    st.success(f"{len(ss.upload_spaces)} space(s)."); log(f"[Upload] spaces: {len(ss.upload_spaces)}")
                except Exception as e:
                    st.error(str(e)); log(f"[Upload][ERROR] load spaces: {e}")
        with s2:
            up_space = st.selectbox("Space", options=ss.get("upload_spaces", []) or ["(none)"], key="upload_space")

        m1, m2 = st.columns([1,3])
        with m1:
            if st.button("Load Models", key="upload_load_models", disabled=not up_space or up_space=="(none)"):
                try:
                    ss.upload_models = list_models_latest(ss.upload_client, ss.upload_conn_key, up_space)
                    st.success(f"{len(ss.upload_models)} model(s)."); log(f"[Upload] models: {len(ss.upload_models)} for {up_space}")
                except Exception as e:
                    st.error(str(e)); log(f"[Upload][ERROR] load models: {e}")
        with m2:
            up_models = ss.get("upload_models", [])
            up_dm_choice = st.selectbox("Model", options=[f"{eid} v{ver}" for (eid, ver) in up_models] or ["(none)"], key="upload_model_pick")
            up_dm_eid, up_dm_ver = ("","")
            if " v" in up_dm_choice:
                up_dm_eid, up_dm_ver = up_dm_choice.rsplit(" v", 1)

        v1, v2 = st.columns([1,3])
        with v1:
            if st.button("Load Views", key="upload_load_views", disabled=not(up_dm_eid and up_dm_ver and up_dm_choice!="(none)")):
                try:
                    ss.upload_views = list_views_for_model(ss.upload_client, ss.upload_conn_key, up_space, up_dm_eid, up_dm_ver)
                    st.success(f"{len(ss.upload_views)} view(s)."); log(f"[Upload] views: {len(ss.upload_views)} for {up_dm_eid} v{up_dm_ver}")
                except Exception as e:
                    st.error(str(e)); log(f"[Upload][ERROR] load views: {e}")
        with v2:
            up_views = ss.get("upload_views", [])
            up_selected_views = st.multiselect("Target Views", options=[f"{eid} v{ver}" for (eid, ver) in up_views], key="upload_selected_views")

        st.subheader("3) Choose source: CSV or MySQL")
        src_tab = st.tabs(["CSV → View", "MySQL → View"])[0:2]

        # ---- CSV → View ----
        with src_tab[0]:
            st.caption("Upload one or more CSV files and map each to a target View.")
            csv_rows = []
            for i in range(len(up_selected_views)):
                colA, colB = st.columns([2,2])
                with colA:
                    csv_file = st.file_uploader(f"CSV for {up_selected_views[i]}", type=["csv"], key=f"upload_csv_{i}")
                with colB:
                    max_rows_csv = st.number_input(f"Max rows ({up_selected_views[i]})", min_value=1, max_value=1_000_000, value=int(u_max_rows or 30000), step=1000, key=f"upload_csv_rows_{i}")
                if csv_file:
                    csv_rows.append((up_selected_views[i], csv_file, int(max_rows_csv)))

            if st.button("Upload CSV(s) to selected view(s)", key="upload_csv_go") and csv_rows:
                errs_all = []
                for vtxt, fobj, lim in csv_rows:
                    v_eid, v_ver = vtxt.rsplit(" v", 1)
                    try:
                        df = pd.read_csv(fobj)
                        if "external_id" not in df.columns:
                            st.error(f"[{fobj.name} → {v_eid}] missing 'external_id' column")
                            continue
                        errs = upload_df_to_view(ss.upload_client, up_space, v_eid, v_ver, df, lim, f"{fobj.name} → {v_eid}")
                        errs_all.extend(errs)
                        if errs:
                            st.warning(f"[{fobj.name} → {v_eid}] finished with {len(errs)} error(s).")
                        else:
                            st.success(f"[{fobj.name} → {v_eid}] uploaded OK.")
                    except Exception as e:
                        st.error(f"[{fobj.name} → {v_eid}] failed: {e}")
                        errs_all.append(str(e))
                if errs_all:
                    log("[Upload] CSV errors: " + " | ".join(errs_all))
                else:
                    log("[Upload] CSV upload completed.")

        # ---- MySQL → View ----
        with src_tab[1]:
            st.caption("Map MySQL tables to one or more Views. Default table name = `{viewname.lower()}_demo`.")
            mysql_defaults = _mysql_env_default()
            g1, g2, g3, g4, g5 = st.columns([2,1,1,1,1])
            with g1:
                v_host = st.text_input("MySQL Host", value=mysql_defaults.get("host",""), key="upload_mysql_host")
            with g2:
                v_port = st.text_input("Port", value=str(mysql_defaults.get("port","3306")), key="upload_mysql_port")
            with g3:
                v_user = st.text_input("User", value=mysql_defaults.get("user",""), key="upload_mysql_user")
            with g4:
                v_show = st.toggle("Show password", value=False, key="upload_mysql_show_pwd")
                v_password = st.text_input("Password", value=mysql_defaults.get("password",""),
                                           type="default" if v_show else "password", key="upload_mysql_password")
            with g5:
                v_db = st.text_input("Database", value=mysql_defaults.get("db",""), key="upload_mysql_db")
            prof = {"host": v_host, "port": v_port, "user": v_user, "password": v_password, "db": v_db}

            if st.button("List tables", key="upload_mysql_list"):
                try:
                    ss.upload_mysql_tables = list_mysql_tables(prof)
                    st.success(f"{len(ss.upload_mysql_tables)} table(s)."); log(f"[Upload] mysql tables: {len(ss.upload_mysql_tables)}")
                except Exception as e:
                    st.error(f"List failed: {e}"); log(f"[Upload][ERROR] list mysql tables: {e}")

            tbls = ss.get("upload_mysql_tables", [])
            map_rows = []
            for vtxt in up_selected_views:
                v_eid, v_ver = vtxt.rsplit(" v", 1)
                default_tbl = _sanitize_table_name(v_eid.lower() + "_demo")
                colA, colB, colC = st.columns([2, 2, 1])
                with colA:
                    table_pick = st.selectbox(f"Table for view {v_eid}", options=(tbls or []) + [default_tbl],
                                              index=((tbls.index(default_tbl) if default_tbl in tbls else (len(tbls) if tbls else 0))),
                                              key=f"upload_map_{v_eid}")
                with colB:
                    max_rows_tbl = st.number_input(f"Max rows ({v_eid})", min_value=1, max_value=1_000_000, value=int(u_max_rows or 30000), step=1000, key=f"upload_tbl_rows_{v_eid}")
                with colC:
                    sync = st.checkbox("Sync", value=False, help="Delete target rows (by external_id) not present in source.", key=f"upload_sync_{v_eid}")
                map_rows.append((table_pick, v_eid, v_ver, int(max_rows_tbl), sync))

            def _delete_missing_by_external_id(client: CogniteClient, space: str, v_eid: str, v_ver: str, keep_external_ids: List[str]) -> Optional[str]:
                try:
                    view_id = dm.ViewId(space=space, external_id=v_eid, version=str(v_ver))
                    existing = list(client.data_modeling.instances.list(instance_type="node", sources=[view_id], limit=None))
                    existing_ids = {getattr(n, "external_id", "") for n in existing}
                    to_delete = [dm.NodeId(space=space, external_id=eid) for eid in existing_ids if eid and eid not in keep_external_ids]
                    if to_delete:
                        client.data_modeling.instances.delete(nodes=to_delete, fail_if_missing=False)
                    return None
                except Exception as e:
                    return str(e)

            if st.button("Upload table mappings", type="primary", key="upload_mysql_go") and map_rows:
                errs_all = []
                for (tbl, v_eid, v_ver, lim, sync) in map_rows:
                    try:
                        df = read_mysql_table(prof, tbl, limit=int(lim))
                        if "external_id" not in df.columns:
                            st.error(f"[{tbl} → {v_eid}] missing 'external_id' column")
                            continue
                        if sync:
                            msg = _delete_missing_by_external_id(ss.upload_client, up_space, v_eid, v_ver, keep_external_ids=df["external_id"].astype(str).tolist())
                            if msg:
                                st.warning(f"[{tbl} → {v_eid}] sync warning: {msg}")
                                log(f"[Upload][SYNC] warn {tbl}->{v_eid}: {msg}")
                        errs = upload_df_to_view(ss.upload_client, up_space, v_eid, v_ver, df, int(lim), f"{tbl} → {v_eid}")
                        errs_all.extend(errs)
                        if errs:
                            st.warning(f"[{tbl} → {v_eid}] finished with {len(errs)} error(s).")
                        else:
                            st.success(f"[{tbl} → {v_eid}] uploaded OK.")
                    except Exception as e:
                        st.error(f"[{tbl} → {v_eid}] failed: {e}")
                        errs_all.append(str(e))
                if errs_all:
                    log("[Upload] MySQL errors: " + " | ".join(errs_all))
                else:
                    log("[Upload] MySQL upload completed.")
# endregion DataMosaix Upload
# ============================================================


# ============================================================
# region Logs
# ============================================================
with tabs[4]:
    st.header("Logs")
    log_text = "\n".join(st.session_state.logs[-500:]) if st.session_state.logs else "No logs yet."
    st.text_area("Activity", value=log_text, height=240, label_visibility="collapsed")
# endregion Logs
# ============================================================


# ============================================================
# region Quick Change
# ============================================================
with tabs[5]:
    st.header("⚡ Quick Change")
    st.caption(
        "One-prompt pipeline: transform source MySQL tables from **Cargill_EAMS** → write `*_demo` tables into "
        "**EAMS_Demo** (preserving `external_id`), then upload those tables into **selected** views from your EAMS Demo CDF."
    )

    candidate_views = [
        "EventLog",
        "RAAIChangeswRegionSegment",
        "RAAIwRegionSegment",
        "AssetHierarchy",
        "ScheduleTasks",
        "AuditEventLog",
        "RAAIwLifecycle",
    ]
    default_max_rows = 30000

    _mysql_default = _mysql_env_default()
    src_prof_qc = {
        "host": _mysql_default["host"],
        "port": _mysql_default["port"],
        "user": _mysql_default["user"],
        "password": _mysql_default["password"],
        "db": "Cargill_EAMS",
    }
    dest_prof_qc = {
        "host": _mysql_default["host"],
        "port": _mysql_default["port"],
        "user": _mysql_default["user"],
        "password": _mysql_default["password"],
        "db": "EAMS_Demo",
    }

    def _best_match_table(src_tables: List[str], want: str) -> Optional[str]:
        lowmap = {t.lower(): t for t in src_tables}
        cands = [want.lower(), want.lower().replace("_", ""), want.lower().replace("-", "")]
        for c in cands:
            if c in lowmap:
                return lowmap[c]
        for t in src_tables:
            if want.lower() in t.lower():
                return t
        return None

    def _build_app_client_from_env() -> Tuple[CogniteClient, str]:
        host = _env("CGX_FTDM_APP_HOST")
        project = _env("CGX_FTDM_APP_PROJECT")
        token_url = _env("CGX_FTDM_APP_TOKENURL")
        client_id = _env("CGX_FTDM_APP_CLIENTID")
        client_secret = _env("CGX_FTDM_APP_SECRET")
        audience = _env("CGX_FTDM_APP_AUDIENCE")  # optional (Auth0)
        if not all([host, project, token_url, client_id, client_secret]):
            raise RuntimeError("CGX_FTDM_APP_* environment variables are missing/incomplete.")
        try:
            creds = OAuthClientCredentials(
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                audience=audience if audience else None,
                scopes=None if audience else ["user_impersonation"],
            )
        except TypeError:
            creds = OAuthClientCredentials(
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scopes=["user_impersonation"],
            )
        cfg = ClientConfig(client_name=project, project=project, credentials=creds, base_url=host)
        key = f"QCAPP|{host}|{project}|{token_url}|{client_id}|{bool(audience)}"
        return CogniteClient(cfg), key

    st.subheader("Step A — Transform source → write `*_demo` tables (MySQL)")
    colA, colB = st.columns([2, 1])
    with colA:
        industry_qc = st.text_input(
            "Target industry",
            value="Water & Wastewater",
            help="Used by the transformer for realistic Regions/Segments/Plants naming."
        )
    with colB:
        max_rows_qc = st.number_input("Max rows per table", min_value=1, max_value=1_000_000,
                                      value=default_max_rows, step=1000)

    if st.button("Run Transform to EAMS_Demo", type="primary", key="qc_transform"):
        try:
            st.info("Listing source tables from **Cargill_EAMS**…")
            src_tables = list_mysql_tables(src_prof_qc)
            st.write(f"• Found {len(src_tables)} tables")

            table_map: Dict[str, str] = {}
            for v in candidate_views:
                t = _best_match_table(src_tables, v)
                if t:
                    table_map[v] = t
                else:
                    st.warning(f"Skipping: no source table matching '{v}'")

            if not table_map:
                raise RuntimeError("No matching source tables found in Cargill_EAMS.")

            st.write("Building naming maps & transforming (external_id preserved)…")
            engine_src = _mysql_engine_from_profile(src_prof_qc)
            names_map = build_name_maps_from_mysql_tables(
                engine_src, src_prof_qc["db"], list(table_map.values()), industry_qc,
                use_ai=bool(_env("OPENAI_API") or _env("OPENAI_API_KEY"))
            )

            # Display naming source to the user
            ai_flag = bool(names_map.get("ai_used"))
            st.write(f"**Naming source:** {'OpenAI (AI)' if ai_flag else 'Fallback heuristics'} — Company: *{names_map.get('company','')}*")
            log(f"[QuickChange] naming source: {'AI' if ai_flag else 'fallback'}; company={names_map.get('company','')}")

            results_sql = []
            for view_eid, src_tbl in table_map.items():
                df_src = read_mysql_table(src_prof_qc, src_tbl)
                orig_ext = df_src.get("external_id").copy() if "external_id" in df_src.columns else None
                df_tx = apply_common_changes(df_src, industry_qc, names_map, server_override=None, add_server_col=False)
                if orig_ext is not None:
                    df_tx["external_id"] = orig_ext  # Preserve external_id exactly
                out_tbl = _sanitize_table_name(view_eid.lower() + "_demo")
                rows = commit_dataframe_to_mysql(df_tx, out_tbl, dest_prof_qc)
                results_sql.append((src_tbl, out_tbl, rows))
                st.write(f"• `{src_tbl}` → `{dest_prof_qc['db']}.{out_tbl}` : {rows:,} rows")
                log(f"[QuickChange] SQL {src_tbl} -> {out_tbl}: {rows}")
            st.success("Transform complete.")
        except Exception as e:
            st.error(f"Transform failed: {e}")
            log(f"[QuickChange][ERROR] transform: {e}")

    st.markdown("---")

    st.subheader("Step B — Select target views from EAMS Demo CDF & upload")
    try:
        app_client, conn_key_qc = _build_app_client_from_env()
    except Exception as e:
        st.error(f"Cannot build CDF app client from environment: {e}")
        st.stop()

    s1, s2 = st.columns([1, 3])
    with s1:
        if st.button("Load Spaces (EAMS Demo)", key="qc_load_spaces"):
            try:
                ss.qc_spaces = list_spaces(app_client, conn_key_qc)
                st.success(f"Loaded {len(ss.qc_spaces)} space(s).")
                log(f"[QuickChange] loaded spaces: {len(ss.qc_spaces)}")
            except Exception as e:
                st.error(str(e)); log(f"[QuickChange][ERROR] load spaces: {e}")
    with s2:
        qc_space = st.selectbox("Space (target)", options=ss.get("qc_spaces", []) or ["(none)"], key="qc_space_pick")

    m1, m2 = st.columns([1, 3])
    with m1:
        disabled = not qc_space or qc_space == "(none)"
        if st.button("Load Models (EAMS Demo)", disabled=disabled, key="qc_load_models"):
            try:
                ss.qc_models = list_models_latest(app_client, conn_key_qc, qc_space)
                st.success(f"Loaded {len(ss.qc_models)} model(s).")
                log(f"[QuickChange] loaded models: {len(ss.qc_models)} for {qc_space}")
            except Exception as e:
                st.error(str(e)); log(f"[QuickChange][ERROR] load models: {e}")
    with m2:
        dm_display_qc = [f"{eid} v{ver}" for (eid, ver) in ss.get("qc_models", [])] or ["(none)"]
        qc_dm_choice = st.selectbox("Model (target)", options=dm_display_qc, key="qc_model_pick")
        qc_dm_eid, qc_dm_ver = ("", "")
        if " v" in qc_dm_choice:
            qc_dm_eid, qc_dm_ver = qc_dm_choice.rsplit(" v", 1)

    v1, v2 = st.columns([1, 3])
    with v1:
        disabled = not (qc_dm_eid and qc_dm_ver and qc_dm_choice != "(none)")
        if st.button("Load Views (EAMS Demo)", disabled=disabled, key="qc_load_views"):
            try:
                ss.qc_views = list_views_for_model(app_client, conn_key_qc, qc_space, qc_dm_eid, qc_dm_ver)
                st.success(f"Loaded {len(ss.qc_views)} view(s).")
                log(f"[QuickChange] loaded views: {len(ss.qc_views)} for {qc_dm_eid} v{qc_dm_ver}")
            except Exception as e:
                st.error(str(e)); log(f"[QuickChange][ERROR] load views: {e}")
    with v2:
        qc_view_display = [f"{eid} v{ver}" for (eid, ver) in ss.get("qc_views", [])] or []
        qc_selected_views = st.multiselect(
            "Select target views to upload to",
            options=qc_view_display,
            default=[x for x in qc_view_display if any(x.lower().startswith(v.lower()) for v in candidate_views)][:3],
            key="qc_selected_views"
        )

    demo_tables = []
    try:
        demo_tables = list_mysql_tables(dest_prof_qc)
    except Exception as e:
        st.warning(f"Could not list tables from `{dest_prof_qc['db']}`: {e}")

    st.markdown("**Table → View mappings** (defaults to `{viewname.lower()}_demo`)")

    qc_pairs = []
    for vtxt in qc_selected_views:
        v_eid, v_ver = vtxt.rsplit(" v", 1)
        default_tbl = _sanitize_table_name(v_eid.lower() + "_demo")
        r1, r2 = st.columns([2, 2])
        with r1:
            tbl_pick = st.selectbox(
                f"Table for view {v_eid}",
                options=(demo_tables or []) + [default_tbl],
                index=(demo_tables.index(default_tbl) if default_tbl in demo_tables else len(demo_tables or [])),
                key=f"qc_tbl_for_{v_eid}"
            )
        with r2:
            lim = st.number_input(
                f"Max rows to upload ({v_eid})",
                min_value=1, max_value=1_000_000,
                value=default_max_rows, step=1000,
                key=f"qc_max_rows_{v_eid}"
            )
        qc_pairs.append((tbl_pick, v_eid, v_ver, lim))

    if st.button("Upload selected mappings", type="primary", key="qc_upload_selected") and qc_pairs:
        try:
            errs_global: List[str] = []
            for (tbl, v_eid, v_ver, lim) in qc_pairs:
                try:
                    df_demo = read_mysql_table(dest_prof_qc, tbl, limit=int(lim))
                except Exception as e:
                    st.error(f"[{tbl} → {v_eid}] read failed: {e}")
                    log(f"[QuickChange][ERROR] read table {tbl}: {e}")
                    continue

                label = f"{tbl} → {v_eid}"
                errs = upload_df_to_view(app_client, qc_space, v_eid, v_ver, df_demo, int(lim), label)
                errs_global.extend(errs)

                if errs:
                    st.warning(f"[{label}] finished with {len(errs)} error(s).")
                else:
                    st.success(f"[{label}] uploaded successfully.")

            if errs_global:
                st.warning("Finished with some errors. See Logs tab.")
                for e in errs_global[:6]:
                    st.write("• " + e)
                log("[QuickChange] upload errors: " + " | ".join(errs_global))
            else:
                st.success("Upload completed successfully 🎉")
                log("[QuickChange] upload completed OK")
        except Exception as e:
            st.error(f"Upload failed: {e}")
            log(f"[QuickChange][ERROR] upload: {e}")
# endregion Quick Change
# ============================================================


# ============================================================
# region About
# ============================================================
with tabs[6]:
    st.header("About")
    st.caption(f"Version {APP_VERSION}")
    st.subheader("Changelog")
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("Changelog not bundled in this image. Add `COPY CHANGELOG.md ./` to your Dockerfile or bind-mount the project.")
# endregion About
# ============================================================
