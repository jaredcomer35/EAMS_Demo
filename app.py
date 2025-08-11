import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
from PIL import Image

from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthClientCredentials
from cognite.client.data_classes import data_modeling as dm

# SQLAlchemy for MySQL commits
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

# ---------- Versioning ----------
APP_VERSION = "1.1.0-dev"  # feature/multiselect-views
__version__ = APP_VERSION

# ---------- Branding ----------
APP_TITLE = "Convergix DataMosaix View Explorer"
APP_ICON_PATH = "assets/convergix_logo.png"
_icon = None
try:
    _icon = Image.open(APP_ICON_PATH)
except Exception:
    pass

st.set_page_config(page_title=f"{APP_TITLE} · v{APP_VERSION}", page_icon=_icon if _icon else "🧭", layout="wide")

# Header
left, right = st.columns([1, 8])
with left:
    if _icon:
        st.image(_icon, width=60)
with right:
    st.title(APP_TITLE)
    st.caption(f"Version {APP_VERSION} (multiselect views)")


# ---------- Utilities ----------
def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def log(msg: str):
    st.session_state.setdefault("logs", [])
    st.session_state.logs.append(msg)


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


# ---------- Profile storage (persisted on disk) ----------
PROFILE_STORE_PATH_CDF = _env("PROFILE_STORE_PATH", "/data/profiles.json")
PROFILE_STORE_PATH_MYSQL = _env("MYSQL_PROFILE_STORE_PATH", "/data/mysql_profiles.json")


def _path(p: str) -> Path:
    pp = Path(p)
    pp.parent.mkdir(parents=True, exist_ok=True)
    return pp


def load_json_profiles(path: str) -> Dict[str, Dict[str, str]]:
    file = _path(path)
    if file.exists():
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass
    return {}


def save_json_profile(path: str, name: str, profile: Dict[str, str]):
    data = load_json_profiles(path)
    data[name] = profile
    with _path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def delete_json_profile(path: str, name: str):
    data = load_json_profiles(path)
    if name in data:
        del data[name]
        with _path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ---------- Env-group profiles ----------
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


# ---------- Cognite client ----------
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


# ---------- Cached calls (ignore client in hash via leading underscore) ----------
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


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=False):
        q = st.text_input("Search across all columns", placeholder="Type to filter…")
        filtered = df
        if q:
            mask = pd.Series(False, index=filtered.index)
            for col in filtered.columns:
                try:
                    mask |= filtered[col].astype(str).str.contains(q, case=False, na=False)
                except Exception:
                    pass
            filtered = filtered[mask]
        cols = st.multiselect("Add per-column filters", sorted(list(df.columns)))
        for c in cols:
            s = filtered[c]
            if pd.api.types.is_numeric_dtype(s):
                min_v, max_v = float(s.min()), float(s.max())
                a, b = st.slider(f"{c} range", min_v, max_v, (min_v, max_v))
                filtered = filtered[(s.astype(float) >= a) & (s.astype(float) <= b)]
            elif pd.api.types.is_datetime64_any_dtype(s):
                s2 = pd.to_datetime(s, errors="coerce")
                filtered = filtered.assign(**{c: s2})
                min_d, max_d = s2.min(), s2.max()
                a, b = st.date_input(f"{c} between", value=(min_d, max_d))
                if a:
                    filtered = filtered[s2 >= pd.to_datetime(a)]
                if b:
                    filtered = filtered[s2 <= pd.to_datetime(b)]
            else:
                uniques = s.dropna().astype(str).unique()
                if len(uniques) <= 50:
                    choices = st.multiselect(f"{c} is one of…", sorted(map(str, uniques)))
                    if choices:
                        filtered = filtered[s.astype(str).isin(choices)]
                else:
                    txt = st.text_input(f"{c} contains…", key=f"txt_{c}")
                    if txt:
                        filtered = filtered[s.astype(str).str.contains(txt, case=False, na=False)]
    return filtered


# ---------- MySQL helpers ----------
def _mysql_env_default() -> Dict[str, str]:
    return {
        "host": _env("MYSQL_HOST", "host.docker.internal"),
        "port": _env("MYSQL_PORT", "3306"),
        "user": _env("MYSQL_USER", ""),
        "password": _env("MYSQL_PASSWORD", ""),
        "db": _env("MYSQL_DB", ""),
    }


def _sanitize_table_name(name: str) -> str:
    import re
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


# ---------- Session state ----------
ss = st.session_state
ss.setdefault("client", None)
ss.setdefault("conn_key", "")
ss.setdefault("spaces", [])
ss.setdefault("models", [])
ss.setdefault("views", [])
ss.setdefault("dfs_by_viewkey", {})       # key: "eid@ver" -> DataFrame
ss.setdefault("combined_df", None)         # union of selected views
ss.setdefault("current_viewkey", "__ALL__") # currently selected in explorer
ss.setdefault("logs", [])
ss.setdefault("prefill", {})
ss.setdefault("table_names_by_viewkey", {})


# ---------- Tabs ----------
tabs = st.tabs(["🔌 Connect & Download", "📊 Data Explorer", "🪵 Logs", "ℹ️ About"])

# ===== Connect & Download =====
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
        picked = st.selectbox("Load CDF profile", options)
    with colp2:
        if st.button("Load", use_container_width=True):
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
        if st.button("Delete", use_container_width=True, disabled=not can_delete):
            name = picked.replace("[saved] ", "", 1)
            delete_json_profile(PROFILE_STORE_PATH_CDF, name)
            st.success(f"Deleted profile '{name}'.")
            log(f"Deleted profile '{name}'.")

    def get_val(name, fallback):
        return ss.prefill.get(name) or env_defaults.get(name) or fallback

    with st.form("conn"):
        c1, c2 = st.columns(2)
        with c1:
            host = st.text_input("Host", value=get_val("host", ""), placeholder="https://api.cognitedata.com").strip()
            project = st.text_input("Project", value=get_val("project", ""), placeholder="your-cdf-project").strip()
            token_url = st.text_input("Token URL", value=get_val("token_url", ""),
                                      placeholder="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token").strip()
            scopes = st.text_input("Scopes (comma-separated)", value=get_val("scopes", "user_impersonation")).strip()
        with c2:
            client_id = st.text_input("Client ID", value=get_val("client_id", "")).strip()
            show = st.toggle("Show client secret", value=False)
            client_secret = st.text_input("Client Secret", value=get_val("client_secret", ""),
                                          type="default" if show else "password").strip()
            max_rows = st.number_input("Max rows to fetch (per view)", min_value=1, value=50000, step=1000)

        sp1, sp2 = st.columns([3, 1])
        with sp1:
            prof_name = st.text_input("Save as CDF profile (persists to /data)", placeholder="e.g., prod-cdf")
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

        connect = st.form_submit_button("Connect", type="primary", disabled=bool(errs))
    if errs:
        st.warning(" • " + "\n • ".join(errs))

    if connect and not errs:
        try:
            ss.client = build_client(host, project, token_url, client_id, client_secret, scopes)
            ss.conn_key = f"{host}|{project}|{token_url}|{client_id}|{scopes}"
            _ = list_spaces(ss.client, ss.conn_key)
            st.success(f"Connected to '{project}'.")
            log("Connected.")
            ss.spaces = []
            ss.models = []
            ss.views = []
            ss.dfs_by_viewkey = {}
            ss.combined_df = None
            ss.current_viewkey = "__ALL__"
            ss.table_names_by_viewkey = {}
        except Exception as e:
            st.error(str(e)); log(f"ERROR connect: {e}")

    st.divider()
    st.header("Download")
    st.caption("Flow: **Load Spaces** → choose Space → **Load Models** → choose Model → **Load Views** → choose one or many → **Fetch data**.")

    cols = st.columns([1, 3])
    with cols[0]:
        if st.button("Load Spaces", use_container_width=True, disabled=ss.client is None):
            try:
                ss.spaces = list_spaces(ss.client, ss.conn_key)
                log(f"Loaded {len(ss.spaces)} spaces.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR load spaces: {e}")
    with cols[1]:
        st.caption("Choose your Space here →")
        space = st.selectbox("Space", options=ss.spaces or ["(none)"], label_visibility="collapsed")

    cols = st.columns([1, 3])
    with cols[0]:
        load_models_disabled = (not space) or space == "(none)"
        if st.button("Load Models", use_container_width=True, disabled=load_models_disabled):
            try:
                ss.models = list_models_latest(ss.client, ss.conn_key, space)
                log(f"Loaded {len(ss.models)} models for space '{space}'.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR load models: {e}")
    with cols[1]:
        st.caption("Choose your Model here →")
        dm_display = [f"{eid} v{ver}" for (eid, ver) in ss.models] or ["(none)"]
        dm_choice = st.selectbox("Model", options=dm_display, label_visibility="collapsed")
        dm_eid, dm_ver = ("", "")
        if " v" in dm_choice:
            dm_eid, dm_ver = dm_choice.rsplit(" v", 1)

    cols = st.columns([1, 3])
    with cols[0]:
        load_views_disabled = not (dm_eid and dm_ver and dm_choice != "(none)")
        if st.button("Load Views", use_container_width=True, disabled=load_views_disabled):
            try:
                ss.views = list_views_for_model(ss.client, ss.conn_key, space, dm_eid, dm_ver)
                log(f"Loaded {len(ss.views)} views for model '{dm_eid}' v{dm_ver}.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR load views: {e}")
    with cols[1]:
        st.caption("Choose your Views here → (multi-select)")
        view_display = [f"{eid} v{ver}" for (eid, ver) in ss.views] or []
        selected_views = st.multiselect("Views", options=view_display, default=view_display[:1])

    cols = st.columns(2)
    with cols[0]:
        fetch_enabled = bool(ss.client and space and dm_eid and dm_ver and selected_views)
        if st.button("Fetch data", type="primary", use_container_width=True, disabled=not fetch_enabled):
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
                        # default table names
                        ss.table_names_by_viewkey.setdefault(key, _sanitize_table_name(v_eid))
                if frames:
                    ss.combined_df = pd.concat(frames, ignore_index=True, sort=False)
                    ss.current_viewkey = "__ALL__"
                    st.success(f"Loaded {len(ss.combined_df):,} rows from {len(frames)} view(s).")
                    log(f"Fetched {len(ss.combined_df):,} rows from {len(frames)} views: {', '.join(fetched_keys)}")
                    st.download_button(
                        "Download CSV (all views)",
                        ss.combined_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dm_eid}_views.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No views selected.")
            except Exception as e:
                st.error(str(e)); log(f"ERROR fetch: {e}")
    with cols[1]:
        if st.button("Clear data", use_container_width=True):
            ss.dfs_by_viewkey = {}
            ss.combined_df = None
            ss.current_viewkey = "__ALL__"
            ss.table_names_by_viewkey = {}
            log("Cleared data.")

    # Quick summary of what we have
    if ss.dfs_by_viewkey:
        st.info(f"Cached views: {len(ss.dfs_by_viewkey)} • Rows (combined): {len(ss.combined_df) if isinstance(ss.combined_df, pd.DataFrame) else 0:,}")


# ===== Data Explorer =====
with tabs[1]:
    st.header("Table & Filters")

    if not ss.dfs_by_viewkey:
        st.info("No data yet. Use **Connect & Download** to fetch one or more views.")
    else:
        # Picker for current view
        keys_sorted = ["__ALL__"] + sorted(ss.dfs_by_viewkey.keys())
        labels = {"__ALL__": "All views (combined)"} | {k: f"{k.split('@',1)[0]} (v{k.split('@',1)[1]})" for k in ss.dfs_by_viewkey}
        pick = st.selectbox("Show", options=keys_sorted, format_func=lambda k: labels[k], index=keys_sorted.index(ss.current_viewkey) if ss.current_viewkey in keys_sorted else 0)
        ss.current_viewkey = pick

        # Select dataframe
        if pick == "__ALL__":
            df_show = ss.combined_df if isinstance(ss.combined_df, pd.DataFrame) else pd.concat(list(ss.dfs_by_viewkey.values()), ignore_index=True, sort=False)
        else:
            df_show = ss.dfs_by_viewkey.get(pick)

        if df_show is None or df_show.empty:
            st.warning("No rows to display.")
        else:
            filtered = filter_dataframe(df_show)
            st.caption(f"Showing {len(filtered):,} / {len(df_show):,} rows")
            st.dataframe(filtered, use_container_width=True, height=520)
            st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                               file_name="filtered.csv", mime="text/csv")
            # Keep filtered only for current view
            ss["filtered_df_current"] = filtered


# ===== Logs =====
with tabs[2]:
    st.header("Logs")
    log_text = "\n".join(ss.logs[-500:]) if ss.logs else "No logs yet."
    st.text_area("Activity", value=log_text, height=240, label_visibility="collapsed")


# ===== About =====
with tabs[3]:
    st.header("About")
    st.caption(f"Version {APP_VERSION}")
    st.subheader("Changelog")
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("Changelog not bundled in this image. Add `COPY CHANGELOG.md ./` to your Dockerfile or bind-mount the project.")


# ===== Commit to MySQL (appears below Data Explorer when data is present) =====
if ss.dfs_by_viewkey:
    st.divider()
    st.subheader("Commit to MySQL DB")

    # Profiles (same as before)
    mysql_envs = mysql_env_profiles()
    mysql_saved = load_json_profiles(PROFILE_STORE_PATH_MYSQL)
    mysql_defaults = {
        "host": _env("MYSQL_HOST", "host.docker.internal"),
        "port": _env("MYSQL_PORT", "3306"),
        "user": _env("MYSQL_USER", ""),
        "password": _env("MYSQL_PASSWORD", ""),
        "db": _env("MYSQL_DB", ""),
    }

    colm1, colm2, colm3, colm4 = st.columns([2, 1, 1, 1])
    with colm1:
        opts = ["(none)"] + [f"[env] {n}" for n in sorted(mysql_envs.keys())] + [f"[saved] {n}" for n in sorted(mysql_saved.keys())]
        picked_mysql = st.selectbox("Load MySQL profile", opts)
    with colm2:
        if st.button("Load MySQL", use_container_width=True):
            if picked_mysql.startswith("[env] "):
                name = picked_mysql.replace("[env] ", "", 1)
                ss["mysql_prof"] = mysql_envs.get(name, {})
                log(f"Loaded MySQL env profile '{name}'.")
            elif picked_mysql.startswith("[saved] "):
                name = picked_mysql.replace("[saved] ", "", 1)
                ss["mysql_prof"] = mysql_saved.get(name, {})
                log(f"Loaded MySQL saved profile '{name}'.")
            else:
                ss["mysql_prof"] = {}
    with colm3:
        can_delete_mysql = picked_mysql.startswith("[saved] ")
        if st.button("Delete MySQL", use_container_width=True, disabled=not can_delete_mysql):
            name = picked_mysql.replace("[saved] ", "", 1)
            delete_json_profile(PROFILE_STORE_PATH_MYSQL, name)
            st.success(f"Deleted MySQL profile '{name}'.")
            log(f"Deleted MySQL profile '{name}'.")
    with colm4:
        prof_preview = ss.get("mysql_prof", {}) or mysql_defaults
        if st.button("Test connection", use_container_width=True):
            ok, msg = test_mysql_connection(prof_preview)
            (st.success if ok else st.error)(msg)
            log(f"MySQL test: {msg}")

    prof = ss.get("mysql_prof", {}) or mysql_defaults
    cma, cmb, cmc, cmd, cme = st.columns([2, 1, 1, 1, 1])
    with cma:
        mysql_host = st.text_input("Host", value=prof.get("host", ""))
    with cmb:
        mysql_port = st.text_input("Port", value=str(prof.get("port", "3306")))
    with cmc:
        mysql_user = st.text_input("User", value=prof.get("user", ""))
    with cmd:
        showp = st.toggle("Show password", value=False)
        mysql_password = st.text_input("Password", value=prof.get("password", ""), type="default" if showp else "password")
    with cme:
        mysql_db = st.text_input("Database", value=prof.get("db", ""))

    spm1, spm2 = st.columns([3, 1])
    with spm1:
        mysql_prof_name = st.text_input("Save MySQL profile as (persists to /data)", placeholder="e.g., local-db")
    with spm2:
        if st.button("Save MySQL", use_container_width=True) and mysql_prof_name:
            try:
                save_json_profile(PROFILE_STORE_PATH_MYSQL, mysql_prof_name, {
                    "host": mysql_host, "port": mysql_port, "user": mysql_user,
                    "password": mysql_password, "db": mysql_db
                })
                st.success(f"Saved MySQL profile '{mysql_prof_name}'.")
                log(f"Saved MySQL profile '{mysql_prof_name}'.")
            except Exception as e:
                st.error(f"Save failed: {e}")
                log(f"ERROR save mysql profile: {e}")

    # Per-view table names
    st.markdown("**Table names per view (for 'Commit all views')**")
    table_cols = st.columns(2)
    idx = 0
    for key in sorted(ss.dfs_by_viewkey.keys()):
        eid, ver = key.split("@", 1)
        with table_cols[idx % 2]:
            ss.table_names_by_viewkey[key] = st.text_input(
                f"{eid} (v{ver})",
                value=ss.table_names_by_viewkey.get(key, _sanitize_table_name(eid)),
                key=f"table_{key}",
            )
        idx += 1

    # Commit buttons
    ca, cb = st.columns(2)

    with ca:
        # Commit only current selection (optionally filtered)
        if ss.current_viewkey != "__ALL__":
            use_filtered = st.checkbox("Commit filtered rows (current view only)", value=False)
            if st.button("Commit current view", type="primary", use_container_width=True):
                try:
                    df_src = ss.get("filtered_df_current") if use_filtered else ss.dfs_by_viewkey.get(ss.current_viewkey)
                    if df_src is None or df_src.empty:
                        raise RuntimeError("No data to commit for the current view.")
                    table_name = ss.table_names_by_viewkey.get(ss.current_viewkey, _sanitize_table_name(ss.current_viewkey.split('@',1)[0]))
                    with st.spinner("Writing current view to MySQL…"):
                        rows = commit_dataframe_to_mysql(
                            df_src,
                            table_name,
                            {"host": mysql_host, "port": mysql_port, "user": mysql_user, "password": mysql_password, "db": mysql_db},
                        )
                    st.success(f"Wrote {rows:,} rows to `{mysql_db}.{table_name}`.")
                    log(f"MySQL commit (single): {rows} rows -> {mysql_db}.{table_name}")
                except OperationalError as oe:
                    st.error(f"MySQL commit failed (OperationalError): {oe.orig}")
                    log(f"ERROR mysql commit single: {oe.orig}")
                except Exception as e:
                    st.error(f"MySQL commit failed: {e}")
                    log(f"ERROR mysql commit single: {e}")
        else:
            st.info("Select a specific view above if you want to commit only that view (with optional filters).")

    with cb:
        # Commit all views (unfiltered, each to its own table name)
        if st.button("Commit ALL views", use_container_width=True):
            try:
                if not ss.dfs_by_viewkey:
                    raise RuntimeError("No views to commit.")
                results = []
                with st.spinner("Writing all views to MySQL…"):
                    for key, df_src in ss.dfs_by_viewkey.items():
                        if df_src is None or df_src.empty:
                            continue
                        table_name = ss.table_names_by_viewkey.get(key, _sanitize_table_name(key.split('@',1)[0]))
                        rows = commit_dataframe_to_mysql(
                            df_src,
                            table_name,
                            {"host": mysql_host, "port": mysql_port, "user": mysql_user, "password": mysql_password, "db": mysql_db},
                        )
                        results.append((key, rows, table_name))
                if results:
                    lines = [f"- {k} → `{mysql_db}.{t}`: {r:,} rows" for (k, r, t) in results]
                    st.success("Committed:\n" + "\n".join(lines))
                    log("MySQL commit (all): " + "; ".join([f"{k}->{t}:{r}" for (k, r, t) in results]))
                else:
                    st.warning("Nothing written (no rows).")
            except OperationalError as oe:
                st.error(f"MySQL commit failed (OperationalError): {oe.orig}")
                log(f"ERROR mysql commit all: {oe.orig}")
            except Exception as e:
                st.error(f"MySQL commit failed: {e}")
                log(f"ERROR mysql commit all: {e}")
