# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2025-08-12
### Added
- **Quick Change**: One-prompt pipeline to transform `Cargill_EAMS` → write `*_demo` to `EAMS_Demo` (preserving `external_id`) and upload to chosen CDF views. Clear logs indicate when OpenAI naming was used or when we fell back to heuristics.
- **DataMosaix Upload**: CSV→View and MySQL→View multi-map UIs; per-mapping row limits (default 30k); default table suggestion `{view}_demo`; optional *Sync* (delete target rows not present in source by `external_id`); chunk size via `CDF_UPLOAD_CHUNK_SIZE` (default 1000).
- **Data Explorer**: Per-column filters (Equals / Not Equals / Contains / Not Contains; numeric ranges; date ops), plus quick stats and Plant counts.

### Changed
- **AI Demo Data**: Never mutate `external_id`. Improved industry-aware naming for Plants/Regions/Segments with richer context; path mutation now only replaces the server segment and preserves the IP portion and remainder.
- **Logging**: All major actions log with timestamps; more verbose progress reporting for Upload and Quick Change.

### Fixed
- Streamlit duplicate key issues in inputs.
- Per-column filter state clearing.
- Default “Max Rows” set to **30,000** where applicable.

### Security
- OpenAI API key sourced from `.env` as `OPENAI_API` (or `OPENAI_API_KEY` as fallback).


## [1.2.0] - 2025-08-11

### Added
- **MySQL Table Viewer** (Data Explorer tab): connect via saved/env profiles, list tables, select one or many to explore, and download filtered CSV.
- **Per-column filter operators** in Data Explorer:
  - Text: *Contains*, *Not Contains*, *Equals*, *Not Equals*
  - Numeric: *Equals*, *Not Equals*, *Range*
  - Date/Datetime: *Between*, *On date*, *Not on date*
- **Quick Stats panel** under the table:
  - Unique counts per column, non-null %, sample values
  - Numeric `describe()` summary
  - Datetime min/max coverage
  - Top-20 counts by `Plant` (if present), with chart
- Profile management UX across CDF/MySQL viewers/destinations persists to `/data` volume.

### Changed
- **AI Demo Data path transform** now **only replaces the server name** and preserves the original IP and the rest of the path  
  (e.g., `MSDUNL227MP!...\\10.4.10.207\\Backplane\\4` → `<NewServer>\\10.4.10.207\\Backplane\\4`).
- Branding updates: app title is **Convergix DataMosaix View Explorer**, updated logo.

### Fixed
- Data Explorer filters no longer clear themselves due to duplicate widget IDs; keys now stable per DB/table/column.
- Commit-to-MySQL section retained after fetching views and clearing.

## [1.1.0] - 2025-08-11
### Added
- **Multiselect Views**: choose multiple Views at once and fetch them in a single run.
- **Combined/Per-View browsing**: Data Explorer lets you switch between **All views (combined)** or a specific view from a dropdown.
- **Per-view MySQL mapping**: set a **table name per View**; commit **current view (optionally filtered)** or **ALL views** (each to its own table).
- Session stores a DataFrame **per view** (keyed `eid@version`) plus a **combined** union.
- CSV download for the combined result.

### Changed
- Fetch button now loops over selected views; default table names auto-suggested from view IDs.
- Clear data resets all cached views, combined frame, and table-name mappings.

### Fixed
- Safer handling when no views are selected or a view returns zero rows.

## [1.0.0] - 2025-08-11
### Added
- **Initial stable release (V1)**: version header shows `v1.0.0`; CHANGELOG visible in the app’s **About** tab.
- Documented branching workflow (feature branches, PRs, tags) and release steps.

### Changed
- Set **main** as the default branch and tagged `v1.0.0` as the baseline for future compares.

### Fixed
- N/A

## [0.1.1] - 2025-08-11
### Added
- **About tab** that renders `CHANGELOG.md` inside the app and shows the running app version.
- Page title now includes the version (e.g., `· v0.1.1`).

### Changed
- Branding image moved to **`assets/convergix_logo.png`** (PNG).
- Tabs updated to include **About**; small UI refactor.

### Fixed
- Minor robustness tweaks while loading the logo (won’t break the app if the file is missing).

## [0.1.0] - 2025-08-11
### Added
- **Branding & versioning**: App titled *Convergix DataMosaix View Explorer* with logo; visible `APP_VERSION`.
- **Connection workflow**: Connect to CDF (Cognite) with fields prefilled from `.env`.
- **Profiles (CDF)**: Load from grouped `.env` or save custom profiles to `/data/profiles.json` (persisted across restarts).
- **Browse CDF**: Load **Spaces → Models → Views** (latest versions), then fetch instances into memory.
- **Caching**: Uses `@st.cache_data` for list/fetch functions; avoids hashing Cognite client via leading `_` arg.
- **Download**: Export fetched data as CSV directly from the UI.
- **Filters**: Data Explorer tab with simple search, per-column filters (numeric ranges, dates, categorical, contains).
- **MySQL commit**: Replace-or-create table with pandas → SQLAlchemy; **URL.create** ensures passwords with special chars (e.g., `@`) work.
- **MySQL profiles**: Load from grouped `.env` or save custom profiles to `/data/mysql_profiles.json`; **Test connection** button.
- **Deployment**: Dockerfile + docker-compose with `/data` volume for persistent profiles and `extra_hosts` for host MySQL.
- **Healthcheck**: Streamlit health endpoint baked into Dockerfile.

### Changed
- Simplified UI cues: replaced colored badges with inline prompts like “Choose your Space here →”.
- App name updated to *Convergix DataMosaix View Explorer*.

### Fixed
- **Streamlit caching error**: `UnhashableParamError` resolved by using `_client` param in cached functions.
- **MySQL connection issue**: Special characters in credentials no longer break the DSN (uses `SQLAlchemy.URL.create`).

---

## Links
[Unreleased]: https://github.com/<org>/<repo>/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/<org>/<repo>/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/jaredcomer35/EAMS_Demo/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/jaredcomer35/EAMS_Demo/compare/v1.0.0...v1.1.0   
[1.0.0]: https://github.com/jaredcomer35/EAMS_Demo/compare/v0.1.1...v1.0.0  
[0.1.1]: https://github.com/jaredcomer35/EAMS_Demo/compare/v0.1.0...v0.1.1  
[0.1.0]: https://github.com/jaredcomer35/EAMS_Demo/releases/tag/v0.1.0
