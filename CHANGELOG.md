# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **DataMosaix Upload – Batch mode:** Select multiple MySQL tables and multiple target Views and run one batch upload.
  - Pairing strategies: **Zip tables ↔ views** (1:1 by order) or **All tables → first selected view**.
  - Auto-mapping of properties by exact name (case-insensitive).
  - Optional **sync delete**: remove nodes not present in the incoming payload (by `external_id`) before upload.
- **DataMosaix Upload – external_id default:** If the source contains a column named `external_id` (any case), it is used by default; otherwise, IDs are generated with a sensible prefix.
- **Logging:** More timestamped log entries for single and batch uploads.

### Changed
- **DataMosaix Upload – MySQL default limit:** Default “Max rows” for MySQL sources is now **30,000**.

### Fixed
- N/A

### Notes

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
[Unreleased]: https://github.com/jaredcomer35/EAMS_Demo/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/jaredcomer35/EAMS_Demo/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/jaredcomer35/EAMS_Demo/compare/v1.0.0...v1.1.0   
[1.0.0]: https://github.com/jaredcomer35/EAMS_Demo/compare/v0.1.1...v1.0.0  
[0.1.1]: https://github.com/jaredcomer35/EAMS_Demo/compare/v0.1.0...v0.1.1  
[0.1.0]: https://github.com/jaredcomer35/EAMS_Demo/releases/tag/v0.1.0
