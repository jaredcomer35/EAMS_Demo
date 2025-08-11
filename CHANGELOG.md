# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- (placeholder) Append mode for MySQL commits.
- (placeholder) Primary key selection & indexing.
- (placeholder) Encryption-at-rest for saved profiles via Docker secrets.
- (placeholder) Admin page for managing saved profiles.

### Changed
- (placeholder) Improve large table rendering performance.

### Fixed
- (placeholder) Minor UI polish and error message clarity.

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

## Release process
1. Update `APP_VERSION` in `app.py` (e.g., `0.1.2`).  
2. Add a new section under **[Unreleased]** → move entries into **[x.y.z] - YYYY-MM-DD**.  
3. Commit changes and tag the release (`git tag vX.Y.Z`; `git push --tags`).  
4. (Optional) Update compare links below.

## Links
> If you host this in GitHub/GitLab, you can wire compare links like:
>
> ```text
> [Unreleased]: https://github.com/ORG/REPO/compare/v0.1.1...HEAD
> [0.1.1]: https://github.com/ORG/REPO/releases/tag/v0.1.1
> [0.1.0]: https://github.com/ORG/REPO/releases/tag/v0.1.0
> ```
