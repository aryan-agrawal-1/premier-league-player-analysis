# Premier League Data Analysis App
I used data from FBREF to analyse player data from the 2025-2026 premier league season. All the maths is done using only numpy to encourage full understanding (no scipy/scikit learn). AI was used to help with frontend implementation, all maths was implemented by me.

# Prerequisites
```bash
pip install -r requirements.txt
```

## Scheduled data refresh
- Automated workflow: `.github/workflows/data-refresh.yml` runs every 10 hours and on manual dispatch. It installs `requirements.txt` followed by `dev-requirements.txt` (for `soccerdata`) before executing `python src/data_loader.py`, `python src/preprocess.py`, and `python src/clustering.py`.
- Keep-alive strategy: if the refresh does not change data artefacts, the workflow updates `app/keepalive.txt` so Streamlit receives a heartbeat commit.
- Secrets: add any FBref or StatsBomb credentials (for example `SOCCERDATA_EMAIL`, `SOCCERDATA_PASSWORD`, or `SOCCERDATA_DATA_DIR`) as repository secrets and map them into the workflow via `env:` if your `.env` expects them.
- Local verification: run the three scripts above in a clean environment or invoke `act -j refresh-and-commit` to smoke-test the pipeline before merging workflow edits.

# Loading in the data
```bash
python src/data_loader.py
```

# Running the app locally
```bash
python src/data_loader.py
python src/preprocessing.py
python src/clustering.py
streamlit run app/streamlit_app.py
```

# Pre-processing
```bash
python src/preprocessing.py
```

## Picking Features
We drop 1/2 of any features that have a pairwise correlation of over 0.95. We use Pearson correlation coefficiants for this.

## Why do we standardise the features?
1. DIFFERENT SCALES: Stats have vastly different ranges:
   - Goals per 90: 0-2 (smaller numbers)
   - Passes per 90: 20-100 (larger numbers)
   - Without standardization, passes would dominate similarity calculations
2. EQUAL WEIGHTING: Standardization ensures each feature contributes equally to similarity. A 'high' value in goals means the same relative thing as a 'high' value in passes (both are ~2 standard deviations above average). Maybe this is something I can look at changing in future to weigh more 'important' contributions?

# Similarity
```bash
python src/similarity.py
```

Computed simply using cosine similarity with normalised values. Maybe I can do something more complex in future to get better values but works for now.

# Clustering
```bash
python src/clustering.py
```

## PCA (Principal Component Analysis)
1. DIMENSIONALITY REDUCTION: With 158 features, visualizing players is impossible. PCA reduces this to 2D for visualization while preserving as much variance as possible.
2. VARIANCE EXPLANATION: Currently captures ~38% variance in 2D (PC1: 23%, PC2: 15%). This is reasonable given the high dimensionality - to capture 80% variance you'd need ~12 components.
3. IMPLEMENTATION: Uses SVD (Singular Value Decomposition) with numpy - no scikit-learn. The principal components show which stat combinations define different player styles (e.g., PC1 might represent "Attacking Productivity").

## K-Means Clustering
1. POSITIONAL CLUSTERING: We cluster separately for each position (Attacker, Midfielder, Defender, Keeper) to find stylistic archetypes within each role. An attacking midfielder and defensive midfielder are both midfielders but have very different styles.
2. K-MEANS++ INITIALIZATION: Uses k-means++ to pick initial cluster centers far apart, reducing the chance of poor local optima. Default clusters: 4 for Attackers/Midfielders/Defenders, 2 for Keepers.
3. STYLISTIC GROUPS: Each cluster represents a playing style (e.g., "Creative Playmaker", "Ball-Winning Destroyer", "Poacher"). Players in the same cluster have similar statistical profiles.
4. MINIMIZING INERTIA: The algorithm minimizes within-cluster sum of squares (inertia) - lower inertia means tighter, more similar clusters. Maybe I can experiment with different numbers of clusters per position or use other metrics like silhouette score in future.

# Visuals
```bash
python -c "from visuals import league_scatter, player_radar_chart, team_profile_heatmap"
```

## What do the visuals cover?
1. PCA MAP: `league_scatter` draws the 2D projection and keeps hover details light so you can scan clusters quickly. Pass in the dataframe returned from preprocessing/clustering with `pc1`, `pc2`, and metadata columns.
2. RADAR COMPARISON: `player_radar_chart` overlays any two numeric vectors (player vs centroid, player vs peer). Scale inputs to positive ranges before calling so the plot stays readable.
3. TEAM SNAPSHOT: `team_profile_heatmap` turns a team-feature matrix into a heatmap, automatically adapting the colour scale to the supplied values.
4. PERFORMANCE: All helpers only depend on numpy/pandas/plotly and avoid extra copies, so you can reuse the same figures in notebooks or Streamlit without reloading data.

# Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## What does the app include?
1. LOADS FROM CACHE: `PlayerVectorStore`, PCA projection, and clustering artefacts are cached with `st.cache_resource`/`st.cache_data`, so reruns stay fast even with large tables.
2. FILTER SIDEBAR: Season, position, team, and minutes sliders drive all downstream views. Filters match preprocessing rules (minimum minutes uses the same column heuristic).
3. LEAGUE MAP: PCA scatter with selectable colour coding (position, team, cluster). Hover shows key stats plus any extra columns like minutes.
4. SIMILARITY PANEL: Dropdown picks a player; table shows top cosine matches (same position), and the radar compares that player either to their cluster centroid, closest peer, or league average fallback.
5. TEAM PROFILES: Heatmap highlights variance-heavy features at the team level so you can compare club footprints quickly.
6. RUNNING LOCALLY: Activate your environment, install `requirements.txt`, then launch the command above and open the provided local URL. Streamlit auto-reloads when you tweak Python modules.
