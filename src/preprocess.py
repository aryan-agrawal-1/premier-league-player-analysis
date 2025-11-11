import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import STAT_TYPES
from abilities import ABILITY_FEATURE_GROUPS


def load_raw_data(data_root=None, season="2025"):
    # load data from data/raw/season
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent / "data" / "raw"
    data_root = Path(data_root)
    
    season_dir = data_root / str(season)
    frames = {}
    missing_types = []
    
    print(f"Looking for stat types in: {season_dir}")
    print(f"Expected stat types: {', '.join(STAT_TYPES)}")
    
    for stat_type in STAT_TYPES:
        file_path = season_dir / f"{stat_type}.parquet"
        if file_path.exists():
            print(f"Loading {stat_type}...")
            df = pd.read_parquet(file_path)
            
            # Drop columns that are completely empty
            empty_cols = [col for col in df.columns if df[col].isna().all()]
            if empty_cols:
                print(f"  Dropping {len(empty_cols)} empty columns: {empty_cols}")
                df = df.drop(columns=empty_cols)
            # add dataframe for that stat type to the frames dictionary
            frames[stat_type] = df
        else:
            missing_types.append(stat_type)
            print(f"Warning: {stat_type} not found at {file_path}")
    
    if missing_types:
        print(f"\nMissing stat type files: {', '.join(missing_types)}")
    print(f"Loaded {len(frames)}/{len(STAT_TYPES)} stat types")
    
    # frames is a dictionary of dataframes for each stat type
    return frames


def _series_equal(left, right, tol=1e-9):
    if len(left) != len(right):
        return False
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_vals = left.to_numpy(dtype=float)
        right_vals = right.to_numpy(dtype=float)
        both_nan = np.isnan(left_vals) & np.isnan(right_vals)
        compare_mask = ~both_nan
        if compare_mask.any():
            if not np.allclose(left_vals[compare_mask], right_vals[compare_mask], rtol=1e-9, atol=tol):
                return False
        return True
    return left.fillna("__NA__").equals(right.fillna("__NA__"))


def _deduplicate_columns(df, stat_type, column_sources):
    suffix = f"_{stat_type}"
    duplicate_cols = [col for col in df.columns if col.endswith(suffix)]
    for dup_col in duplicate_cols:
        base_col = dup_col[:-len(suffix)]
        if not base_col:
            continue
        if base_col not in df.columns:
            new_name = base_col or dup_col
            df = df.rename(columns={dup_col: new_name})
            column_sources.setdefault(new_name, set()).add(stat_type)
            if dup_col in column_sources and dup_col != new_name:
                del column_sources[dup_col]
            continue
        base_series = df[base_col]
        candidate_series = df[dup_col]
        if _series_equal(base_series, candidate_series):
            df = df.drop(columns=[dup_col])
            column_sources.setdefault(base_col, set()).add(stat_type)
            if dup_col in column_sources:
                del column_sources[dup_col]
            print(f"  Dropped duplicate column {dup_col} (identical to {base_col})")
            continue
        overlap_mask = base_series.notna() & candidate_series.notna()
        if overlap_mask.any():
            base_overlap = base_series[overlap_mask]
            cand_overlap = candidate_series[overlap_mask]
            if not _series_equal(base_overlap, cand_overlap):
                column_sources.setdefault(dup_col, set()).add(stat_type)
                continue
        base_non_na = base_series.notna().sum()
        cand_non_na = candidate_series.notna().sum()
        if base_non_na == 0 and cand_non_na > 0:
            df[base_col] = candidate_series
            df = df.drop(columns=[dup_col])
            column_sources.setdefault(base_col, set()).add(stat_type)
            if dup_col in column_sources:
                del column_sources[dup_col]
            print(f"  Replaced empty column {base_col} with data from {dup_col}")
            continue
        fill_mask = base_series.isna() & candidate_series.notna()
        if fill_mask.any():
            df.loc[fill_mask, base_col] = candidate_series.loc[fill_mask]
            df = df.drop(columns=[dup_col])
            column_sources.setdefault(base_col, set()).add(stat_type)
            if dup_col in column_sources:
                del column_sources[dup_col]
            print(f"  Filled missing values in {base_col} using {dup_col}")
            continue
        column_sources.setdefault(dup_col, set()).add(stat_type)
    return df


def merge_stat_tables(frames):
    # start with standard stats as base
    merged = frames["standard"].copy()
    column_sources = {}
    for col in merged.columns:
        column_sources[col] = {"standard"}
    
    # merge other stat types on player, team and season
    merge_keys = ["player", "team", "season"]
    
    # Identifier columns that should NOT be merged from other tables
    # (they're already in standard, no need to duplicate)
    identifier_cols = {"league", "nation", "pos", "age", "born"}
    
    skipped_types = []
    merged_types = ["standard"]
    
    for stat_type, df in frames.items():
        if stat_type == "standard":
            continue
        
        # Get columns to merge (exclude merge keys AND identifier columns)
        cols_to_merge = [
            col for col in df.columns 
            if col not in merge_keys and col not in identifier_cols
        ]
        
        if not cols_to_merge:
            skipped_types.append(stat_type)
            print(f"Skipping {stat_type} - no columns to merge (total columns: {len(df.columns)})")
            continue

        df_to_merge = df[merge_keys + cols_to_merge].copy()
        print(f"Merging {stat_type} ({len(df_to_merge)} rows, {len(cols_to_merge)} columns)...")

        merged = merged.merge(
            df_to_merge,
            on=merge_keys,
            how="outer",
            suffixes=("", f"_{stat_type}")
        )

        merged = _deduplicate_columns(merged, stat_type, column_sources)

        for col in merged.columns:
            if col not in column_sources:
                column_sources[col] = {stat_type}
        
        merged_types.append(stat_type)
    
    print(f"\nMerged stat types: {', '.join(merged_types)}")
    if skipped_types:
        print(f"Skipped stat types: {', '.join(skipped_types)}")
    
    return merged, column_sources


def classify_position(pos_str):
    if pd.isna(pos_str):
        return ["Unknown"]
    
    pos_str = str(pos_str).upper()
    positions = []
    
    if "GK" in pos_str:
        positions.append("Keeper")
    
    if "DF" in pos_str:
        positions.append("Defender")
    
    if "MF" in pos_str:
        positions.append("Midfielder")
    
    if "FW" in pos_str:
        positions.append("Attacker")
    
    if not positions:
        return ["Unknown"]
    
    return positions


def expand_positions(df):
    # Expand rows for players with multiple positions
    rows_list = []
    
    for idx, row in df.iterrows():
        positions = row.get("position", [])
        if not isinstance(positions, list):
            positions = [positions]
        
        for pos in positions:
            row_copy = row.copy()
            row_copy["position"] = pos
            rows_list.append(row_copy)
    
    expanded = pd.DataFrame(rows_list)
    return expanded.reset_index(drop=True)


def filter_minutes(df, min_minutes=100):
    # Find minutes column (currently Playing Time_Min)
    minutes_col = None
    for col in df.columns:
        if "Min" in col and "90" not in col:
            minutes_col = col
            break
    
    if minutes_col is None:
        print("Warning: Could not find minutes column")
        return df
    
    initial_count = len(df)
    df_filtered = df[df[minutes_col] >= min_minutes].copy()
    removed = initial_count - len(df_filtered)
    print(f"Filtered by minutes (>= {min_minutes}): {initial_count} -> {len(df_filtered)} (removed {removed})")
    
    return df_filtered


def normalize_per90(df):
    # Find 90s column
    col_90s = None
    for col in df.columns:
        if "90s" in col or col == "90s":
            col_90s = col
            break
    
    if col_90s is None:
        print("Warning: Could not find 90s column for normalization")
        return df
    
    df = df.copy()
    
    # Identify columns that should be normalized per 90
    # Skip identifier columns and already-normalized
    skip_cols = {
        "player", "team", "season", "league", "nation", "pos", "age", "born",
        "position", col_90s
    }
    
    # Also skip any columns that are identifier columns with prefixes
    skip_patterns = ["league", "nation", "pos", "age", "born"]
    
    # skip columns that are already per-90 (various formats)
    per90_indicators = [
        "per 90", "per90", "per_90", "/90", "_90", "90s"
    ]
    
    per90_cols = set()
    for col in df.columns:
        col_lower = col.lower()
        if any(indicator in col_lower for indicator in per90_indicators):
            per90_cols.add(col)
            continue
        if col_lower.endswith("90") and not col_lower.endswith("90s"):
            per90_cols.add(col)
        if col.startswith("Per 90 Minutes_"):
            per90_cols.add(col)
    
    # Find numeric columns that aren't already per-90
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_normalize = []
    for col in numeric_cols:
        # Skip if it's an identifier column
        if col in skip_cols:
            continue
        # Skip if it contains identifier patterns (like "shooting_born")
        if any(pattern in col.lower() for pattern in skip_patterns):
            continue
        # Skip if already per-90
        if col in per90_cols:
            continue
        # Skip if it's a merge suffix column
        if col.endswith("_y"):
            continue
        # Only normalize if we have valid 90s data
        if df[col_90s].notna().any():
            cols_to_normalize.append(col)
    
    print(f"Normalizing {len(cols_to_normalize)} columns to per-90...")
    
    for col in cols_to_normalize:
        per90_col = f"{col}_per90"
        df[per90_col] = df[col] / df[col_90s].replace(0, np.nan)
    
    return df


def _strip_rate_suffix(stat_name):
    if not isinstance(stat_name, str):
        return stat_name
    if stat_name.endswith("90s"):
        stripped = stat_name[:-3]
        return stripped or stat_name
    if stat_name.endswith("90"):
        stripped = stat_name[:-2]
        return stripped or stat_name
    return stat_name


def _extract_base_stat_name(col_name):
    # Extract the base stat name from various formats
    # "Per 90 Minutes_Gls" -> "Gls"
    # "Performance_Gls_per90" -> "Gls"
    # "Standard_Gls_per90" -> "Gls"
    # "Standard_Sh/90_per90" -> "Sh"
    
    # Remove "Per 90 Minutes_" prefix
    if col_name.startswith("Per 90 Minutes_"):
        return _strip_rate_suffix(col_name.replace("Per 90 Minutes_", ""))
    
    # Remove "_per90" suffix
    if col_name.endswith("_per90"):
        base = col_name[:-6]
    else:
        base = col_name
    
    # Remove per-90 indicators
    base = base.replace("/90", "").replace("_90", "")
    
    # Extract stat name after last underscore or colon
    if "_" in base:
        parts = base.split("_")
        # Skip prefixes like "Performance", "Standard", "Pass Types"
        stat_name = parts[-1]
        return _strip_rate_suffix(stat_name)
    elif ":" in base:
        stat_name = base.split(":")[-1]
        return _strip_rate_suffix(stat_name)
    
    return _strip_rate_suffix(base)


def select_features(df, position):
    # Exclude identifier columns and metadata
    identifier_cols = {
        "player", "team", "season", "league", "nation", "pos", "age", "born",
        "position", "Playing Time_MP", "Playing Time_Starts", "Playing Time_Min"
    }
    
    # Pattern to identify identifier columns (even with prefixes)
    identifier_patterns = ["league", "nation", "pos", "age", "born"]
    
    # Track which base stats we've seen to avoid duplicates
    seen_base_stats = {}
    
    feature_cols = []
    for col in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Skip explicit identifier columns
        if col in identifier_cols:
            continue
        # Skip columns containing identifier patterns
        if any(pattern in col.lower() for pattern in identifier_patterns):
            continue
        # Skip merge duplicates
        if col.endswith("_y"):
            continue
        # Skip 90s column itself (not a feature)
        if col.lower() == "90s" or col.lower().endswith("_90s"):
            continue
        # Skip double-normalized columns (e.g., "Per 90 Minutes_Gls_per90")
        if col.endswith("_per90") and "Per 90" in col:
            continue
        # Skip columns that normalize already-per-90 columns (e.g., "90s_per90")
        if col.endswith("_per90"):
            base_col = col[:-6]
            if base_col in df.columns and any(ind in base_col.lower() for ind in ["/90", "per 90", "per90"]):
                continue
        
        # Include columns that are per-90 (either from FBref "Per 90 Minutes_" 
        # or our normalized "_per90" columns)
        is_per90 = (
            "per90" in col.lower() or 
            "per 90" in col.lower() or 
            "per_90" in col.lower() or
            col.startswith("Per 90 Minutes_") or
            "/90" in col
        )
        
        if is_per90:
            base_stat = _extract_base_stat_name(col)
            
            # Prefer original "Per 90 Minutes_" columns over normalized ones
            if base_stat in seen_base_stats:
                existing_col = seen_base_stats[base_stat]
                # If current is "Per 90 Minutes_" and existing is normalized, replace
                if col.startswith("Per 90 Minutes_") and not existing_col.startswith("Per 90 Minutes_"):
                    feature_cols.remove(existing_col)
                    feature_cols.append(col)
                    seen_base_stats[base_stat] = col
                # Otherwise, skip this duplicate
                else:
                    continue
            else:
                feature_cols.append(col)
                seen_base_stats[base_stat] = col
    
    return feature_cols

# higher score for per 90 features
def _feature_preference_score(feature_name):
    score = 0
    if feature_name.startswith("Per 90 Minutes_"):
        score += 3
    lower_name = feature_name.lower()
    if lower_name.endswith("_per90"):
        score += 2
    if "per90" in lower_name or "per 90" in lower_name or "/90" in lower_name:
        score += 1
    if "_per90" in feature_name:
        score += 1
    return score

# 2 features with high correlation -> 1 must be dropped
def _choose_feature_to_drop(left, right, protected=None):
    protected = protected or set()
    if left in protected and right not in protected:
        return right
    if right in protected and left not in protected:
        return left
    left_score = _feature_preference_score(left)
    right_score = _feature_preference_score(right)
    if left_score > right_score:
        return right
    if right_score > left_score:
        return left
    if len(left) <= len(right):
        return right
    return left

# Compute paiwise Pearson correlation coefficients, anything highly related gets dropped
def detect_correlated_pairs(df, feature_cols, threshold=0.95, min_samples=10):
    matrix_columns = []
    for feature in feature_cols:
        # each feature returns a 'Series' data type from pandas, we change this to an array and append to matrix_cloumns
        series = df[feature]
        float_array = series.values.astype(float)
        matrix_columns.append(float_array)

    # Stack the cols into a 2D numpy array with shape (n rows, n features) so every column is a feature
    X = np.column_stack(matrix_columns)

    # mask for non-NaN values
    mask = np.isfinite(X)
    # convert the boolean mask to integers (True becomes 1, False becomes 0)
    mask_int = mask.astype(int)

    # Compute the Pairwise Valid Sample Count matrix (valid_count)
    # matrix multiplication (M_int.T @ M_int) calculates the sum of valid samples for every pair (i, j)
    # shape of valid_count is (n_features, n_features).
    # The diagonal elements valid_count[i, i] contain the total non-NaN samples for column i
    valid_count = mask_int.T @ mask_int

    n_rows, n_features = X.shape
    # Initialize arrays to store means standard deviations and validity flags.
    mu = np.zeros(n_features)
    sigma = np.zeros(n_features)
    # validity_mask tracks columns that have sufficient samples and non-zero std
    is_valid_col = np.ones(n_features, dtype=bool)

    for i in range(n_features):
        # extract values for column i
        xi = X[:, i]

        # drop NaNs
        finite_xi = xi[np.isfinite(xi)]
        n_valid = len(finite_xi)

        # check for insufficient samples and mark column as invalid if needed
        if n_valid < min_samples:
            is_valid_col[i] = False
            continue

        # compute mean and sd
        mu_i = np.mean(finite_xi)
        sigma_i = np.std(finite_xi, ddof=1)

        # check for zero standard deviation (constant column)
        if sigma_i == 0.0:
            is_valid_col[i] = False
            continue

        # Store calculated stats
        mu[i] = mu_i
        sigma[i] = sigma_i

        # standardise valid rows
        valid_indices = np.isfinite(xi)
        X[valid_indices, i] = (xi[valid_indices] - mu_i) / sigma_i

    Z = X.copy()
    Z[np.isnan(Z)] = 0.0
    # Compute raw covariance matrix
    C = Z.T @ Z

    dof = valid_count - 1
    denominator = np.maximum(dof, 1) # minimum 1

    # final correlation matrix
    R = C / denominator
    valid_rows = is_valid_col[:, np.newaxis]
    valid_cols = is_valid_col[np.newaxis, :]
    # The resulting valid_mask is True only if BOTH row and column feature are valid.
    valid_mask = valid_rows & valid_cols

    # Apply the mask to the signed correlation matrix R.
    R[~valid_mask] = 0.0
    R_signed = R.copy()

    # convert to abs correlations
    R_abs = np.abs(R_signed)

    # Zero out diagonal and lower triangle
    # np.triu(..., k=1) keeps the upper triangle (k=1 excludes the diagonal, k=0 includes it)
    # This prepares the matrix for fast iteration over unique pairs
    R_upper_abs = np.triu(R_abs, k=1) 

    correlated_pairs = []
    n_features = R_upper_abs.shape[0]

    # Iterate over the upper triangle (i < j)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            abs_r = R_upper_abs[i, j]
            count = valid_count[i, j]

            # Check if the pair meets both criteria
            if count >= min_samples and abs_r >= threshold:
                # Retrieve the signed correlation value
                signed_r = R_signed[i, j]
                # Append tuple: (feature_name_i, feature_name_j, signed_r, valid_count)
                correlated_pairs.append((
                    feature_cols[i], 
                    feature_cols[j], 
                    signed_r, 
                    count
                ))
    
    # sort key is the absolute value of the correlation (index 2), descending.
    sorted_pairs = sorted(
        correlated_pairs, 
        key=lambda x: abs(x[2]), 
        reverse=True
    )

    return sorted_pairs


# returns the features we want to keep and all the corellated pairs
def resolve_correlated_features(
    df,
    feature_cols,
    threshold=0.95,
    drop=False,
    protected=None,
    min_samples=10,
):
    try:
        correlated_pairs = detect_correlated_pairs(
            df,
            feature_cols,
            threshold=threshold,
            min_samples=min_samples,
        )
    except NotImplementedError as exc:
        print(f"Correlation analysis skipped: {exc}")
        return feature_cols, []

    if not correlated_pairs:
        print("No correlated feature pairs detected.")
        return feature_cols, []

    print(f"Detected {len(correlated_pairs)} correlated feature pairs (|r| >= {threshold}).")
    protected = protected or set()
    drop_candidates = set()
    for left, right, corr_value, overlap_count in correlated_pairs:
        print(
            f"  Pair: {left} & {right} -> corr={corr_value:.4f}, overlap={overlap_count}"
        )
        if not drop:
            continue
        to_remove = _choose_feature_to_drop(left, right, protected=protected)
        if to_remove in protected:
            print(f"    Skipping removal for protected feature: {to_remove}")
            continue
        drop_candidates.add(to_remove)

    if drop_candidates:
        print(f"Dropping {len(drop_candidates)} correlated features: {sorted(drop_candidates)}")
        filtered_features = [col for col in feature_cols if col not in drop_candidates]
    else:
        filtered_features = feature_cols

    return filtered_features, correlated_pairs


def _collect_protected_features():
    protected = set()
    for position_map in ABILITY_FEATURE_GROUPS.values():
        for columns in position_map.values():
            protected.update(columns)
    return protected


def standardize_features(df, feature_cols):
    # Standardise using numpy (mean=0, std=1)
    df_std = df.copy()
    
    means = {}
    stds = {}
    
    for col in feature_cols:
        values = df[col].values
        valid_mask = ~np.isnan(values)
        
        if valid_mask.sum() == 0:
            continue
        
        mean_val = np.mean(values[valid_mask])
        std_val = np.std(values[valid_mask])
        
        if std_val == 0:
            std_val = 1.0
        
        means[col] = mean_val
        stds[col] = std_val
        
        # standadisation formula (central limit theorem)
        df_std[col] = (values - mean_val) / std_val
    
    scaler_params = {"means": means, "stds": stds}
    
    return df_std, scaler_params


def preprocess_pipeline(
    data_root=None,
    season="2025",
    min_minutes=100,
    corr_threshold=0.95,
    drop_correlated=True,
    min_corr_samples=10,
    protect_core_features=True,
):
    print("=" * 60)
    print("Starting preprocessing pipeline...")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\nStep 1: Loading raw data...")
    frames = load_raw_data(data_root=data_root, season=season)
    
    if not frames:
        raise ValueError("No data frames loaded")
    
    # Step 2: Merge tables
    print("\nStep 2: Merging stat tables...")
    merged, column_sources = merge_stat_tables(frames)
    print(f"Merged shape: {merged.shape}")
    
    # Step 3: Filter by minutes
    print("\nStep 3: Filtering by minimum minutes...")
    merged = filter_minutes(merged, min_minutes=min_minutes)
    
    # Step 4: Classify positions
    print("\nStep 4: Classifying positions...")
    merged["position"] = merged["pos"].apply(classify_position)
    
    # Expand rows for players with multiple positions
    print("Expanding rows for players with multiple positions...")
    initial_rows = len(merged)
    merged = expand_positions(merged)
    expanded_rows = len(merged)
    print(f"Rows expanded: {initial_rows} -> {expanded_rows} (+{expanded_rows - initial_rows})")
    print(f"Position distribution:\n{merged['position'].value_counts()}")
    
    # Step 5: Normalize to per-90
    print("\nStep 5: Normalizing to per-90...")
    merged = normalize_per90(merged)
    
    # Step 6: Select features
    print("\nStep 6: Selecting features...")
    feature_cols = select_features(merged, None)
    print(f"Selected {len(feature_cols)} features")
    
    # Step 7: Correlation analysis
    print("\nStep 7: Correlation analysis...")
    protected_features = set()
    if protect_core_features:
        protected_features = _collect_protected_features()
        if protected_features:
            print(f"Protecting {len(protected_features)} core features from removal.")
    feature_cols, correlated_pairs = resolve_correlated_features(
        merged,
        feature_cols,
        threshold=corr_threshold,
        drop=drop_correlated,
        protected=protected_features,
        min_samples=min_corr_samples,
    )
    print(f"Post-correlation feature count: {len(feature_cols)}")
    
    # Step 8: Standardize features
    print("\nStep 8: Standardizing features...")
    merged_std, scaler_params = standardize_features(merged, feature_cols)
    
    # Step 9: Save processed data
    print("\nStep 9: Saving processed data...")
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_tables_dir = output_dir / "stat_tables"
    raw_tables_dir.mkdir(parents=True, exist_ok=True)
    for stat_type, df in frames.items():
        raw_table_path = raw_tables_dir / f"{stat_type}.parquet"
        df.to_parquet(raw_table_path, index=False)
        print(f"Saved raw {stat_type} table to {raw_table_path}")

    unscaled_output_path = output_dir / "player_vectors_unscaled.parquet"
    merged.to_parquet(unscaled_output_path, index=False)
    print(f"Saved unstandardized vectors to {unscaled_output_path}")

    output_path = output_dir / "player_vectors.parquet"
    merged_std.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Save scaler params as numpy format
    scaler_path = output_dir / "scaler_params.npz"
    np.savez(scaler_path, **scaler_params)
    print(f"Saved scaler params to {scaler_path}")

    column_sources_serializable = {}
    for col, sources in column_sources.items():
        column_sources_serializable[col] = sorted(list(sources))
    column_sources_path = output_dir / "column_sources.json"
    with column_sources_path.open("w", encoding="utf-8") as handle:
        json.dump(column_sources_serializable, handle, indent=2)
    print(f"Saved column source map to {column_sources_path}")

    keeper_cols = []
    for col, sources in column_sources.items():
        for source in sources:
            if source.startswith("keeper"):
                keeper_cols.append(col)
                break
    keeper_cols = sorted(set(keeper_cols))
    if keeper_cols:
        keeper_cols_path = output_dir / "keeper_features.json"
        with keeper_cols_path.open("w", encoding="utf-8") as handle:
            json.dump(keeper_cols, handle, indent=2)
        print(f"Keeper-specific feature columns: {len(keeper_cols)} (saved to {keeper_cols_path})")
    else:
        print("No keeper-specific feature columns detected.")
    
    if correlated_pairs:
        print("\nCorrelated feature pairs (top 10):")
        limit = min(10, len(correlated_pairs))
        for idx in range(limit):
            left, right, corr_value, overlap_count = correlated_pairs[idx]
            print(
                f"  {idx + 1:2d}. {left} vs {right} -> corr={corr_value:.4f}, overlap={overlap_count}"
            )
        if len(correlated_pairs) > limit:
            remaining = len(correlated_pairs) - limit
            print(f"  ... {remaining} additional pairs not shown.")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Final shape: {merged_std.shape}")
    print(f"Features: {len(feature_cols)}")
    print("\nFeature list:")
    for i, feat in enumerate(sorted(feature_cols), 1):
        print(f"  {i:3d}. {feat}")
    print("=" * 60)
    
    return merged_std, scaler_params, feature_cols, column_sources, correlated_pairs


if __name__ == "__main__":
    preprocess_pipeline()

