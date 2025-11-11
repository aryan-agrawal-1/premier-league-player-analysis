import json
from pathlib import Path

import numpy as np
import pandas as pd


# Feature groups resolved from the processed dataset
ABILITY_FEATURE_GROUPS = {
    "Keeper": {
        "goalkeeping": [
            "Performance_Save%",
            "Performance_CS%",
            "Expected_PSxG+/-",
            "Performance_Saves",
            "Performance_GA90",
        ],
        "distribution": [
            "Passes_Att (GK)",
            "Passes_Launch%",
            "Launched_Cmp%",
            "Goal Kicks_Launch%",
            "Sweeper_#OPA/90",
        ],
    },
    "Defender": {
        "defensive": [
            "Tackles_Tkl_per90",
            "Tackles_TklW_per90",
            "Blocks_Blocks_per90",
            "Int_per90",
            "Performance_Recov_per90",
            "Tkl+Int_per90",
        ],
        "progressive": [
            "Progression_PrgP_per90",
            "Progression_PrgC_per90",
            "Carries_PrgC_per90",
            "Carries_PrgDist_per90",
            "Pass Types_Crs_per90",
            "Receiving_PrgR_per90",
        ],
    },
    "Midfielder": {
        "defensive": [
            "Tackles_Mid 3rd_per90",
            "Challenges_Tkl_per90",
            "Performance_TklW_per90",
            "Performance_Fls_per90",
            "Performance_Recov_per90",
            "Performance_Int_per90",
        ],
        "attacking": [
            "Per 90 Minutes_xG+xAG",
            "Per 90 Minutes_Ast",
            "SCA_SCA90",
            "GCA_GCA90",
            "KP_per90",
            "Carries_1/3_per90",
        ],
    },
    "Attacker": {
        "shooting": [
            "Per 90 Minutes_Gls",
            "Per 90 Minutes_xG",
            "Standard_SoT/90",
            "Expected_G-xG_per90",
            "Per 90 Minutes_npxG",
        ],
        "technical": [
            "Carries_Carries_per90",
            "Take-Ons_Succ_per90",
            "Take-Ons_Succ%",
            "Pass Types_TB_per90",
            "SCA Types_PassLive_per90",
            "Receiving_PrgR_per90",
        ],
    },
}


# Weighting schema, Each bucket's weights sum to 1.0 so the aggregation behaves like a weighted mean. Callers can supply
# their own mapping to override any subset
ABILITY_FEATURE_WEIGHTS = {
    "Keeper": {
        "goalkeeping": {
            "Performance_Save%": 0.30,
            "Performance_CS%": 0.10,
            "Expected_PSxG+/-": 0.30,
            "Performance_Saves": 0.15,
            "Performance_GA90": 0.15,
        },
            "distribution": {
            "Passes_Att (GK)": 0.10,
            "Passes_Launch%": 0.20,
            "Launched_Cmp%": 0.20,
            "Goal Kicks_Launch%": 0.25,
            "Sweeper_#OPA/90": 0.25,
        },
    },
    "Defender": {
        "defensive": {
            "Tackles_Tkl_per90": 0.20,
            "Tackles_TklW_per90": 0.15,
            "Blocks_Blocks_per90": 0.15,
            "Int_per90": 0.15,
            "Performance_Recov_per90": 0.15,
            "Tkl+Int_per90": 0.20,
        },
        "progressive": {
            "Progression_PrgP_per90": 0.25,
            "Progression_PrgC_per90": 0.20,
            "Carries_PrgC_per90": 0.15,
            "Carries_PrgDist_per90": 0.15,
            "Pass Types_Crs_per90": 0.10,
            "Receiving_PrgR_per90": 0.15,
        },
    },
    "Midfielder": {
        "defensive": {
            "Tackles_Mid 3rd_per90": 0.20,
            "Challenges_Tkl_per90": 0.15,
            "Performance_TklW_per90": 0.10,
            "Performance_Fls_per90": 0.05,
            "Performance_Recov_per90": 0.25,
            "Performance_Int_per90": 0.25,
        },
        "attacking": {
            "Per 90 Minutes_xG+xAG": 0.20,
            "Per 90 Minutes_Ast": 0.15,
            "SCA_SCA90": 0.15,
            "GCA_GCA90": 0.15,
            "KP_per90": 0.15,
            "Carries_1/3_per90": 0.20,
        },
    },
    "Attacker": {
        "shooting": {
            "Per 90 Minutes_Gls": 0.30,
            "Per 90 Minutes_xG": 0.25,
            "Standard_SoT/90": 0.15,
            "Expected_G-xG_per90": 0.10,
            "Per 90 Minutes_npxG": 0.20,
        },
        "technical": {
            "Carries_Carries_per90": 0.10,
            "Take-Ons_Succ_per90": 0.25,
            "Take-Ons_Succ%": 0.20,
            "Pass Types_TB_per90": 0.15,
            "SCA Types_PassLive_per90": 0.15,
            "Receiving_PrgR_per90": 0.15,
        },
    },
}



# Metrics whose larger values should penalise a player. They will be flipped in
# the weighting step so that every ability score remains "higher is better".
NEGATIVE_FEATURES = {
    "Performance_GA90",
    "Performance_Fls_per90",
}


def get_bucket_feature_map(vector_store, custom_map=None):
    # Return the feature map to use for ability calculations.

    # verifies that each referenced column exists in the processed dataset

    feature_cols = set(vector_store.feature_cols)
    feature_map = custom_map or ABILITY_FEATURE_GROUPS

    missing = {}
    resolved = {}
    for position, buckets in feature_map.items():
        position_payload = {}
        for bucket, columns in buckets.items():
            present = [col for col in columns if col in feature_cols]
            position_payload[bucket] = present
            absent = sorted(set(columns) - feature_cols)
            if absent:
                missing.setdefault(position, {})[bucket] = absent
        resolved[position] = position_payload

    if missing:
        raise ValueError(
            "Ability feature map references missing columns: "
            + json.dumps(missing, indent=2)
        )

    return resolved


def get_feature_index_map(vector_store):
    # Build a reusable lookup so we can slice the feature matrix efficiently

    return {name: idx for idx, name in enumerate(vector_store.feature_cols)}


def get_position_indices(vector_store, position):
    # Return the dataframe row indices for a specific positional label

    positions = vector_store.df["position"].fillna("Unknown").to_numpy()
    mask = positions == position
    return np.where(mask)[0]


def slice_feature_matrix(vector_store, row_indices, feature_names, index_map=None):
    # pull a feature sub-matrix for the requested rows and columns

    if index_map is None:
        index_map = get_feature_index_map(vector_store)

    column_indices = [index_map[name] for name in feature_names]
    return vector_store.feature_matrix[np.ix_(row_indices, column_indices)]


def compute_weighted_bucket_scores(feature_matrix, feature_names, feature_weights=None):
    if feature_matrix.size == 0 or not feature_names:
        return np.zeros(feature_matrix.shape[0], dtype=float)

    n_features = len(feature_names)

    # Resolve weights to a numpy vector aligned with feature_names.
    if feature_weights is None:
        weights = np.full(n_features, 1.0 / n_features, dtype=float)
    elif isinstance(feature_weights, dict):
        weights = np.array(
            [float(feature_weights.get(name, 0.0)) for name in feature_names],
            dtype=float,
        )
    else:
        weights = np.asarray(feature_weights, dtype=float)
        if weights.shape != (n_features,):
            raise ValueError("feature_weights sequence must match feature_names length")

    # If the supplied weights sum to zero (e.g. all missing), fall back to uniform.
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or np.isclose(weight_sum, 0.0):
        weights = np.full(n_features, 1.0 / n_features, dtype=float)
    else:
        weights = weights / weight_sum

    matrix = feature_matrix.astype(float, copy=True)

    for col_idx, name in enumerate(feature_names):
        if name in NEGATIVE_FEATURES:
            matrix[:, col_idx] = -matrix[:, col_idx]

    weighted = matrix * weights
    with np.errstate(invalid="ignore"):
        scores = np.nanmean(weighted, axis=1)

    # Replace NaN (all-missing rows) with zeros so downstream z-scores stay defined.
    return np.nan_to_num(scores, nan=0.0)


def normalize_scores_within_position(raw_scores):

    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim != 1:
        scores = scores.reshape(-1)

    if scores.size == 0:
        return np.zeros_like(scores, dtype=float)

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(scores)
        std = np.nanstd(scores)

    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or np.isclose(std, 0.0):
        std = 1.0

    z_scores = (scores - mean) / std
    return np.nan_to_num(z_scores, nan=0.0)


def compile_position_scores(vector_store, position, bucket_map, feature_weights=None):

    indices = get_position_indices(vector_store, position)
    if indices.size == 0:
        return {"indices": indices, "raw": {}, "z": {}}

    index_map = get_feature_index_map(vector_store)
    raw_scores = {}
    z_scores = {}

    for bucket_name, feature_names in bucket_map.items():
        if not feature_names:
            continue

        bucket_weights = None
        if isinstance(feature_weights, dict):
            bucket_weights = feature_weights.get(bucket_name)
        elif feature_weights is not None:
            bucket_weights = feature_weights

        sub_matrix = slice_feature_matrix(
            vector_store,
            indices,
            feature_names,
            index_map=index_map,
        )

        raw = compute_weighted_bucket_scores(sub_matrix, feature_names, bucket_weights)
        z = normalize_scores_within_position(raw)

        raw_scores[bucket_name] = raw
        z_scores[bucket_name] = z

    return {"indices": indices, "raw": raw_scores, "z": z_scores}


def build_ability_dataframe(vector_store, feature_map=None, feature_weights=None):

    resolved_map = feature_map or get_bucket_feature_map(vector_store)
    frames = []

    if feature_weights is None:
        feature_weights = ABILITY_FEATURE_WEIGHTS

    for position, bucket_map in resolved_map.items():
        if not bucket_map:
            continue

        position_weights = None
        if isinstance(feature_weights, dict):
            position_weights = feature_weights.get(position)

        scores = compile_position_scores(
            vector_store,
            position,
            bucket_map,
            feature_weights=position_weights,
        )

        if scores["indices"].size == 0:
            continue

        meta_cols = list(vector_store.metadata_cols or [])
        if "position" not in meta_cols:
            meta_cols.append("position")

        meta_frame = vector_store.df.iloc[scores["indices"]][meta_cols].copy()
        meta_frame["position"] = position

        for bucket_name in bucket_map.keys():
            if bucket_name not in scores["raw"]:
                continue
            meta_frame[f"{bucket_name}_raw"] = scores["raw"][bucket_name]
            meta_frame[f"{bucket_name}_z"] = scores["z"][bucket_name]

        frames.append(meta_frame)

    if not frames:
        return pd.DataFrame(), resolved_map

    combined = pd.concat(frames, ignore_index=True)
    return combined, resolved_map


def export_feature_map(output_path=None, feature_map=None):
   # Persist the resolved feature map

    if feature_map is None:
        feature_map = ABILITY_FEATURE_GROUPS

    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "ability_feature_map.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_map, handle, indent=2)

    return output_path
