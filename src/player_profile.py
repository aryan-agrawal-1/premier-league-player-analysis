import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from abilities import RADAR_BUCKET_DEFINITIONS, build_radar_bucket_dataframe
from similarity import PlayerVectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
STAT_TABLE_DIR = PROCESSED_DIR / "stat_tables"
FEATURE_LABEL_MAP_PATH = PROCESSED_DIR / "feature_label_map.json"

HERO_STATS = [
    ("Playing Time_Min", "Minutes"),
    ("Performance_Gls", "Goals"),
    ("Performance_Ast", "Assists"),
]

STAT_TABLE_TABS = [
    ("standard", "Overview"),
    ("shooting", "Shooting"),
    ("passing", "Passing"),
    ("passing_types", "Passing Types"),
    ("goal_shot_creation", "Shot & Goal Creation"),
    ("possession", "Possession"),
    ("defense", "Defending"),
    ("playing_time", "Playing Time"),
    ("misc", "Misc"),
    ("keeper", "Goalkeeping"),
    ("keeper_adv", "Advanced GK"),
]

STAT_META_COLUMNS = {
    "player",
    "team",
    "season",
    "league",
    "nation",
    "pos",
    "position",
    "age",
    "born",
    "squad",
}

RAW_STAT_LABEL_OVERRIDES = {
    "90s": "90s Played",
    "90s_keeper_adv": "90s Played (Advanced GK)",
    "Aerial Duels_Lost": "Aerial Duels Lost",
    "Aerial Duels_Won%": "Aerial Duels Win %",
    "Ast": "Assists",
    "Att": "Pass Attempts",
    "Blocks_Blocks": "Total Blocks",
    "Blocks_Sh": "Total Shots Blocked",
    "Carries_1/3": "Final Third Entries",
    "Carries_PrgC": "Progressive Carries",
    "Carries_PrgDist": "Progressive Distance",
    "Carries_TotDist": "Total Distance Carried",
    "Challenges_Att": "Attempted Challenges",
    "Challenges_Tkl": "Tackles",
    "Challenges_Tkl%": "Tackles Won %",
    "Crosses_Opp": "Crosses Faced",
    "Crosses_Stp": "Crosses Stopped",
    "Crosses_Stp%": "Crosses Stopped %",
    "Expected_PSxG": "Post-Shot xG",
    "Expected_PSxG+/-": "Post-Shot xG Differential",
    "Expected_PSxG/SoT": "Post-Shot xG per SoT",
    "Expected_npxG": "Non-Penalty xG",
    "Expected_npxG+xAG": "Non-Penalty xG + xAG",
    "Expected_npxG/Sh": "Non-Penalty xG per Shot",
    "Expected_xAG": "Expected Assisted Goals (xAG)",
    "Expected_xG": "Expected Goals (xG)",
    "GCA Types_Def": "Goal-Creating Actions: Defensive Actions",
    "GCA Types_Fld": "Fouls Won",
    "GCA Types_PassDead": "Goal-Creating Actions: Dead-Ball Passes",
    "GCA Types_PassLive": "Goal-Creating Actions: Live Passes",
    "GCA Types_Sh": "Shots",
    "GCA Types_TO": "Take-Ons",
    "GCA_GCA90": "Goal-Creating Actions per 90",
    "Goal Kicks_Att": "Goal Kick Attempts",
    "Goal Kicks_AvgLen": "Goal Kick Avg Length",
    "Goal Kicks_Launch%": "Goal Kick Launch %",
    "Goals_CK": "Goals Conceded from Corner Kicks",
    "Goals_FK": "Goals Conceded from Free Kicks",
    "Goals_GA": "Goals Conceded",
    "Goals_OG": "Goals Conceded from Own Goals",
    "Goals_PKA": "Goals Conceded from Penalties",
    "Launched_Att": "Launched Pass Attempts",
    "Launched_Cmp": "Launched Passes Completed",
    "Launched_Cmp%": "Launched Passes Completion %",
    "Long_Att": "Long Pass Attempts",
    "Long_Cmp": "Long Passes Completed",
    "Long_Cmp%": "Long Passes Completion %",
    "Medium_Att": "Medium Pass Attempts",
    "Medium_Cmp": "Medium Passes Completed",
    "Medium_Cmp%": "Medium Passes Completion %",
    "Outcomes_Cmp": "Outcomes: Completed",
    "Pass Types_FK": "Pass Types: Free Kicks",
    "Passes_Att (GK)": "Passes Attempted (GK)",
    "Passes_AvgLen": "Pass Avg Length",
    "Passes_Launch%": "Pass Launch %",
    "Passes_Thr": "Throws",
    "Penalty Kicks_PKA": "Penalties Faced",
    "Penalty Kicks_PKatt": "Penalty Attempts Faced",
    "Penalty Kicks_PKm": "Penalties Missed",
    "Penalty Kicks_PKsv": "Penalties Saved",
    "Penalty Kicks_Save%": "Penalty Save %",
    "Performance_Ast": "Assists",
    "Performance_CS": "Clean Sheets",
    "Performance_CS%": "Clean Sheet %",
    "Performance_Crs": "Crosses",
    "Performance_D": "Draws",
    "Performance_Fld": "Fouls Drawn",
    "Performance_G+A": "Goals + Assists",
    "Performance_G-PK": "Non-PK Goals",
    "Performance_GA": "Goals Against",
    "Performance_GA90": "Goals Against per 90",
    "Performance_Gls": "Goals",
    "Performance_Int": "Interceptions",
    "Performance_L": "Losses",
    "Performance_Off": "Offsides",
    "Performance_Save%": "Save %",
    "Performance_Saves": "Saves",
    "Performance_SoTA": "Shots on Target Faced",
    "Performance_TklW": "Tackles Won",
    "Performance_W": "Wins",
    "Playing Time_90s": "90s Played",
    "Playing Time_90s_keeper": "90s Played",
    "Playing Time_MP_keeper": "Matches Played",
    "Playing Time_Min_keeper": "Minutes Played",
    "Playing Time_Starts_keeper": "Starts",
    "PrgP": "Progressive Passes",
    "Receiving_PrgR": "Progressive Receptions",
    "SCA Types_Sh": "Shots",
    "SCA_SCA90": "Shot-Creating Actions per 90",
    "Short_Att": "Short Pass Attempts",
    "Short_Cmp": "Short Passes Completed",
    "Short_Cmp%": "Short Passes Completion %",
    "Standard_Dist": "Avg Shot Distance",
    "Standard_G/Sh": "Goals per Shot",
    "Standard_G/SoT": "Goals per Shot on Target",
    "Standard_Gls": "Goals",
    "Standard_PK": "Penalty Goals",
    "Standard_PKatt": "Penalty Attempts",
    "Standard_SoT%": "Shots on Target %",
    "Starts_Mn/Start": "Minutes per Start",
    "Starts_Starts": "Matches Started",
    "Subs_Mn/Sub": "Minutes per Sub App",
    "Sweeper_#OPA": "Defensive Actions Outside Box",
    "Sweeper_AvgDist": "Sweeper Keeper: Avg Distance",
    "Take-Ons_Att": "Take-Ons Attempted",
    "Take-Ons_Succ%": "Take-On Success %",
    "Take-Ons_Tkld%": "Take-On Tackled %",
    "Team Success (xG)_On-Off": "Team Success (xG): On/Off Differential",
    "Team Success (xG)_xG+/-90": "xG Differential per 90",
    "Team Success_+/-90": "Goal Differential per 90",
    "Team Success_On-Off": "Team Success: On/Off Differential",
    "Touches_Att 3rd": "Touches in Attacking Third",
    "Touches_Def 3rd": "Touches in Defensive Third",
    "Touches_Live": "Live-Ball Touches",
    "Touches_Mid 3rd": "Touches in Middle Third",
    "xAG": "Expected Assisted Goals",
}


def _normalize_stat_key(name):
    if name.endswith("_per90"):
        return name[:-6]
    if name.endswith("/90"):
        return name[:-3]
    return name


def _strip_per90_phrase(label):
    cleaned = label.replace(" per 90", "")
    cleaned = cleaned.replace("Per 90", "Per 90")
    return " ".join(cleaned.split())


def _build_stat_label_lookup():
    lookup = {}
    if FEATURE_LABEL_MAP_PATH.exists():
        with FEATURE_LABEL_MAP_PATH.open() as handle:
            payload = json.load(handle)
        original_map = payload.get("original_to_label", {})
        for original, label in original_map.items():
            base_key = _normalize_stat_key(original)
            if base_key == original:
                continue
            lookup[base_key] = _strip_per90_phrase(label)
    lookup.update(RAW_STAT_LABEL_OVERRIDES)
    return lookup


STAT_LABEL_LOOKUP = _build_stat_label_lookup()


def _label_from_lookup(name):
    if not isinstance(name, str):
        return None
    if name in STAT_LABEL_LOOKUP:
        return STAT_LABEL_LOOKUP[name]
    normalized = _normalize_stat_key(name)
    return STAT_LABEL_LOOKUP.get(normalized)


def _clean_numeric(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _align_raw_frame(raw_df, target_index):
    if raw_df is None:
        return None
    if not raw_df.index.equals(target_index):
        return raw_df.reindex(target_index)
    return raw_df


@lru_cache(maxsize=1)
def load_player_vector_store():
    return PlayerVectorStore()


@lru_cache(maxsize=1)
def load_raw_player_vectors():
    path = PROCESSED_DIR / "player_vectors_unscaled.parquet"
    return pd.read_parquet(path)


@lru_cache(maxsize=None)
def load_stat_table(table_name):
    path = STAT_TABLE_DIR / ("%s.parquet" % table_name)
    if not path.exists():
        raise FileNotFoundError("Stat table %s not found" % table_name)
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_radar_bucket_frame():
    store = load_player_vector_store()
    raw_df = _align_raw_frame(load_raw_player_vectors(), store.df.index)
    frame, bucket_weights = build_radar_bucket_dataframe(
        store,
        feature_frame=raw_df,
        bucket_defs=RADAR_BUCKET_DEFINITIONS,
    )
    return frame, bucket_weights


def _select_player_index(store_df, player_name, team=None, season=None):
    mask = store_df["player"] == player_name
    if team:
        mask = mask & (store_df["team"] == team)
    if season and "season" in store_df.columns:
        mask = mask & (store_df["season"] == season)
    matches = store_df[mask]
    if matches.empty and season and "season" in store_df.columns:
        mask = (store_df["player"] == player_name)
        if team:
            mask = mask & (store_df["team"] == team)
        matches = store_df[mask]
    if matches.empty:
        raise ValueError("Player %s not found in processed vectors" % player_name)
    return matches.index[0]


def _build_peer_frame(raw_df, position, season=None):
    peers = raw_df[raw_df["position"] == position]
    if season and "season" in raw_df.columns:
        season_peers = peers[peers["season"] == season]
        if not season_peers.empty:
            peers = season_peers
    return peers


def _build_bucket_tables(player_row, peer_df, bucket_weights):
    tables = {}
    for bucket, weights in bucket_weights.items():
        entries = []
        for feature, weight in weights.items():
            if feature not in peer_df.columns:
                continue
            raw_value = _clean_numeric(player_row.get(feature))
            entries.append(
                {
                    "stat": feature,
                    "value": 0.0 if np.isnan(raw_value) else raw_value,
                    "weight": float(weight),
                }
            )
        tables[bucket] = pd.DataFrame(entries)
    return tables


def _normalize_bucket_scores(radar_row, cohort_frame, bucket_config):
    payload = []
    for bucket in bucket_config.keys():
        raw_col = f"{bucket}_raw"
        raw_value = 0.0
        if radar_row is not None and raw_col in radar_row:
            raw_value = _clean_numeric(radar_row.get(raw_col)) or 0.0
        cohort_series = None
        if cohort_frame is not None and raw_col in cohort_frame.columns:
            cohort_series = pd.to_numeric(cohort_frame[raw_col], errors="coerce")
        max_value = 0.0
        if cohort_series is not None and not cohort_series.dropna().empty:
            max_value = float(np.nanmax(np.abs(cohort_series.to_numpy())))
            if not np.isfinite(max_value):
                max_value = 0.0
        if max_value <= 0.0:
            max_value = abs(raw_value) if abs(raw_value) > 0 else 1.0

        normalized = 0.0
        if raw_value > 0 and max_value > 0:
            normalized = (raw_value / max_value) * 100.0
            normalized = min(max(normalized, 0.0), 100.0)

        payload.append(
            {
                "bucket": bucket,
                "score": float(normalized),
                "raw_value": float(raw_value),
                "max_value": float(max_value),
            }
        )
    return payload


def _is_raw_stat_column(name):
    lowered = name.lower()
    return (
        "per 90" not in lowered
        and "per90" not in lowered
        and "/90" not in name
    )


def _format_stat_label(column):
    label = _label_from_lookup(column)
    if label:
        return label
    if not isinstance(column, str):
        return column
    if "_" in column:
        prefix, suffix = column.split("_", 1)
        prefix_clean = " ".join(prefix.replace("_", " ").split())
        suffix_clean = " ".join(suffix.replace("_", " ").split())
        if prefix_clean and suffix_clean:
            return "%s: %s" % (prefix_clean.strip().title(), suffix_clean.strip())
    return " ".join(column.replace("_", " ").split()).title()


def _format_stat_section(series):
    records = []
    for column, value in series.items():
        if column in STAT_META_COLUMNS:
            continue
        if not _is_raw_stat_column(column):
            continue
        if pd.isna(value):
            continue
        records.append(
            {
                "stat": _format_stat_label(column),
                "value": value,
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)


def _build_stat_sections(player_name, team=None, season=None):
    sections = []
    for table_name, label in STAT_TABLE_TABS:
        try:
            table = load_stat_table(table_name)
        except FileNotFoundError:
            continue
        if "player" not in table.columns:
            continue
        mask = table["player"] == player_name
        if team and "team" in table.columns:
            mask = mask & (table["team"] == team)
        if season and "season" in table.columns:
            mask = mask & (table["season"] == season)
        row = table[mask]
        if row.empty and season and "season" in table.columns:
            mask = table["player"] == player_name
            if team and "team" in table.columns:
                mask = mask & (table["team"] == team)
            row = table[mask]
        if row.empty:
            continue
        series = row.iloc[0]
        formatted = _format_stat_section(series)
        if formatted is None or formatted.empty:
            continue
        sections.append(
            {
                "key": table_name,
                "label": label,
                "data": formatted,
            }
        )
    return sections


def _format_bucket_feature_list(bucket_weights):
    formatted = {}
    for bucket, weights in bucket_weights.items():
        entries = []
        for feature in weights.keys():
            label = _label_from_lookup(feature)
            if not label:
                label = " ".join(
                    feature.replace("_per90", "").replace("_", " ").split()
                ).title()
            entries.append(label)
        if entries:
            formatted[bucket] = entries
    return formatted


def _build_hero_metrics(raw_row):
    metrics = []
    for column, label in HERO_STATS:
        if column not in raw_row:
            continue
        value = _clean_numeric(raw_row.get(column))
        if np.isnan(value):
            continue
        metrics.append({"label": label, "value": value, "column": column})
    return metrics


def get_player_profile(player_name, team=None, season=None):
    store = load_player_vector_store()
    raw_df = _align_raw_frame(load_raw_player_vectors(), store.df.index)
    radar_frame, bucket_weights = load_radar_bucket_frame()

    idx = _select_player_index(store.df, player_name, team=team, season=season)
    meta_row = store.df.loc[idx]
    raw_row = raw_df.loc[idx]
    position = meta_row.get("position", "Unknown")
    resolved_season = meta_row.get("season")

    bucket_config = bucket_weights.get(position, {})
    radar_row = radar_frame.loc[idx] if idx in radar_frame.index else None

    cohort_frame = radar_frame[radar_frame["position"] == position]
    if resolved_season and "season" in radar_frame.columns:
        season_frame = cohort_frame[cohort_frame["season"] == resolved_season]
        if not season_frame.empty:
            cohort_frame = season_frame

    radar_payload = _normalize_bucket_scores(radar_row, cohort_frame, bucket_config)

    peer_df = _build_peer_frame(raw_df, position, season=resolved_season)
    bucket_tables = _build_bucket_tables(raw_row, peer_df, bucket_config)
    stat_sections = _build_stat_sections(
        player_name,
        team=meta_row.get("team"),
        season=resolved_season,
    )
    bucket_feature_list = _format_bucket_feature_list(bucket_config)

    metadata = {
        "player": meta_row.get("player"),
        "team": meta_row.get("team"),
        "nation": meta_row.get("nation"),
        "position": position,
        "season": resolved_season,
        "age": meta_row.get("age"),
        "minutes": _clean_numeric(raw_row.get("Playing Time_Min")),
    }

    return {
        "index": idx,
        "metadata": metadata,
        "hero_metrics": _build_hero_metrics(raw_row),
        "radar": {
            "buckets": [payload["bucket"] for payload in radar_payload],
            "values": radar_payload,
            "scale": "Scores scaled vs best %s (%s season)" % (
                position.lower(),
                resolved_season or "all",
            ),
        },
        "bucket_tables": bucket_tables,
        "bucket_definitions": bucket_config,
        "bucket_features": bucket_feature_list,
        "stat_sections": stat_sections,
    }

