import sys
import warnings
import os
from pathlib import Path

import streamlit as st

# Suppress noisy warnings for a clean UI
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.bool8.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*BlockManager.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*is_sparse.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*length-1.*get_group.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*bottleneck.*")

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from player_profile import get_player_profile
from streamlit_app import load_store
from visuals import radial_score_chart, plotly_config


def _valid_query_value(value):
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _load_players_dataframe():
    store = load_store()
    return store.df.copy()


def _first_query_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _resolve_query_defaults():
    params = st.query_params
    return {
        "player": _first_query_value(params.get("player")),
        "team": _first_query_value(params.get("team")),
        "season": _first_query_value(params.get("season")),
    }


def _assign_query_param(key, value):
    params = st.query_params
    if _valid_query_value(value):
        params[key] = value
    else:
        params.pop(key, None)


def _update_query_params(player_name, team, season):
    _assign_query_param("player", player_name)
    _assign_query_param("team", team)
    _assign_query_param("season", season)


def _build_player_options(df):
    options = []
    for idx, row in df.iterrows():
        label = "%s — %s (%s)" % (
            row.get("player", "Unknown"),
            row.get("team", "Unknown"),
            row.get("season", "Unknown"),
        )
        options.append(
            {
                "label": label,
                "player": row.get("player"),
                "team": row.get("team"),
                "season": row.get("season"),
                "index": idx,
            }
        )
    return options


def _render_hero_section(profile):
    metadata = profile["metadata"]
    hero_metrics = profile["hero_metrics"]
    radar_values = profile["radar"]["values"]
    scale_note = profile["radar"].get("scale")
    bucket_feature_map = profile.get("bucket_features", {})
    if not radar_values:
        st.warning("Ability scores unavailable for this player.")
        return

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"## {metadata.get('player', 'Unknown')}",
        )
        subtitle = []
        if metadata.get("team"):
            subtitle.append(metadata["team"])
        if metadata.get("nation"):
            subtitle.append(metadata["nation"])
        if metadata.get("position"):
            subtitle.append(metadata["position"])
        if metadata.get("season"):
            subtitle.append(str(metadata["season"]))
        if subtitle:
            st.caption(" • ".join(subtitle))

        if hero_metrics:
            metric_cols = st.columns(len(hero_metrics))
            for metric, col in zip(hero_metrics, metric_cols):
                display_value = metric["value"]
                if isinstance(display_value, (int, float)) and abs(display_value) >= 1000:
                    display_value = round(display_value, 1)
                col.metric(metric["label"], display_value)

    with right:
        labels = [entry["bucket"] for entry in radar_values]
        player_values = [entry["score"] for entry in radar_values]
        radar_fig = radial_score_chart(
            player_values,
            labels,
            max_value=100.0,
            title="Ability score (0-100)",
        )
        st.plotly_chart(radar_fig, config=plotly_config())
        if scale_note:
            st.caption(scale_note)

        if bucket_feature_map:
            with st.expander("Ability Inputs"):
                for bucket in labels:
                    features = bucket_feature_map.get(bucket, [])
                    if not features:
                        continue
                    st.caption(f"{bucket}: {', '.join(features)}")


def _render_stat_tabs(profile):
    sections = profile.get("stat_sections", [])
    if not sections:
        st.info("Detailed stat tables unavailable for this player.")
        return

    # Filter out irrelevant sections for goalkeepers
    metadata = profile.get("metadata", {})
    position = metadata.get("position", "").lower()
    if position == "keeper":
        excluded_keys = {"standard", "shooting", "goal_shot_creation", "defense", "misc"}
        sections = [s for s in sections if s.get("key") not in excluded_keys]
        sections.reverse()

    if not sections:
        st.info("No relevant stat tables available for this player.")
        return

    tabs = st.tabs([section["label"] for section in sections])
    for tab, section in zip(tabs, sections):
        with tab:
            data = section.get("data")
            if data is None or data.empty:
                st.info("No stats available in this category.")
                continue
            st.dataframe(
                data,
                hide_index=True,
                width='stretch',
            )


def main():
    st.title("Player Profile Explorer")
    st.caption("Inspect individual player DNA with per-position radar categories.")

    # Check if we navigated from another page with pending player selection
    if '_pending_player' in st.session_state:
        _update_query_params(
            st.session_state.get('_pending_player'),
            st.session_state.get('_pending_team'),
            st.session_state.get('_pending_season')
        )
        # Clear pending state
        st.session_state.pop('_pending_player', None)
        st.session_state.pop('_pending_team', None)
        st.session_state.pop('_pending_season', None)

    df = _load_players_dataframe()
    if df.empty:
        st.warning("Player vectors unavailable. Please refresh after data generation.")
        return

    defaults = _resolve_query_defaults()

    def on_filter_change():
        # Clear player query params when filters change
        params = st.query_params
        params.pop("player", None)
        params.pop("team", None)

    season_options = ["All"]
    if "season" in df.columns:
        season_options += sorted(df["season"].dropna().unique().tolist())
    season_default = defaults["season"] if defaults["season"] in season_options else "All"
    season_choice = st.selectbox(
        "Season",
        season_options,
        index=season_options.index(season_default),
        key="season_filter",
        on_change=on_filter_change,
    )

    position_options = ["All"]
    if "position" in df.columns:
        position_options += sorted(df["position"].dropna().unique().tolist())
    position_default = "All"
    if defaults["player"]:
        player_rows = df[df["player"] == defaults["player"]]
        if not player_rows.empty:
            detected_position = player_rows.iloc[0].get("position")
            if detected_position in position_options:
                position_default = detected_position
    position_choice = st.selectbox(
        "Position",
        position_options,
        index=position_options.index(position_default),
        key="position_filter",
        on_change=on_filter_change,
    )

    filtered_df = df.copy()
    if season_choice != "All" and "season" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["season"] == season_choice]
    if position_choice != "All" and "position" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["position"] == position_choice]

    if filtered_df.empty:
        st.info("No players match the current filters.")
        return

    player_options = _build_player_options(filtered_df)
    option_labels = [opt["label"] for opt in player_options]

    default_index = 0
    if defaults["player"]:
        for idx, opt in enumerate(player_options):
            if opt["player"] == defaults["player"]:
                if defaults["team"] and opt["team"] != defaults["team"]:
                    continue
                if defaults["season"] and opt["season"] != defaults["season"]:
                    continue
                default_index = idx
                break

    selected_label = st.selectbox("Player", option_labels, index=default_index)
    selection = next(opt for opt in player_options if opt["label"] == selected_label)
    _update_query_params(selection["player"], selection["team"], selection["season"])

    try:
        profile = get_player_profile(
            selection["player"],
            team=selection["team"],
            season=selection["season"],
        )
    except Exception as exc:
        st.error("Unable to load profile: %s" % exc)
        return

    _render_hero_section(profile)

    st.write("---")
    _render_stat_tabs(profile)


if __name__ == "__main__":
    main()

