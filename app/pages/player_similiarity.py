import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_app import (
    load_store,
    load_pca_table,
    detect_minutes_column,
    apply_filters,
    load_cluster_table,
    load_raw_player_vectors,
)
from visuals import player_radar_chart


def pick_radar_features(store_df, feature_cols, player_idx, comparison_series, limit=8):
    # Focus the radar on features where the player deviates most once scaled by positional variance
    player_series = store_df.loc[player_idx, feature_cols]
    comparison_series = comparison_series.reindex(feature_cols)

    player_numeric = pd.to_numeric(player_series, errors='coerce')
    comparison_numeric = pd.to_numeric(comparison_series, errors='coerce')

    diff = player_numeric - comparison_numeric
    position_label = store_df.loc[player_idx, 'position'] if 'position' in store_df.columns else None

    if position_label is not None and not pd.isna(position_label):
        position_frame = store_df[store_df['position'] == position_label][feature_cols]
    else:
        position_frame = store_df[feature_cols]

    position_std = position_frame.std(skipna=True)
    global_std = store_df[feature_cols].std(skipna=True)
    position_std = position_std.replace(0, np.nan)
    if position_std.isna().all():
        position_std = global_std.replace(0, np.nan)
    position_std = position_std.fillna(global_std)
    position_std = position_std.replace(0, np.nan).fillna(1.0)

    effect = (diff.abs() / position_std).fillna(0)
    top_features = effect.nlargest(min(limit, len(effect))).index.tolist()
    if not top_features:
        top_features = feature_cols[:min(limit, len(feature_cols))]
    return top_features, player_numeric[top_features], comparison_numeric[top_features]

def scale_vectors_for_radar(store_df, feature_list, player_series, comparison_series, raw_player_series=None, raw_comparison_series=None):
    # Min-max to keep radar axes positive; fall back to ones if range collapses
    feature_frame = store_df[feature_list]
    mins = feature_frame.min()
    maxs = feature_frame.max()
    ranges = (maxs - mins).replace(0, 1.0)

    player_scaled = ((player_series - mins) / ranges).fillna(0).to_numpy()
    comparison_scaled = ((comparison_series - mins) / ranges).fillna(0).to_numpy()
    labels = feature_list
    
    if raw_player_series is None:
        player_raw = pd.to_numeric(player_series, errors='coerce').fillna(0).to_numpy()
    else:
        player_raw = pd.to_numeric(raw_player_series.reindex(feature_list), errors='coerce').fillna(0).to_numpy()
    
    if raw_comparison_series is None:
        comparison_raw = pd.to_numeric(comparison_series, errors='coerce').fillna(0).to_numpy()
    else:
        comparison_raw = pd.to_numeric(raw_comparison_series.reindex(feature_list), errors='coerce').fillna(0).to_numpy()
    
    return player_scaled, comparison_scaled, labels, player_raw, comparison_raw




def main():
    store = load_store()
    store_df = store.df.copy()
    raw_store_df = load_raw_player_vectors()
    # Align raw table with the similarity store to keep indices consistent
    if not raw_store_df.index.equals(store_df.index):
        raw_store_df = raw_store_df.reindex(store_df.index)
    projection_df, pca_components, axis_metadata = load_pca_table()
    available_positions = sorted(projection_df['position'].dropna().unique().tolist())

    cluster_df, centroids_dict, _ = load_cluster_table()

    # ==== Similarity Finder ====
    st.title('Player Similarity Finder')
    st.caption("Using cosine similarities to try and find similar players and displaying a radar chart of the most important attribute for the player's cluster")
    
    # Separate filters for similarity finder
    similarity_position_choice = st.selectbox('Position', available_positions, key='similarity_position')
    
    similarity_df = store_df[store_df['position'] == similarity_position_choice]
    similarity_team_options = sorted(set(similarity_df['team'].dropna().tolist()))
    similarity_team_selection = st.multiselect('Team', similarity_team_options, default=[], key='similarity_teams')
    
    similarity_minutes_col = detect_minutes_column(similarity_df)
    similarity_min_minutes = None
    if similarity_minutes_col and not similarity_df[similarity_minutes_col].isna().all():
        minutes_min = int(np.floor(similarity_df[similarity_minutes_col].min()))
        minutes_max = int(np.ceil(similarity_df[similarity_minutes_col].max()))
        default_min = max(100, minutes_min)
        default_min = min(default_min, minutes_max)
        similarity_min_minutes = st.slider('Minimum minutes', minutes_min, minutes_max, default_min, step=10, key='similarity_minutes')
    
    filtered_similarity_df = apply_filters(similarity_df, 'All', [similarity_position_choice], similarity_team_selection, similarity_minutes_col, similarity_min_minutes)
    
    if filtered_similarity_df.empty:
        st.info('Adjust the filters to pick at least one player.')
    else:
        # Build a unique list keyed by player/team/position/season so dropdowns stay unambiguous
        options = []
        seen = set()
        for idx, row in filtered_similarity_df.iterrows():
            key = (row['player'], row['team'], row['position'], row['season'])
            if key in seen:
                continue
            seen.add(key)
            label = '%s — %s (%s) [%s]' % (row['player'], row['team'], row['position'], row['season'])
            options.append({'label': label, 'player': row['player'], 'team': row['team'], 'position': row['position'], 'season': row['season'], 'index': idx})

        option_labels = [opt['label'] for opt in options]
        selected_label = st.selectbox('Select player', option_labels)
        selected = next(opt for opt in options if opt['label'] == selected_label)

        similar = store.find_similar_players(
            selected['player'],
            team=selected['team'],
            position=selected['position'],
            top_n=5,
            filter_position=selected['position'],
        )

        if not similar.empty:
            display_cols = ['player', 'team', 'position', 'similarity']
            table = similar[display_cols].copy()
            table['similarity'] = table['similarity'].round(3)
            st.dataframe(table, width='stretch', hide_index=True)
        else:
            st.info('No comparable players found for this selection.')

        # ---- Radar Chart ----
        st.markdown('**Profile comparison**')
        player_idx = selected['index']

        comparison_series = None
        comparison_label = None
        comparison_raw_series = None
        player_raw_series = raw_store_df.loc[player_idx, store.feature_cols] if player_idx in raw_store_df.index else None

        if not cluster_df.empty and player_idx in set(cluster_df['index']):
            cluster_row = cluster_df[cluster_df['index'] == player_idx].iloc[0]
            centroid_key = '%s_centroids' % cluster_row['position']
            centroids = centroids_dict.get(centroid_key)
            if centroids is not None and cluster_row['cluster'] < centroids.shape[0]:
                comparison_label = '%s centroid C%d' % (cluster_row['position'], cluster_row['cluster'] + 1)
                comparison_series = pd.Series(centroids[int(cluster_row['cluster'])], index=store.feature_cols)
                cluster_members = cluster_df[cluster_df['cluster'] == cluster_row['cluster']]['index'].tolist()
                if cluster_members:
                    comparison_raw_series = raw_store_df.loc[cluster_members, store.feature_cols].mean()

        if comparison_series is None:
            if not similar.empty:
                comp_row = similar.iloc[0]
                candidate_indices = store.get_player_index(comp_row['player'], team=comp_row['team'])
                if candidate_indices:
                    comparison_label = '%s (%s)' % (comp_row['player'], comp_row['team'])
                    comparison_series = store_df.loc[candidate_indices[0], store.feature_cols]
                    comparison_raw_series = raw_store_df.loc[candidate_indices[0], store.feature_cols] if candidate_indices[0] in raw_store_df.index else None
        if comparison_series is None:
            comparison_label = 'League average'
            comparison_series = store_df[store.feature_cols].mean()
            comparison_raw_series = raw_store_df[store.feature_cols].mean()

        feature_list, player_focus, comparison_focus = pick_radar_features(store_df, store.feature_cols, player_idx, comparison_series)
        player_scaled, comparison_scaled, labels, player_raw_values, comparison_raw_values = scale_vectors_for_radar(
            store_df,
            feature_list,
            player_focus,
            comparison_focus,
            raw_player_series=player_raw_series,
            raw_comparison_series=comparison_raw_series,
        )

        radar_fig = player_radar_chart(
            player_scaled,
            comparison_scaled,
            labels,
            player_label=selected['player'],
            comparison_label=comparison_label,
            title='Top feature comparison',
            player_hover_values=player_raw_values,
            comparison_hover_values=comparison_raw_values,
            hover_precision=2,
        )
        st.plotly_chart(radar_fig, width='stretch')

    st.write('---')

    st.caption('Tip: Use the filters above each section to explore clusters and similar profiles.')

if __name__ == '__main__':
    main()