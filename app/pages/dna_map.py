import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Suppress FutureWarning from Plotly Express internal pandas groupby operations
warnings.filterwarnings('ignore', category=FutureWarning, 
                        message='.*length-1.*get_group.*')


# Make sure we can import our project modules no matter where Streamlit launches from
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from similarity import PlayerVectorStore
from clustering import load_pca_results, load_clustering_results
from visuals import league_scatter


st.set_page_config(page_title='Player DNA Map', layout='wide')


@st.cache_resource
def load_store():
    # PlayerVectorStore loads the processed matrix and keeps the numpy views handy
    return PlayerVectorStore()


@st.cache_data
def load_pca_table():
    # Pull projection + metadata, flatten the npz payload for cacheability
    pack = load_pca_results()
    projection_df = pack['projection_df'].copy()
    components_raw = pack['components_data']
    components = {}
    try:
        for key in components_raw.files:
            value = components_raw[key]
            if key == 'feature_cols':
                continue
            if '__' in key:
                position, metric = key.split('__', 1)
                if position not in components:
                    components[position] = {}
                components[position][metric] = value
            else:
                components[key] = value
    finally:
        components_raw.close()
    axis_metadata = pack.get('axis_metadata', {})
    return projection_df, components, axis_metadata


@st.cache_data
def load_cluster_table():
    # Grab clustering assignments and centroids so we can map back clusters in the UI
    pack = load_clustering_results()
    cluster_df = pack['cluster_df'].copy()
    centroids_raw = pack['centroids_data']
    centroids = {}
    try:
        for key in centroids_raw.files:
            centroids[key] = centroids_raw[key]
    finally:
        centroids_raw.close()
    metadata = pack.get('metadata', {})
    return cluster_df, centroids, metadata


def detect_minutes_column(df):
    # Reuse the same heuristic as preprocessing: first column with "Min" but not "90"
    for col in df.columns:
        if 'Min' in col and '90' not in col:
            return col
    return None


def apply_filters(df, season_value, positions, teams, minutes_col, min_minutes):
    # Minimal filtering pipeline keeps operations vectorised
    filtered = df
    if season_value != 'All':
        filtered = filtered[filtered['season'] == season_value]
    if positions:
        if isinstance(positions, str):
            positions = [positions]
        filtered = filtered[filtered['position'].isin(positions)]
    if teams:
        filtered = filtered[filtered['team'].isin(teams)]
    if minutes_col and min_minutes is not None:
        filtered = filtered[filtered[minutes_col] >= min_minutes]
    return filtered


def build_cluster_columns(base_df, cluster_df):
    # Map integer cluster ids back onto the combined dataframe
    base_df = base_df.copy()
    if cluster_df.empty:
        base_df['cluster_id'] = np.nan
        base_df['cluster_label'] = None
        return base_df

    lookup = cluster_df.set_index('index')['cluster']
    cluster_series = base_df.index.to_series().map(lookup)
    base_df['cluster_id'] = cluster_series

    cluster_labels = np.full(len(base_df), None, dtype=object)
    mask = cluster_series.notna()
    if mask.any():
        cluster_labels[mask.values] = (
            base_df.loc[mask, 'position'].fillna('Unknown').astype(str)
            + ' C'
            + (cluster_series[mask].astype(int) + 1).astype(str)
        )
    base_df['cluster_label'] = cluster_labels
    return base_df

def render_league_map_tab(store, store_df, projection_df, pca_components, axis_metadata, cluster_df, position_choice):
    projection_slice = projection_df[projection_df['position'] == position_choice]
    combined_df = store_df.join(projection_slice[['pc1', 'pc2']], how='inner')
    combined_df = combined_df[combined_df['position'] == position_choice]
    combined_df = build_cluster_columns(combined_df, cluster_df)

    minutes_col = detect_minutes_column(combined_df)

    # Filters above the graph (only affect the graph)
    team_options = sorted(set(combined_df['team'].dropna().tolist()))
    team_selection = st.multiselect('Team', team_options, default=[], key=f'teams_{position_choice}')

    min_minutes = None
    minutes_min = None
    default_minutes = None
    if minutes_col and not combined_df[minutes_col].isna().all():
        minutes_min = int(np.floor(combined_df[minutes_col].min()))
        minutes_max = int(np.ceil(combined_df[minutes_col].max()))
        default_minutes = max(100, minutes_min)
        default_minutes = min(default_minutes, minutes_max)
        min_minutes = st.slider('Minimum minutes', minutes_min, minutes_max, default_minutes, step=10, key=f'minutes_{position_choice}')

        scatter_color_options = {
        'Team': 'team',
        'Cluster': 'cluster_label',
    }
    color_choice = st.selectbox('Colour players by', list(scatter_color_options.keys()), index=0, key=f'color_{position_choice}')
    
    # Apply filters separately: minutes-only for axis ranges, both for display
    minutes_only_df = apply_filters(combined_df, 'All', [position_choice], [], minutes_col, min_minutes)
    filtered_df = apply_filters(combined_df, 'All', [position_choice], team_selection, minutes_col, min_minutes)

    col1, col2 = st.columns(2)
    col1.metric('Players', len(filtered_df))
    col2.metric('Teams', filtered_df['team'].nunique())

    positional_components = pca_components.get(position_choice, {})
    explained = positional_components.get('explained_variance_ratio')
    if explained is not None and len(explained) >= 2:
        st.caption('PCA variance (%s): PC1 %.1f%% | PC2 %.1f%%' % (position_choice, explained[0] * 100, explained[1] * 100))

    hover_columns = ['team', 'position']
    if minutes_col:
        hover_columns.append(minutes_col)

    if filtered_df.empty:
        st.info('No players match the current filters.')
    else:
        axis_info = axis_metadata.get(position_choice, [])

        x_axis_label = 'PC 1'
        y_axis_label = 'PC 2'
        axis_notes = []
        if axis_info:
            if len(axis_info) >= 1:
                first_features = axis_info[0].get('features', [])
                x_axis_label = x_axis_label
                if first_features:
                    axis_notes.append('PC1 loads: %s' % ', '.join('%s (%.2f)' % (feat['name'], feat['weight']) for feat in first_features))
            if len(axis_info) >= 2:
                second_features = axis_info[1].get('features', [])
                y_axis_label = y_axis_label
                if second_features:
                    axis_notes.append('PC2 loads: %s' % ', '.join('%s (%.2f)' % (feat['name'], feat['weight']) for feat in second_features))

        # Use minutes-filtered data for axis ranges when slider is moved from default, otherwise use full data
        # Team filter doesn't affect axis ranges
        minutes_filtering = min_minutes is not None and default_minutes is not None and min_minutes != default_minutes
        axis_data = minutes_only_df if minutes_filtering else combined_df
        fig = league_scatter(
            filtered_df,
            color_by=scatter_color_options[color_choice],
            hover_stats=hover_columns,
            title='Player DNA Map',
            x_label=x_axis_label,
            y_label=y_axis_label,
            full_data=axis_data
        )
        st.plotly_chart(fig, width='stretch')
        if axis_notes:
            st.caption(' | '.join(axis_notes))


def main():
    store = load_store()
    store_df = store.df.copy()

    projection_df, pca_components, axis_metadata = load_pca_table()
    cluster_df, centroids_dict, _ = load_cluster_table()

    available_positions = sorted(projection_df['position'].dropna().unique().tolist())
    if not available_positions:
        st.error('No PCA projections available. Please recompute projections.')
        return

    st.title('Player DNA Map — English Premier League')
    st.caption('Using PCA to find 2 axes for each position and the k means to produce clusters')

    st.write('---')

    # ==== League Map with Tabs ====
    st.subheader('League Map')
    
    position_map = {
        'Goalkeepers': 'Keeper',
        'Defenders': 'Defender',
        'Midfielders': 'Midfielder',
        'Attackers': 'Attacker'
    }
    
    tab1, tab2, tab3, tab4 = st.tabs(['Goalkeepers', 'Defenders', 'Midfielders', 'Attackers'])
    
    with tab1:
        if 'Keeper' in available_positions:
            render_league_map_tab(store, store_df, projection_df, pca_components, axis_metadata, cluster_df, 'Keeper')
        else:
            st.info('No goalkeeper data available.')
    
    with tab2:
        if 'Defender' in available_positions:
            render_league_map_tab(store, store_df, projection_df, pca_components, axis_metadata, cluster_df, 'Defender')
        else:
            st.info('No defender data available.')
    
    with tab3:
        if 'Midfielder' in available_positions:
            render_league_map_tab(store, store_df, projection_df, pca_components, axis_metadata, cluster_df, 'Midfielder')
        else:
            st.info('No midfielder data available.')
    
    with tab4:
        if 'Attacker' in available_positions:
            render_league_map_tab(store, store_df, projection_df, pca_components, axis_metadata, cluster_df, 'Attacker')
        else:
            st.info('No attacker data available.')

    st.write('---')


if __name__ == '__main__':
    main()

