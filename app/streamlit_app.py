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
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from similarity import PlayerVectorStore
from clustering import load_pca_results, load_clustering_results
from visuals import league_scatter, player_radar_chart


st.set_page_config(page_title='Premier League Data', layout='wide', page_icon='⚽')


@st.cache_resource
def load_store():
    return PlayerVectorStore()


@st.cache_data
def load_raw_player_vectors():
    data_dir = ROOT_DIR / 'data' / 'processed'
    path = data_dir / 'player_vectors_unscaled.parquet'
    df = pd.read_parquet(path)
    return df


@st.cache_data
def load_pca_table():
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
    for col in df.columns:
        if 'Min' in col and '90' not in col:
            return col
    return None


def apply_filters(df, season_value, positions, teams, minutes_col, min_minutes):
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


def create_dna_map_preview():
    try:
        store = load_store()
        store_df = store.df.copy()
        projection_df, pca_components, axis_metadata = load_pca_table()
        cluster_df, _, _ = load_cluster_table()
        
        if 'Keeper' not in projection_df['position'].values:
            return None
        
        keeper_projection = projection_df[projection_df['position'] == 'Keeper']
        keeper_df = store_df.join(keeper_projection[['pc1', 'pc2']], how='inner')
        keeper_df = keeper_df[keeper_df['position'] == 'Keeper']
        
        if keeper_df.empty:
            return None
        
        fig = league_scatter(
            keeper_df.head(50),
            color_by='team',
            hover_stats=['team'],
            title='',
            x_label='PC 1',
            y_label='PC 2',
            full_data=keeper_df
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        return fig
    except Exception:
        return None


def create_similarity_preview():
    try:
        store = load_store()
        store_df = store.df.copy()
        
        if store_df.empty or len(store.feature_cols) < 3:
            return None
        
        sample_player = store_df.iloc[0]
        player_idx = store_df.index[0]
        
        similar = store.find_similar_players(
            sample_player['player'],
            team=sample_player.get('team'),
            position=sample_player.get('position'),
            top_n=1,
            filter_position=sample_player.get('position'),
        )
        
        comparison_series = None
        comparison_label = 'League Average'
        
        if not similar.empty:
            comp_row = similar.iloc[0]
            candidate_indices = store.get_player_index(comp_row['player'], team=comp_row['team'])
            if candidate_indices:
                comparison_series = store_df.loc[candidate_indices[0], store.feature_cols]
                comparison_label = comp_row['player']
        
        if comparison_series is None:
            comparison_series = store_df[store.feature_cols].mean()
        
        player_series = store_df.loc[player_idx, store.feature_cols]
        
        feature_list = store.feature_cols[:6]
        player_values = player_series[feature_list].fillna(0).to_numpy()
        comparison_values = comparison_series[feature_list].fillna(0).to_numpy()
        
        mins = store_df[feature_list].min()
        maxs = store_df[feature_list].max()
        ranges = (maxs - mins).replace(0, 1.0)
        
        player_scaled = ((player_values - mins) / ranges).fillna(0).to_numpy()
        comparison_scaled = ((comparison_values - mins) / ranges).fillna(0).to_numpy()
        
        labels = [col.replace('_per90', '').replace('_', ' ')[:15] for col in feature_list]
        
        fig = player_radar_chart(
            player_scaled,
            comparison_scaled,
            labels,
            player_label=sample_player['player'][:15],
            comparison_label=comparison_label[:15],
            title=''
        )
        fig.update_layout(
            height=300,
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5, font=dict(size=10))
        )
        return fig
    except Exception:
        return None


def create_positional_preview():
    try:
        ROOT_DIR = Path(__file__).resolve().parent.parent
        SRC_DIR = ROOT_DIR / 'src'
        if str(SRC_DIR) not in sys.path:
            sys.path.append(str(SRC_DIR))
        
        from abilities import build_ability_dataframe
        
        store = load_store()
        df, _ = build_ability_dataframe(store)
        
        if df.empty:
            return None
        
        mid_df = df[df['position'] == 'Midfielder'].copy()
        if mid_df.empty or 'defensive_z' not in mid_df.columns or 'attacking_z' not in mid_df.columns:
            return None
        
        sample_df = mid_df.head(50).copy()
        
        # Build customdata with player name and team for hover template
        hover_custom_cols = ['player']
        if 'team' in sample_df.columns:
            hover_custom_cols.append('team')
        sample_df_aligned = sample_df.reset_index(drop=True)
        customdata_full = sample_df_aligned[hover_custom_cols].to_numpy()
        
        import plotly.express as px
        fig = px.scatter(
            sample_df_aligned,
            x='defensive_z',
            y='attacking_z',
            color='team',
            title='',
            labels={'defensive_z': 'Defensive Ability', 'attacking_z': 'Attacking Ability'}
        )
        
        # Build custom hover template: Name (bold), x-axis score, y-axis score, team
        hover_template = f'<b>%{{customdata[0]}}</b><br>Defensive Ability score: %{{x:.3f}}<br>Attacking Ability score: %{{y:.3f}}'
        if len(hover_custom_cols) > 1 and hover_custom_cols[1] == 'team':
            hover_template += '<br>%{customdata[1]}'
        hover_template += '<extra></extra>'
        
        # Handle multiple traces when coloring by team
        if 'team' in sample_df_aligned.columns:
            color_groups = sample_df_aligned.groupby('team', sort=False)
            color_to_customdata = {}
            for color_val, group in color_groups:
                color_mask = sample_df_aligned['team'] == color_val
                color_to_customdata[color_val] = customdata_full[color_mask]
            
            color_to_customdata_str = {str(k): v for k, v in color_to_customdata.items()}
            
            for trace in fig.data:
                trace_name = trace.name if hasattr(trace, 'name') else None
                if trace_name is not None:
                    trace_name_str = str(trace_name)
                    if trace_name_str in color_to_customdata_str:
                        trace_customdata = color_to_customdata_str[trace_name_str]
                    elif trace_name in color_to_customdata:
                        trace_customdata = color_to_customdata[trace_name]
                    else:
                        trace_customdata = customdata_full[:len(trace.x)] if len(trace.x) > 0 else customdata_full
                else:
                    trace_customdata = customdata_full[:len(trace.x)] if len(trace.x) > 0 else customdata_full
                
                trace.update(
                    customdata=trace_customdata,
                    hovertemplate=hover_template,
                )
        else:
            # Single trace
            fig.update_traces(
                customdata=customdata_full,
                hovertemplate=hover_template,
            )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        return fig
    except Exception:
        return None


def create_panel(title, description, preview_fig, page_path):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(description)

        if preview_fig:
            st.plotly_chart(preview_fig, width='stretch', config={'displayModeBar': False})
        else:
            st.info("Preview unavailable")

        st.markdown("")  # spacing before button

        if st.button(f"Explore {title}", key=f"btn_{title}"):
            st.switch_page(page_path)
            
def main():
    st.markdown("""
        <style>
        /* Scoped to the bordered containers Streamlit creates */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #EDEEEF; /* slightly darker than #F2F3F4 */
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: all 0.2s ease-in-out;
            padding: 1.25rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            transform: translateY(-2px);
            background-color: #E8E9EA;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('Football Data Analysis Project')
    st.markdown('**A linear algebra based analysis using techniques learned from my university course and applied to the premier league to explore player characteristics**')
    st.caption('Data from FBref | All Features are standardised')
    
    st.write('---')
    
    st.markdown('### Explore Features')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dna_preview = create_dna_map_preview()
        create_panel(
            'DNA Map',
            'Visualize player styles using PCA dimensionality reduction. Explore how players cluster by position and team.',
            dna_preview,
            'pages/dna_map.py'
        )
    
    with col2:
        similarity_preview = create_similarity_preview()
        create_panel(
            'Player Similarity',
            'Find players with similar statistical profiles using cosine similarity. Compare player profiles with radar charts.',
            similarity_preview,
            'pages/player_similiarity.py'
        )
    
    with col3:
        positional_preview = create_positional_preview()
        create_panel(
            'Positional Abilities',
            'Analyze player abilities across different dimensions. Compare players by attacking, defensive, and creative metrics.',
            positional_preview,
            'pages/position_abilities.py'
        )
    
    st.write('---')
    
    st.markdown("""
    ### About
    
    This project visualizes and compares player styles using advanced stats from FBref. 
    Each player is represented as a vector of numerical features derived from various stat categories.
    
    Using linear algebra and dimensionality reduction, we can map out stylistic similarities and explore:
    - Which players have similar statistical profiles
    - How player types cluster across the league
    - What attributes define those clusters
    """)


if __name__ == '__main__':
    main()
