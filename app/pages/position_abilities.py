import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress FutureWarning from Plotly Express internal pandas groupby operations
warnings.filterwarnings('ignore', category=FutureWarning, 
                        message='.*length-1.*get_group.*')

# Make sure we can import our project modules
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from similarity import PlayerVectorStore
from abilities import build_ability_dataframe
from team_colors import TEAM_COLORS


st.set_page_config(page_title='Positional Abilities', layout='wide', page_icon='⚽')


@st.cache_resource
def load_store():
    return PlayerVectorStore()


@st.cache_data
def load_ability_scores():
    store = load_store()
    df, feature_map = build_ability_dataframe(store)
    return df, feature_map


def detect_minutes_column(df):
    # Prefer exact match for "Playing Time_Min"
    if 'Playing Time_Min' in df.columns:
        return 'Playing Time_Min'
    
    # Fallback: find any column with 'Min' but not '90'
    for col in df.columns:
        if 'Min' in col and '90' not in col:
            return col
    return None


def create_ability_scatter(df, x_col, y_col, x_label, y_label, color_by='team', hover_cols=None, full_df=None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    if hover_cols is None:
        hover_cols = ['player', 'team']
    
    available_cols = [x_col, y_col]
    if color_by in df.columns:
        available_cols.append(color_by)
    available_cols.extend([c for c in hover_cols if c in df.columns and c not in available_cols])
    
    df_clean = df[available_cols].dropna(subset=[x_col, y_col])
    
    if df_clean.empty:
        return None
    
    # Sort by color_by column to ensure alphabetical legend order
    if color_by in df_clean.columns:
        df_clean = df_clean.sort_values(by=color_by)
    
    # Build customdata with player name and team for hover template
    hover_custom_cols = ['player']
    if 'team' in df_clean.columns:
        hover_custom_cols.append('team')
    df_aligned = df_clean.reset_index(drop=True)
    customdata_full = df_aligned[hover_custom_cols].to_numpy()
    
    # Prepare color mapping if coloring by team
    color_discrete_map = None
    category_orders = None
    if color_by == 'team' and color_by in df_clean.columns:
        unique_teams = sorted(df_clean[color_by].unique())
        color_discrete_map = {team: TEAM_COLORS.get(team, '#808080') for team in unique_teams}
        category_orders = {color_by: unique_teams}
    elif color_by in df_clean.columns:
        # For other color_by columns, ensure alphabetical order
        unique_vals = sorted(df_clean[color_by].unique())
        category_orders = {color_by: unique_vals}
    
    # Suppress FutureWarning from Plotly Express internal pandas groupby operations
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, 
                                message='.*length-1.*get_group.*')
        fig = px.scatter(
            df_aligned,
            x=x_col,
            y=y_col,
            color=color_by if color_by in df_clean.columns else None,
            color_discrete_map=color_discrete_map,
            category_orders=category_orders,
            title=f'{x_label} vs {y_label}',
            labels={x_col: x_label, y_col: y_label},
        )
    
    if full_df is not None and x_col in full_df.columns and y_col in full_df.columns:
        full_clean = full_df[[x_col, y_col]].dropna(subset=[x_col, y_col])
        if not full_clean.empty:
            x_min, x_max = full_clean[x_col].min(), full_clean[x_col].max()
            y_min, y_max = full_clean[y_col].min(), full_clean[y_col].max()
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05
            fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
            fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding])
    
    # Build custom hover template: Name (bold), x-axis score, y-axis score, team
    hover_template = f'<b>%{{customdata[0]}}</b><br>{x_label} score: %{{x:.3f}}<br>{y_label} score: %{{y:.3f}}'
    if len(hover_custom_cols) > 1 and hover_custom_cols[1] == 'team':
        hover_template += '<br>%{customdata[1]}'
    hover_template += '<extra></extra>'
    
    # Update traces with custom hover template and customdata
    if color_by and color_by in df_aligned.columns:
        # Group dataframe by color value to get row ranges for each trace
        color_groups = df_aligned.groupby(color_by, sort=False)
        
        # Create a mapping from color value to customdata array
        color_to_customdata = {}
        for color_val, group in color_groups:
            color_mask = df_aligned[color_by] == color_val
            color_to_customdata[color_val] = customdata_full[color_mask]
        
        # Match each trace to its color group by trace name
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
                    # Fallback: try to match by first point's coordinates
                    if len(trace.x) > 0:
                        first_x, first_y = trace.x[0], trace.y[0]
                        matching_rows = df_aligned[
                            (np.abs(df_aligned[x_col] - first_x) < 1e-6) & 
                            (np.abs(df_aligned[y_col] - first_y) < 1e-6)
                        ]
                        if len(matching_rows) > 0:
                            matched_color = matching_rows[color_by].iloc[0]
                            if matched_color in color_to_customdata:
                                trace_customdata = color_to_customdata[matched_color]
                            else:
                                trace_customdata = customdata_full[:len(trace.x)]
                        else:
                            trace_customdata = customdata_full[:len(trace.x)]
                    else:
                        trace_customdata = customdata_full
            else:
                trace_customdata = customdata_full[:len(trace.x)] if len(trace.x) > 0 else customdata_full
            
            trace.update(
                customdata=trace_customdata,
                marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color='rgba(255,255,255,0.6)')),
                hovertemplate=hover_template,
            )
    else:
        # Single trace, use full customdata
        fig.update_traces(
            customdata=customdata_full,
            marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color='rgba(255,255,255,0.6)')),
            hovertemplate=hover_template,
        )
    
    # Ensure legend is sorted alphabetically
    if color_by in df_clean.columns:
        fig.update_layout(
            legend=dict(
                itemsizing='constant',
                traceorder='normal'
            )
        )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
    )
    
    return fig


def create_top_performers_bar(df, ability_col, top_n=10, title=None):
    if df.empty or ability_col not in df.columns:
        return None
    
    df_clean = df[[ability_col, 'player', 'team']].dropna(subset=[ability_col])
    
    if df_clean.empty:
        return None
    
    top = df_clean.nlargest(top_n, ability_col)
    top = top.sort_values(ability_col, ascending=True)
    
    if title is None:
        title = f'Top {top_n} Performers'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[f"{row['player']} ({row['team']})" for _, row in top.iterrows()],
        x=top[ability_col],
        orientation='h',
        marker=dict(color=top[ability_col], colorscale='Viridis', showscale=True),
        text=[f"{val:.2f}" for val in top[ability_col]],
        textposition='outside',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Z-Score',
        yaxis_title='',
        margin=dict(l=200, r=40, t=60, b=40),
        height=max(400, top_n * 40),
    )
    
    return fig


def render_position_tab(df, position, ability1_name, ability1_col, ability2_name, ability2_col, minutes_col=None, full_df=None, tab_key=''):
    pos_df = df[df['position'] == position].copy()
    
    if pos_df.empty:
        st.info(f'No {position} players found matching the filters.')
        return
    
    st.subheader(f'{position}s')
    
    # Filters above the graph
    team_options = sorted(set(pos_df['team'].dropna().tolist()))
    team_selection = st.multiselect('Team', team_options, default=[], key=f'teams_{position}_{tab_key}')
    
    min_minutes = None
    minutes_min = None
    default_minutes = None
    if minutes_col and minutes_col in pos_df.columns and not pos_df[minutes_col].isna().all():
        minutes_min = int(np.floor(pos_df[minutes_col].min()))
        minutes_max = int(np.ceil(pos_df[minutes_col].max()))
        default_minutes = max(100, minutes_min)
        default_minutes = min(default_minutes, minutes_max)
        min_minutes = st.slider('Minimum minutes', minutes_min, minutes_max, default_minutes, step=50, key=f'minutes_{position}_{tab_key}')
    
    # Apply filters separately: minutes-only for axis ranges, both for display
    minutes_only_pos_df = pos_df.copy()
    if minutes_col and min_minutes is not None:
        minutes_only_pos_df = minutes_only_pos_df[minutes_only_pos_df[minutes_col] >= min_minutes]
    
    filtered_pos_df = pos_df.copy()
    if team_selection:
        filtered_pos_df = filtered_pos_df[filtered_pos_df['team'].isin(team_selection)]
    if minutes_col and min_minutes is not None:
        filtered_pos_df = filtered_pos_df[filtered_pos_df[minutes_col] >= min_minutes]
    
    # Metrics row above the graph
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(f'Total {position}s', len(filtered_pos_df))
    
    with metric_col2:
        st.metric(f'Teams', filtered_pos_df['team'].nunique())
    
    with metric_col3:
        if ability1_col in filtered_pos_df.columns:
            top1 = filtered_pos_df.nlargest(1, ability1_col)
            if not top1.empty:
                st.metric(
                    f'Top {ability1_name}',
                    f"{top1.iloc[0]['player']}",
                    delta=f"{top1.iloc[0][ability1_col]:.2f}"
                )
    
    with metric_col4:
        if ability2_col in filtered_pos_df.columns:
            top2 = filtered_pos_df.nlargest(1, ability2_col)
            if not top2.empty:
                st.metric(
                    f'Top {ability2_name}',
                    f"{top2.iloc[0]['player']}",
                    delta=f"{top2.iloc[0][ability2_col]:.2f}"
                )
    
    # Use minutes-filtered data for axis ranges when slider is moved from default, otherwise use full data
    # Team filter doesn't affect axis ranges
    full_pos_df = None
    if full_df is not None:
        full_pos_df = full_df[full_df['position'] == position].copy()
    
    minutes_filtering = min_minutes is not None and default_minutes is not None and min_minutes != default_minutes
    axis_data = minutes_only_pos_df if minutes_filtering else full_pos_df
    
    # Full-width scatter plot
    st.plotly_chart(
        create_ability_scatter(
            filtered_pos_df,
            ability1_col,
            ability2_col,
            ability1_name,
            ability2_name,
            color_by='team',
            hover_cols=['player', 'team'],
            full_df=axis_data
        ),
        width='stretch'
    )
    
    st.write('---')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.plotly_chart(
            create_top_performers_bar(
                filtered_pos_df,
                ability1_col,
                top_n=10,
                title=f'Top {ability1_name}'
            ),
            width='stretch'
        )
    
    with col4:
        st.plotly_chart(
            create_top_performers_bar(
                filtered_pos_df,
                ability2_col,
                top_n=10,
                title=f'Top {ability2_name}'
            ),
            width='stretch'
        )


def main():
    st.title('Positional Abilities Dashboard')
    st.caption('Compare players within each position using weighted ability scores on selected features')
    
    df, feature_map = load_ability_scores()
    
    if df.empty:
        st.error('No ability scores available. Please run the preprocessing pipeline first.')
        return
    
    minutes_col = detect_minutes_column(df)
    
    tab1, tab2, tab3, tab4 = st.tabs(['Goalkeepers', 'Defenders', 'Midfielders', 'Attackers'])
    
    with tab1:
        render_position_tab(
            df,
            'Keeper',
            'Goalkeeping Ability',
            'goalkeeping_z',
            'Distribution Ability',
            'distribution_z',
            minutes_col,
            full_df=df,
            tab_key='keeper'
        )
    
    with tab2:
        render_position_tab(
            df,
            'Defender',
            'Defensive Ability',
            'defensive_z',
            'Progressive Ability',
            'progressive_z',
            minutes_col,
            full_df=df,
            tab_key='defender'
        )
    
    with tab3:
        render_position_tab(
            df,
            'Midfielder',
            'Defensive Ability',
            'defensive_z',
            'Attacking Ability',
            'attacking_z',
            minutes_col,
            full_df=df,
            tab_key='midfielder'
        )
    
    with tab4:
        render_position_tab(
            df,
            'Attacker',
            'Shooting Ability',
            'shooting_z',
            'Technical Ability',
            'technical_z',
            minutes_col,
            full_df=df,
            tab_key='attacker'
        )
    
    st.write('---')
    
    with st.expander('About Ability Scores'):
        st.markdown("""
        **How scores are calculated:**
        
        1. **Feature Selection**: Each position has two ability buckets, each containing 5-6 relevant statistics
        2. **Weighting**: Features are weighted based on domain knowledge (e.g., Save% weighted higher than GA90 for goalkeepers)
        3. **Normalization**: Negative-impact metrics (like Goals Against, Fouls) are flipped so higher is always better
        4. **Aggregation**: Weighted linear combination of standardized features
        5. **Z-Scoring**: Scores are normalized within each position to ensure comparability
        
        **Interpreting Z-Scores:**
        - **Positive values**: Above average for the position
        - **Zero**: Average for the position
        - **Negative values**: Below average for the position
        - **Magnitude**: How many standard deviations away from the mean
        
        **Ability Definitions:**
        - **Goalkeepers**: Goalkeeping (shot-stopping, clean sheets) vs Distribution (passing, sweeper actions)
        - **Defenders**: Defensive (tackles, blocks, interceptions) vs Progressive (progressive passes/carries)
        - **Midfielders**: Defensive (tackles, recoveries) vs Attacking (xG+xAG, assists, key passes)
        - **Attackers**: Shooting (goals, xG, shots on target) vs Technical (dribbling, passing, receiving)
        """)


if __name__ == '__main__':
    main()

