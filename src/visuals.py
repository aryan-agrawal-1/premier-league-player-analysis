import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from team_colors import TEAM_COLORS

PLOTLY_BASE_CONFIG = {
    # Keep charts responsive inside Streamlit containers and drop the Plotly watermark
    'responsive': True,
    'displaylogo': False,
}


def plotly_config(overrides=None):
    cfg = dict(PLOTLY_BASE_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


__all__ = [
    'league_scatter',
    'player_radar_chart',
    'plotly_config',
]


def league_scatter(data, color_by='position', hover_stats=None, title='Player DNA Map', x_label=None, y_label=None, full_data=None):
    # Build a PCA scatter so we can see how players cluster in the reduced space and optionally rename axes

    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')

    required = {'pc1', 'pc2', 'player'}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError('data is missing required columns: %s' % ', '.join(sorted(missing)))

    df = data.copy()
    # Hover template only needs player name and team
    hover_cols = ['player']
    # Always include team if it exists in the dataframe
    if 'team' in df.columns:
        hover_cols.append('team')

    if color_by:
        if color_by not in df.columns:
            raise ValueError('color_by column %s not found in data' % color_by)
        # Sort by color_by column to ensure alphabetical legend order
        df = df.sort_values(by=color_by)

    # Plotly handles projection axes + colouring; we just make sure inputs are clean
    df_aligned = df.reset_index(drop=True)
    customdata_full = df_aligned[hover_cols].to_numpy()
    
    # Prepare color mapping if coloring by team
    color_discrete_map = None
    category_orders = None
    if color_by == 'team' and color_by in df_aligned.columns:
        unique_teams = sorted(df_aligned[color_by].unique())
        color_discrete_map = {team: TEAM_COLORS.get(team, '#808080') for team in unique_teams}
        category_orders = {color_by: unique_teams}
    elif color_by and color_by in df_aligned.columns:
        # For other color_by columns, ensure alphabetical order
        unique_vals = sorted(df_aligned[color_by].unique())
        category_orders = {color_by: unique_vals}
    
    # Suppress FutureWarning from Plotly Express internal pandas groupby operations
    # This warning occurs when plotly uses get_group(name) instead of get_group((name,))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, 
                                message='.*length-1.*get_group.*')
        fig = px.scatter(
            df_aligned,
            x='pc1',
            y='pc2',
            color=color_by if color_by else None,
            color_discrete_map=color_discrete_map,
            category_orders=category_orders,
            title=title,
            labels={'pc1': x_label or 'PC 1', 'pc2': y_label or 'PC 2'},
        )
    
    # Set fixed axis ranges based on full dataset if provided
    if full_data is not None and isinstance(full_data, pd.DataFrame):
        if 'pc1' in full_data.columns and 'pc2' in full_data.columns:
            full_clean = full_data[['pc1', 'pc2']].dropna(subset=['pc1', 'pc2'])
            if not full_clean.empty:
                x_min, x_max = full_clean['pc1'].min(), full_clean['pc1'].max()
                y_min, y_max = full_clean['pc2'].min(), full_clean['pc2'].max()
                x_padding = (x_max - x_min) * 0.05
                y_padding = (y_max - y_min) * 0.05
                fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
                fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding])
    # Build hover template: Name (bold), PC1 score, PC2 score, team
    hover_template = '<b>%{customdata[0]}</b><br>PC1 score: %{x:.3f}<br>PC2 score: %{y:.3f}'
    # Add team if it exists (should be at index 1)
    if len(hover_cols) > 1 and hover_cols[1] == 'team':
        hover_template += '<br>%{customdata[1]}'
    hover_template += '<extra></extra>'

    # When Plotly groups by color, it preserves row order within each trace
    # Match traces to color groups by trace name
    if color_by:
        # Group dataframe by color value to get row ranges for each trace
        color_groups = df_aligned.groupby(color_by, sort=False)
        
        # Create a mapping from color value to customdata array
        color_to_customdata = {}
        for color_val, group in color_groups:
            color_mask = df_aligned[color_by] == color_val
            color_to_customdata[color_val] = customdata_full[color_mask]
        
        # Match each trace to its color group by trace name
        # Convert color values to strings for matching (Plotly may convert names to strings)
        color_to_customdata_str = {str(k): v for k, v in color_to_customdata.items()}
        
        for trace in fig.data:
            # Trace name should match the color value
            trace_name = trace.name if hasattr(trace, 'name') else None
            if trace_name is not None:
                trace_name_str = str(trace_name)
                if trace_name_str in color_to_customdata_str:
                    trace_customdata = color_to_customdata_str[trace_name_str]
                elif trace_name in color_to_customdata:
                    trace_customdata = color_to_customdata[trace_name]
                else:
                    # Fallback: try to match by first point's color value
                    if len(trace.x) > 0:
                        # Find rows with matching coordinates
                        first_x, first_y = trace.x[0], trace.y[0]
                        matching_rows = df_aligned[
                            (np.abs(df_aligned['pc1'] - first_x) < 1e-6) & 
                            (np.abs(df_aligned['pc2'] - first_y) < 1e-6)
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
                # No trace name, use fallback
                trace_customdata = customdata_full[:len(trace.x)] if len(trace.x) > 0 else customdata_full
            
            trace.update(
                customdata=trace_customdata,
                marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color='rgba(255,255,255,0.6)')),
                hovertemplate=hover_template,
            )
    else:
        # Single trace, use full customdata
        fig.update_traces(
            customdata=customdata_full,
            marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color='rgba(255,255,255,0.6)')),
            hovertemplate=hover_template,
        )
    # Ensure legend is sorted alphabetically
    if color_by:
        fig.update_layout(
            legend=dict(
                itemsizing='constant',
                traceorder='normal'
            )
        )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title=x_label or 'PC 1',
        yaxis_title=y_label or 'PC 2'
    )
    return fig


def player_radar_chart(
    player_values,
    comparison_values,
    feature_labels,
    player_label='Player',
    comparison_label='Comparison',
    title='Player Profile Comparison',
    player_hover_values=None,
    comparison_hover_values=None,
    hover_precision=2
):
    # Overlay player vs comparison vector so we can see strengths and trade-offs quickly
    player_vec = np.asarray(player_values, dtype=float)
    comp_vec = np.asarray(comparison_values, dtype=float)
    labels = list(feature_labels)
    if len(player_vec) != len(comp_vec) or len(player_vec) != len(labels):
        raise ValueError('feature lengths do not align')
    if len(labels) < 3:
        raise ValueError('radar charts need at least three features')
    # Close the polygon by repeating the first point at the end
    wrap_labels = labels + [labels[0]]
    player_points = np.append(player_vec, player_vec[0])
    comp_points = np.append(comp_vec, comp_vec[0])
    value_fmt = f'.{max(0, int(hover_precision))}f'
    player_custom = None
    comparison_custom = None
    player_hover_template = (
        '<b>%{theta}</b><br>'
        f'{player_label}: %{{r:{value_fmt}}}<extra></extra>'
    )
    comparison_hover_template = (
        '<b>%{theta}</b><br>'
        f'{comparison_label}: %{{r:{value_fmt}}}<extra></extra>'
    )

    if player_hover_values is not None and comparison_hover_values is not None:
        player_hover = np.asarray(player_hover_values, dtype=float)
        comparison_hover = np.asarray(comparison_hover_values, dtype=float)
        if len(player_hover) != len(labels) or len(comparison_hover) != len(labels):
            raise ValueError('hover value lengths do not align with feature labels')
        hover_pairs_player = np.column_stack([player_hover, comparison_hover])
        hover_pairs_comparison = np.column_stack([comparison_hover, player_hover])
        # Repeat the first row to close the polygon
        hover_pairs_player = np.vstack([hover_pairs_player, hover_pairs_player[0]])
        hover_pairs_comparison = np.vstack([hover_pairs_comparison, hover_pairs_comparison[0]])
        player_custom = hover_pairs_player
        comparison_custom = hover_pairs_comparison
        player_hover_template = (
            '<b>%{theta}</b><br>'
            f'{player_label}: %{{customdata[0]:{value_fmt}}}<br>'
            f'{comparison_label}: %{{customdata[1]:{value_fmt}}}<extra></extra>'
        )
        comparison_hover_template = (
            '<b>%{theta}</b><br>'
            f'{comparison_label}: %{{customdata[0]:{value_fmt}}}<br>'
            f'{player_label}: %{{customdata[1]:{value_fmt}}}<extra></extra>'
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=player_points,
            theta=wrap_labels,
            fill='toself',
            name=player_label,
            line=dict(width=2),
            customdata=player_custom,
            hovertemplate=player_hover_template,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=comp_points,
            theta=wrap_labels,
            fill='toself',
            name=comparison_label,
            line=dict(width=2),
            customdata=comparison_custom,
            hovertemplate=comparison_hover_template,
        )
    )
    # Keep polar axis positive even if both vectors collapse to zeros
    max_val = np.nanmax(np.concatenate([player_vec, comp_vec]))
    if np.isnan(max_val) or max_val == 0:
        max_val = 1.0
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.05])),
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig