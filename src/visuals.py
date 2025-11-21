import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

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
    'radial_score_chart',
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
    comparison_values=None,
    feature_labels=None,
    player_label='Player',
    comparison_label='Comparison',
    title='Player Profile Comparison',
    player_hover_values=None,
    comparison_hover_values=None,
    hover_precision=2
):
    # Overlay player vs optional comparison vector to highlight strengths quickly
    if feature_labels is None:
        feature_labels = []
    player_vec = np.asarray(player_values, dtype=float)
    labels = list(feature_labels)
    if len(player_vec) != len(labels):
        raise ValueError('player feature lengths do not align')
    if len(labels) < 3:
        raise ValueError('radar charts need at least three features')

    comp_vec = None
    if comparison_values is not None:
        comp_vec = np.asarray(comparison_values, dtype=float)
        if len(comp_vec) != len(labels):
            raise ValueError('comparison feature lengths do not align')

    wrap_labels = labels + [labels[0]]
    player_points = np.append(player_vec, player_vec[0])
    value_fmt = f'.{max(0, int(hover_precision))}f'
    player_custom = None
    player_hover_template = (
        '<b>%{theta}</b><br>'
        f'{player_label}: %{{r:{value_fmt}}}<extra></extra>'
    )

    comparison_custom = None
    comparison_hover_template = (
        '<b>%{theta}</b><br>'
        f'{comparison_label}: %{{r:{value_fmt}}}<extra></extra>'
    )

    if player_hover_values is not None and comparison_hover_values is not None:
        player_hover = np.asarray(player_hover_values, dtype=float)
        comparison_hover = np.asarray(comparison_hover_values, dtype=float)
        if player_hover.shape[0] != len(labels) or comparison_hover.shape[0] != len(labels):
            raise ValueError('hover value lengths must match feature labels')
        player_hover = player_hover.reshape(len(labels))
        comparison_hover = comparison_hover.reshape(len(labels))
        hover_pairs_player = np.column_stack([player_hover, comparison_hover])
        hover_pairs_comparison = np.column_stack([comparison_hover, player_hover])
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
    else:
        if player_hover_values is not None:
            player_hover = np.asarray(player_hover_values, dtype=float)
            if player_hover.shape[0] != len(labels):
                raise ValueError('player hover values must match feature labels')
            if player_hover.ndim == 1:
                player_hover = player_hover.reshape(-1, 1)
            player_custom = np.vstack([player_hover, player_hover[0]])
            player_hover_template = (
                '<b>%{theta}</b><br>'
                + '<br>'.join(
                    f'{player_label} metric {idx + 1}: %{{customdata[{idx}]:{value_fmt}}}'
                    for idx in range(player_hover.shape[1])
                )
                + '<extra></extra>'
            )
        if comparison_hover_values is not None and comp_vec is not None:
            comparison_hover = np.asarray(comparison_hover_values, dtype=float)
            if comparison_hover.shape[0] != len(labels):
                raise ValueError('comparison hover values must match feature labels')
            if comparison_hover.ndim == 1:
                comparison_hover = comparison_hover.reshape(-1, 1)
            comparison_custom = np.vstack([comparison_hover, comparison_hover[0]])
            comparison_hover_template = (
                '<b>%{theta}</b><br>'
                + '<br>'.join(
                    f'{comparison_label} metric {idx + 1}: %{{customdata[{idx}]:{value_fmt}}}'
                    for idx in range(comparison_hover.shape[1])
                )
                + '<extra></extra>'
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

    if comp_vec is not None:
        comp_points = np.append(comp_vec, comp_vec[0])
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

    axis_values = player_vec
    if comp_vec is not None:
        axis_values = np.concatenate([axis_values, comp_vec])
    max_val = np.nanmax(axis_values)
    if np.isnan(max_val) or max_val == 0:
        max_val = 1.0
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.05])),
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _resolve_color_scale(scale):
    if scale is None:
        return pc.sequential.YlOrBr
    if isinstance(scale, str):
        return pc.get_colorscale(scale)
    return scale


def _score_to_color(value, max_value, color_scale=None):
    scale = _resolve_color_scale(color_scale)
    if max_value <= 0:
        max_value = 1.0
    norm = min(max(value / max_value, 0.0), 1.0)
    return pc.sample_colorscale(scale, norm)[0]


def radial_score_chart(
    values,
    labels,
    *,
    max_value=100.0,
    title='Ability score (0-100)',
    color_scale=None,
    text_font='Inter'
):
    values = np.asarray(values, dtype=float)
    labels_list = list(labels)
    if values.size != len(labels_list):
        raise ValueError('values and labels must be the same length')
    if values.size < 3:
        raise ValueError('need at least three values for radial chart')

    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    theta = np.linspace(0, 360, num=len(clean_values), endpoint=False)
    width = (360 / len(clean_values)) * 0.85
    colors = [_score_to_color(v, max_value, color_scale) for v in clean_values]

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=clean_values,
            theta=theta,
            width=width,
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.6)', width=1.2),
            ),
            opacity=0.95,
            customdata=np.array(labels_list).reshape(-1, 1),
            hovertemplate='<b>%{customdata[0]}</b><br>Score: %{r:.1f}<extra></extra>',
        )
    )

    text_r = np.clip(clean_values + max_value * 0.05, 0, max_value * 1.05)
    fig.add_trace(
        go.Scatterpolar(
            r=text_r,
            theta=theta,
            mode='markers',
            marker=dict(
                symbol='square',
                size=20,
                color='rgba(0,0,0,0.5)',
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            hoverinfo='skip',
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=text_r,
            theta=theta,
            mode='text',
            text=[f'{v:.0f}' for v in clean_values],
            textfont=dict(color='#ffffff', size=12, family=text_font),
            hoverinfo='skip',
            textposition='middle center',
        )
    )

    fig.update_layout(
        title=title,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(range=[0, max_value], showticklabels=False, ticks=''),
            angularaxis=dict(
                showline=False,
                tickmode='array',
                tickvals=theta,
                ticktext=labels_list,
                tickfont=dict(size=12, color='#111111', family=text_font),
                rotation=90,
                direction='clockwise',
            ),
        ),
        showlegend=False,
        margin=dict(l=100, r=100, t=60, b=80),
    )
    return fig