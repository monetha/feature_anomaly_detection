import plotly.graph_objects as go

def plot_cu_graph(cu_chart_df, rules):
    x = list(round(cu_chart_df['users_ratio'] * 100))
    x.insert(0, 0)

    y = list(round(cu_chart_df['ltv_ratio'] * 100))
    y.insert(0, 0)

    s = list(cu_chart_df['segment_'])
    r = [rules[i] for i in s]
    r.insert(0, '')

    fig = go.Figure(data=go.Scatter(x=x,
                                    y=y,
                                    text=r,
                                    name='By Segments',
                                    line=dict(color='rgb(49,130,189)'),
                                    hovertemplate=
                                    '<br><b>% of Users</i>: %{x}' + '<br><b>% LTV</b>: %{y}<br>' + '<br><b>Rule</b>: %{text}<br>'))

    fig.add_trace(go.Scatter(x=x,
                             y=x,
                             name='No model',
                             mode='lines',
                             line=dict(color='rgb(115,115,115)', dash='dash')))

    fig.update_layout(
        xaxis_title="% of Users",
        yaxis_title="% of total LTV",
        font=dict(family='Arial', size=16),
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            )
        ),
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )

    annotations = []

    for i, j, s in zip(x[1:], y[1:], list(cu_chart_df['segment_'])):
        annotations.append(dict(x=i - 1, y=j, xanchor='right', yanchor='middle',
                                text='S#' + str(s),
                                font=dict(family='Arial',
                                          size=12),
                                showarrow=False))

    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='Users contribution to total LTV by Segments',
                            font=dict(family='Arial',
                                      size=24,
                                      color='rgb(37,37,37)'),
                            showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.update_layout(hovermode='x unified')

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    return(fig)
