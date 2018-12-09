import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objs as go
from datetime import datetime as dt

# Third party packages
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from collections import OrderedDict

# Project specific imports
from calc_uls import compute_capacity_surface
from calc_uls import utilization_ratio
from geometry import order_polygon_vertices
from geometry import line_hull_intersection
from geometry import point_to_point_dist_3d

field_color = '#F5F5F5'
field_pad = 10
margin = 10
headline_color = '#F1A44F'


def generate_table1(dataframe, max_rows=10):
    '''
    Return html table converted from pandas dataframe
    '''

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

COLORS = [
    {   'background': 'white', 'text': '#3CCC3C'  },
    {   'background': 'white', 'text': '#3CCC3C'  },
    {   'background': 'white', 'text': '#3CCC3C'  },
    {   'background': 'white', 'text': '#FF4C4C'  },]

def cell_style(value, min_value, max_value):
    style = {}
    if is_numeric(value):
        if value <= 0.25:
            style = {
                'backgroundColor': COLORS[0]['background'],
                'color': COLORS[0]['text']
            }
        elif value <= 0.5:
            style = {
                'backgroundColor': COLORS[1]['background'],
                'color': COLORS[1]['text']
            }
        elif value <= 1:
            style = {
                'backgroundColor': COLORS[2]['background'],
                'color': COLORS[2]['text']
            }
        elif value > 1:
            style = {
                'backgroundColor': COLORS[3]['background'],
                'color': COLORS[3]['text']
            }
    return style

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def generate_table(dataframe, max_rows=100):
    max_value = dataframe.max(numeric_only=True).max()
    min_value = dataframe.min(numeric_only=True).max()
    rows = []
    for i in range(min(len(dataframe), max_rows)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i]['UR[-]']
            style = cell_style(value, min_value, max_value)
            row.append(html.Td(dataframe.iloc[i][col], style=style))

        rows.append(html.Tr(row))

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        rows, style={'color': 'black', 'background': '#F8F8F8', 'fontSize': 12,
                     'height': 2, 'line-height': 2 })



app = dash.Dash('Section analysis')

app.layout = html.Div([
    html.H1('Capacity surface visualization', style={'color': headline_color, 'fontSize': 30}),

html.Div(
    [
        html.H2("Row with columns"),
        dbc.Row(dbc.Col(html.Div("A single column"))),
        dbc.Row(
            [
            dbc.Col(html.Div(
                dbc.InputGroup(
                    [dbc.InputGroupAddon("Ec", addon_type="append"),
                    dbc.Input(placeholder="200000"),
                    dbc.InputGroupAddon("MPa", addon_type="append"),]
                ),
            ),),
            dbc.Col(html.Div(
                dbc.InputGroup(
                    [dbc.InputGroupAddon("Ec", addon_type="append"),
                    dbc.Input(placeholder="200000"),
                    dbc.InputGroupAddon("MPa", addon_type="append"),]
                ),
            ),),
            dbc.Col(html.Div(
                dbc.InputGroup(
                    [dbc.InputGroupAddon("Ec", addon_type="append"),
                    dbc.Input(placeholder="200000"),
                    dbc.InputGroupAddon("MPa", addon_type="append"),]
                ),
            ),),
            ]
        ),
    ]
),

    html.Div(
        [
            dbc.Button('Primary', color='primary', outline=True, className='mr-1')
        ]
    ),


    html.Div(
        className='row',
        children=[
            html.Div(
                className='col s2 m2 l2',
                children=[
                    html.Div([
                        # Div for specifying the cross section vertices
                        html.Label('Cross section vertices', style={
                                   'color': headline_color, 'fontSize': 16}),
                        dash_table.DataTable(
                            id='section-vertices',
                            columns=(
                                [{'id': p, 'name': p}
                                    for p in ['x[mm]', 'y[mm]']]
                            ),
                            data=[{'x[mm]':  '200', 'y[mm]':  '200'},
                                  {'x[mm]': '-200', 'y[mm]':  '200'},
                                  {'x[mm]': '-200', 'y[mm]': '-200'},
                                  {'x[mm]':  '200', 'y[mm]': '-200'},
                                  {'x[mm]':  '', 'y[mm]': ''},
                                  {'x[mm]':  '', 'y[mm]': ''},
                                  {'x[mm]':  '', 'y[mm]': ''},
                                  {'x[mm]':  '', 'y[mm]': ''},
                                  {'x[mm]':  '', 'y[mm]': ''},
                                  {'x[mm]':  '', 'y[mm]': ''}],
                            editable=True,
                            style_table={
                            # 'maxHeight': '500',
                            # 'overflowY': 'scroll'
                            },
                        ),
                    ],),
                ], style={'backgroundColor': 'white', 'width': '47%', 'padding': field_pad, 'border-radius': 5,
                          'margin': 0, 'float': 'left'},
            ),
            html.Div(
                className='col s2 m2 l2',
                children=[
                    html.Div([
                        # Div for specifying rebar locations
                        html.Label('Rebar coordinates', style={
                                   'color': headline_color, 'fontSize': 16}),
                        dash_table.DataTable(
                            id='rebar-locations',
                            columns=(
                                [{'id': p, 'name': p} for p in ['xs[mm]', 'ys[mm]']]
                            ),
                            data=[{'xs[mm]':  '140', 'ys[mm]':  '140'},
                                  {'xs[mm]': '-140', 'ys[mm]':  '140'},
                                  {'xs[mm]': '-140', 'ys[mm]': '-140'},
                                  {'xs[mm]':  '140', 'ys[mm]': '-140'},
                                  {'xs[mm]':  '140', 'ys[mm]':  '0'},
                                  {'xs[mm]': '-140', 'ys[mm]':  '0'},
                                  {'xs[mm]':  '0',   'ys[mm]':  '140'},
                                  {'xs[mm]':  '0',   'ys[mm]':  '-140'},
                                  {'xs[mm]':  '',    'ys[mm]':   ''},
                                  {'xs[mm]':  '',    'ys[mm]':   ''},
                                  ],
                            editable=True
                        ),
                    ],),
                ], style={'backgroundColor': 'white', 'width': '47%', 'padding': field_pad, 'border-radius': 5,
                          'margin': 0, 'float': 'right'}
            ),
        ], style={'backgroundColor': '#FFFFFF', 'padding': 0, 'border-radius': 5},
    ),

    # Div for alerts
    html.Div([dbc.Alert('Some rebars are located outside the cross section!', color='primary')]),
    html.Div([dbc.Alert('Polygon is self-intersecting!', color='danger')]),

    # Div for graphs
    html.Div(
        className='row',
        children=[
            html.Div(
                className='col s2 m2 l2',
                children=[
                    html.Div([
                        dcc.Graph(
                            id='section-plot',
                        ),
                    ],),
                ], style={'backgroundColor': field_color, 'width': '47%', 'padding': field_pad, 'border-radius': 5,
                          'margin': margin, 'float': 'left'},
            ),
            html.Div(
                className='col s2 m2 l2',
                children=[
                    html.Div([
                        dcc.Graph(
                            id='capacity-surface',
                        ),
                    ],),
                ], style={'backgroundColor': field_color, 'width': '47%', 'padding': field_pad, 'border-radius': 5,
                          'margin': margin, 'float': 'right'}
            ),
        ], style={'backgroundColor': '#FFFFFF', 'padding': 0, 'border-radius': 5},
    ),

    # Div for load combinations and results
    html.Div(
        className='row',
        children=[
            html.Div(
                className='col s2 m2 l2',
                children=[
                    html.Label('Load combinations', style={'color': headline_color, 'fontSize': 16}),
                    dash_table.DataTable(
                        id='load-combs',
                        columns=(
                            [{'id': p, 'name': p}
                                for p in ['P[kN]', 'Mx[kNm]', 'My[kNm]']]
                        ),
                        data=[{'P[kN]':  '200', 'Mx[kNm]':  '200', 'My[kNm]': '200'},
                              {'P[kN]': '-200', 'Mx[kNm]':  '250', 'My[kNm]': '150'},
                              {'P[kN]': '-500', 'Mx[kNm]': '-300', 'My[kNm]': '75'},
                              {'P[kN]':  '300', 'Mx[kNm]': '-350', 'My[kNm]': '200'}],
                        editable=True
                    ),
                ], style={'backgroundColor': 'white', 'width': '47%', 'padding': field_pad, 'border-radius': 5,
                            'margin': 0, 'float': 'left'},
            ),
            html.Div(
                className='col s2 m2 l2',
                children=[
                html.Label('Result table', style={'color': headline_color, 'fontSize': 16}),
                    html.Div(
                        # Div for result table
                        id='result-table',
                        # Table is returned here from a callback
                    ),
                ], style={'backgroundColor': 'white', 'width': '47%', 'padding': field_pad, 'border-radius': 5,
                          'margin': 0, 'float': 'right'}
            ),
        ], style={'backgroundColor': '#FFFFFF', 'padding': 0, 'border-radius': 5},
    ),


# Hidden div for storing the computed capacity surface so it can be shared by many callbacks
# without computing it over and over each time
    html.Div(id='capacity-surface-results', style={'display': 'none'})

], className='container', style={'width': '95%'})



# ------------------------------
# CALLBACKS
# ------------------------------

# Compute capacity surface and store in hidden div
@app.callback(
    Output('capacity-surface-results', 'children'),
    [Input('section-vertices', 'data'),
    Input('section-vertices', 'columns'),
    Input('rebar-locations', 'data'),
    Input('rebar-locations', 'columns'), ])
def calc_cap_surf_and_store(xy, sv_col, xsys, rebar_col):
    # Read in section data as dataframes
    df_sv=pd.DataFrame(xy, columns = [c['name'] for c in sv_col])
    df_rebars=pd.DataFrame(xsys, columns = [c['name'] for c in rebar_col])
    
    # Extract section vertices, convert from strings to floats and close polgyon
    x=[float(c) for c in list(df_sv['x[mm]']) if c]
    y=[float(c) for c in list(df_sv['y[mm]']) if c]

    xr=[float(c) for c in list(df_rebars['xs[mm]'])if c]
    yr=[float(c) for c in list(df_rebars['ys[mm]'])if c]

    # Set calculation parameters
    eps_cu=0.0035       # NOTE eps_cu = 0.00035 in Eurocode for concrete strengths < C50
    fck=25    # [MPa]
    gamma_c=1.0
    fcd=fck/gamma_c
    Es=200*10**3  # [MPa]
    fyk=500     # [MPa]
    gamma_s=1.0
    fyd=fyk/gamma_s
    As=3.14159*25**2/4  # [mm^2]

    # Compute capacity surface
    P, Mx, My, _, _=compute_capacity_surface(
        x, y, xr, yr, fcd, fyd, Es, eps_cu, As, lambda_ = 0.80,  rotation_step = 5, vertical_step = 6)

    return pd.DataFrame({'P': P, 'Mx': Mx, 'My': My}).to_json(date_format='iso', orient='split')


@app.callback(
    Output('section-plot', 'figure'),
    [Input('section-vertices', 'data'),
    Input('section-vertices', 'columns'),
    Input('rebar-locations', 'data'),
    Input('rebar-locations', 'columns')])
def update_section_plot(xy, sv_col, xsys, rebar_col):
    # TODO Check if polygon defined by concrete vertices intersects itself
    # TODO Check if rebars are all inside polygon, display the 'warning' as text below graph. If calculate button is ___
    # TODO ___ pressed, display 'critical error' to user.
    # TODO Check if any rebars are on top of eachother
    # Read in data as dataframes
    df_sv = pd.DataFrame(xy, columns=[c['name'] for c in sv_col])
    df_rebars = pd.DataFrame(xsys, columns=[c['name'] for c in rebar_col])

    # Extract section vertices and convert from strings to float (if not empty strings)
    x = [float(c) for c in list(df_sv['x[mm]']) if c]
    y = [float(c) for c in list(df_sv['y[mm]']) if c]
    # Order polygon vertices so plot is displayed correctly
    x, y = order_polygon_vertices(x, y, x, y, counterclockwise=True)
    # Close polgyon
    x.append(x[0])
    y.append(y[0])
    # Extract rebar coordinates
    xs = df_rebars['xs[mm]']
    ys = df_rebars['ys[mm]']
    # Create plot
    concrete = go.Scatter(
        x=x,
        y=y,
        fill='toself',
        fillcolor='rgb(190, 190, 190)',
        mode='lines',
        line=dict(color='rgb(48,48,48)'),
        opacity=0.95,
    )
    rebars = go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        line=dict(
            color='rgb(20, 20, 20)',
        ),
        opacity=0.7,
        marker={
            'size': 10,
        },
    )

    return {
        'data': [concrete, rebars],
        'layout': go.Layout(
            # title = 'Cross Section Geometry',
            xaxis=dict(title='x', showgrid=False),
            yaxis=dict(title='y', scaleanchor='x',
                       scaleratio=1, showgrid=False),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            # margin = {'l': 50, 'b': 50, 't': 50, 'r': 50},
            hovermode='closest',
        )
    }


# Update capacity surface
@app.callback(
    Output(component_id='capacity-surface', component_property='figure'),
    [Input('capacity-surface-results', 'children'),
     Input('load-combs', 'data'),
     Input('load-combs', 'columns')])
def update_capacity_surface(cap_surf_results, loads, load_col):
    # Extract results from capacity surface calculation stored in hidden div
    df_cap_surf = pd.read_json(cap_surf_results, orient='split')
    P = df_cap_surf['P']
    Mx = df_cap_surf['Mx']
    My = df_cap_surf['My']
 
    # Since input is given in [MPa] and [mm], the results come out in [N] and [Nm]. Convert to [kN] and [kNm]
    P = [i/10**3 for i in P]
    Mx = [i/10**6 for i in Mx]
    My = [i/10**6 for i in My]

    # Read in load combinations as dataframe and convert to list of floats
    df_loads = pd.DataFrame(loads, columns=[c['name'] for c in load_col])
    Ped = [float(c) for c in df_loads['P[kN]']]
    Mxed = [float(c) for c in df_loads['Mx[kNm]']]
    Myed = [float(c) for c in df_loads['My[kNm]']]


    # Compute utilization ratio for each load combination
    ur = utilization_ratio(Ped, Mxed, Myed, P, Mx, My)

    # Extract safe combinations (UR <= 1.00)
    ur_safe = [u for u in ur if u <= 1.00]
    Ped_safe = [Ped[i] for i in range(len(ur)) if ur[i] <= 1.00]
    Mxed_safe = [Mxed[i] for i in range(len(ur)) if ur[i] <= 1.00]
    Myed_safe = [Myed[i] for i in range(len(ur)) if ur[i] <= 1.00]

    # Extract unsafe combinations (UR > 1.00)
    ur_unsafe = [u for u in ur if u > 1.00]
    Ped_unsafe = [Ped[i] for i in range(len(ur)) if ur[i] > 1.00]
    Mxed_unsafe = [Mxed[i] for i in range(len(ur)) if ur[i] > 1.00]
    Myed_unsafe = [Myed[i] for i in range(len(ur)) if ur[i] > 1.00]

    # Compute and plot convex hull of point cloud
    cap_surf = go.Mesh3d(x=Mx, y=My, z=P, alphahull=0,
                         opacity=0.4, color='#FFB653')
    points = go.Scatter3d(x=Mx, y=My, z=P, marker=dict(
        size=3, opacity=0.2), line=dict(width=0), mode='markers')

    # Create trace of safe combinations
    safe_combs = go.Scatter3d(x=Mxed_safe, y=Myed_safe, z=Ped_safe, marker=dict(
        size=5, color='green'), line=dict(width=0), mode='markers')

    # Create trace of unsafe combinations
    unsafe_combs = go.Scatter3d(x=Mxed_unsafe, y=Myed_unsafe, z=Ped_unsafe, marker=dict(
        size=5, color='red'), line=dict(width=0), mode='markers')


    return {
        'data': [cap_surf, points, safe_combs, unsafe_combs],
        'layout': go.Layout(
            # title = 'Capacity surface',
            scene=dict(
                xaxis=dict(title='Mx [kNm]'),
                yaxis=dict(title='My [kNm]'),
                zaxis=dict(title='P [kN]'),
            ),
            paper_bgcolor=field_color,
            # margin={'l': 50, 'b': 50, 't': 50, 'r': 50},
            # orientation=???,
            showlegend=False,
            margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
            # hovermode='closest',
        )
    }

# Update result table
@app.callback(
    Output('result-table', 'children'),
    [Input('capacity-surface-results', 'children'),
     Input('load-combs', 'data'),
     Input('load-combs', 'columns')])
def update_columns(cap_surf_results, loads, load_col):
    # TODO THIS COMPUTATION HAS ALREADY BEEN DONE FOR THE CAPACITY SURFACE. SHOULD BE STORED
    # ____ AND REUSED

    # Extract results from capacity surface calculation stored in hidden div
    df_cap_surf = pd.read_json(cap_surf_results, orient='split')
    P = df_cap_surf['P']
    Mx = df_cap_surf['Mx']
    My = df_cap_surf['My']

    # Since input is given in [MPa] and [mm], the results come out in [N] and [Nm]. Convert to [kN] and [kNm]
    P = [i/10**3 for i in P]
    Mx = [i/10**6 for i in Mx]
    My = [i/10**6 for i in My]

    # Read in load combinations as dataframe and convert to list of floats
    df_loads = pd.DataFrame(loads, columns=[c['name'] for c in load_col])
    Ped = [float(c) for c in df_loads['P[kN]']]
    Mxed = [float(c) for c in df_loads['Mx[kNm]']]
    Myed = [float(c) for c in df_loads['My[kNm]']]

    # Compute utilization ratio for each load combination
    ur = utilization_ratio(Ped, Mxed, Myed, P, Mx, My)

    Ped = [round(elem, 2) for elem in Ped]
    Mx = [round(elem, 2) for elem in Mx]
    My = [round(elem, 2) for elem in My]
    ur = [round(elem, 2) for elem in ur]

    data = OrderedDict()
    data["P[kN]"] = Ped
    data["Mx[kNm]"] = Mxed
    data["My[kNm]"] = Myed
    data["UR[-]"] = ur
    df = pd.DataFrame(data=data)
    # Update table
    return generate_table(df)




# external_css = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css']
external_css = ["https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"]
# external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

for css in external_css:
    app.css.append_css({'external_url': css})

for js in external_css:
    app.scripts.append_script({'external_url': js})

# # Activate the two lines below when running offline
# app.css.config.serve_locally = True
# app.scripts.config.serve_locally = True

if __name__ == '__main__':
    app.run_server(debug=True)
