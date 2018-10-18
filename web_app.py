import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from pandas_datareader import data as web
from datetime import datetime as dt

# Imports - project specific files
import regex_point_syntax

app = dash.Dash('Hello World')

app.layout = html.Div([
    dcc.Markdown('''
#### Test

Dash supports [Markdown](http://commonmark.org/help). \
$$ x^2 $$
Markdown is a simple way $d$ to write and format text.
It includes a syntax for things like **bold text** and *italics*,
[links](http://commonmark.org/help), inline `code` snippets, lists,
quotes, and more.
***
'''),
    html.Div(
        className='row',
        children=[
            html.Div(
                className='three columns',
                children=[
                    html.Div([
                        # Div for specifying the cross section vertices
                        # html.Label('Define cross section vertices'),
                        dcc.Input(
                            id='section-vertices',
                            placeholder='Syntax: (x1, y1), (x2, y2), ..., (xn, yn)',
                            type='text',
                            value='(-8, 8), (8, 8), (8, -8), (-8, -8)',
                            style={'width': '100%'},
                        ),
                    ],),
                ], style={'backgroundColor': 'lightgrey', 'width': '30%', 'padding': '10', 'border-radius': '5'},
            ),
            html.Div(
                className='three columns',
                children=[
                    html.Div([
                        # Div for specifying the cross section vertices
                        dcc.Input(
                            id='rebar-locations',
                            placeholder='Syntax: (x1, y1), (x2, y2), ..., (xn, yn)',
                            # TODO See if there is a type containing only numbers and spceial characters ()[]{}
                            type='text',
                            value='(-5.6, 5.6), (0, 5.6), (5.6, 5.6), (5.6, 0), (5.6, -5.6), (0, -5.6), (-5.6, -5.6), (-5.6, 0)',
                            style={'width': '100%'},
                        ),
                    ],),
                ], style={'backgroundColor': 'lightgrey', 'width': '30%', 'padding': '10'}
            ),
            html.Div(
                className='three columns',
                children=[
                    html.Div([
                        # Div for specifying the cross section vertices
                        dcc.Input(
                            id='other2',
                            placeholder='Syntax: (x1, y1), (x2, y2), ..., (xn, yn)',
                            # TODO See if there is a type containing only numbers and spceial characters ()[]{}
                            type='text',
                            value='(-5.6, 5.6), (0, 5.6), (5.6, 5.6), (5.6, 0), (5.6, -5.6), (0, -5.6), (-5.6, -5.6), (-5.6, 0)',
                            style={'width': '100%'},
                        ),
                    ]),
                ], style={'backgroundColor': 'lightgrey', 'width': '30%', 'padding': '10'},
            ),
        ], style={'backgroundColor': 'grey', 'padding': '30'},
    ),
        



    # html.Div([
    #     # Div for specifying the cross section vertices
    #     html.Label('Define cross section vertices'),
    #     dcc.Input(
    #         id='section-vertices',
    #         placeholder='Syntax: (x1, y1), (x2, y2), ..., (xn, yn)',
    #         type='text',
    #         value='(-8, 8), (8, 8), (8, -8), (-8, -8)',
    #         style={'width': '45%'}
    #     ),

    #     dcc.Input(
    #         id='rebar-locations',
    #         placeholder='Syntax: (x1, y1), (x2, y2), ..., (xn, yn)',
    #         # TODO See if there is a type containing only numbers and spceial characters ()[]{}
    #         type='text',
    #         value='(-5.6, 5.6), (0, 5.6), (5.6, 5.6), (5.6, 0), (5.6, -5.6), (0, -5.6), (-5.6, -5.6), (-5.6, 0)',
    #         style={'width': '45%'}
    #     ),

    # Div for graphs
    html.Div(
        className="row",
        children=[
            html.Div(
                className="six columns",
                children=[
                    html.Div([
                        dcc.Graph(
                            id='section-plot',
                        ),
                    ]),
                ],
            ),
            html.Div(
                className="six columns",
                children=[
                    html.Div([
                        dcc.Graph(
                            id='capacity-surface',
                        ),
                    ]),
                ],
            ),
            html.Div(
                className="six columns",
                children=[
                    html.Div([
                        dcc.Graph(
                            id='loads',
                        ),
                    ]),
                ],
            ),
        ]
    ),


    # html.Div([
    #     # Div for specifying the cross section vertices
    #     html.Label('Define cross section vertices'),
    #     dcc.Graph(
    #         id='section-vertices',
    #         style={'width': '45%'}
    #     ),

    #     dcc.Input(
    #         id='rebar-locations',
    #         placeholder='Syntax: (x1, y1), (x2, y2), ..., (xn, yn)',
    #         # TODO See if there is a type containing only numbers and spceial characters ()[]{}
    #         type='text',
    #         value='(-5.6, 5.6), (0, 5.6), (5.6, 5.6), (5.6, 0), (5.6, -5.6), (0, -5.6), (-5.6, -5.6), (-5.6, 0)',
    #         style={'width': '45%'}
    #     ),

    # ]),


    # dcc.Dropdown(
    #     id = 'dropdown-to-hide-element',
    #     options=[
    #         {'label': 'Show element', 'value': 'on'},
    #         {'label': 'Hide element', 'value': 'off'}
    #     ],
    #     value = 'on'
    # ),
    
    # # Create Div to place a conditionally hidden element inside
    # html.Div([
    #     # Create element to hide, in this case an Input
    #     dcc.Input(
    #     id = 'element-to-hide',
    #     placeholder = 'something',
    #     type = 'something',
    #     value = 'Can you see me?',
    #     )
    # ])

    # ], style={'backgroundColor': 'grey'},),

# ], style={'color': 'black'},),

], style={'margin': 50})

# @app.callback(
#     Output(component_id='element-to-hide', component_property='style'),
#     [Input(component_id='dropdown-to-hide-element', component_property='value')])

# def show_hide_element(visibility_state):
#     if visibility_state == 'on':
#         return {'display': 'block'}
#     if visibility_state == 'off':
#         return {'display': 'none'}


# UPDATE SECTION PLOT FOR CONCRETE GEOMETRY AND REBAR INPUT
@app.callback(
    Output(component_id='section-plot', component_property='figure'),
    [Input(component_id='section-vertices', component_property='value'),
    Input(component_id='rebar-locations', component_property='value')]
)
def update_section_plot(section_vertices, rebar_locations):

    # Extract only correctly typed points
    x, y, c = regex_point_syntax.get_points(section_vertices)       # Concrete section vertices
    xr, yr, _ = regex_point_syntax.get_points(rebar_locations)     # Rebar locations

    # TODO Check if polygon defined by concrete vertices intersects itself
    # TODO Check if rebars are all inside polygon, display the 'warning' as text below graph. If calculate button is ___
    # TODO ___ pressed, display 'critical error' to user.

    # Append first point to create a closed polygon
    if len(c) > 1:
        x.append(x[0])
        y.append(y[0])

    # Create plot
    trace1 = go.Scatter(
        x = x,
        y = y,
        fill = 'toself',
        fillcolor = 'rgb(190, 190, 190)',
        mode = 'lines',
        line = dict(
            color = 'rgb(48,48,48)',
        ),
        opacity = 0.7,
        marker = {
            'size': 6,
            'line': {'width': 0.25, 'color': 'white'},
        },
    )
    trace2 = go.Scatter(
        x = xr,
        y = yr,
        mode = 'markers',
        line = dict(
            color = 'rgb(20, 20, 20)',
        ),
        opacity = 0.7,
        marker = {
            'size': 10,
        },
    )

    return {
        'data': [trace1, trace2],
        'layout': go.Layout(
            # title = 'Cross Section Geometry',
            xaxis = {'title': 'x'},
            yaxis = dict(title='y', scaleanchor='x', scaleratio=1), # Set aspect equal
            showlegend=False,
            margin = {'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode = 'closest',
        )
    }


# Update capacity surface
@app.callback(
    Output(component_id='capacity-surface', component_property='figure'),
    [Input(component_id='section-vertices', component_property='value'),
     Input(component_id='rebar-locations', component_property='value')]
)
def update_capacity_surface(section_vertices, rebar_locations):
    # Plot points from analysis
    # There should be a 'calculate' button
    pass


app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server(debug=True)
