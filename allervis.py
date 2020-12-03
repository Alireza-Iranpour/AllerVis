import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------
# Data handling

data_path = 'Food Allergies Data'
allergen_paths = {
    'Beef': f'{data_path}//beef-and-buffalo-meat-consumption-per-person.csv',
    'Seafood': f'{data_path}//fish-and-seafood-consumption-per-capita.csv',
    'Cereals': f'{data_path}//per-capita-consumption-of-cereals-by-commodity-type-daily-kilocalories.csv',
    'Egg': f'{data_path}//per-capita-egg-consumption-kilograms-per-year.csv',
    'Milk': f'{data_path}//per-capita-milk-consumption.csv',
    'Peanut': f'{data_path}//per-capita-peanut-consumption.csv'
}

list_of_allergens = list(allergen_paths.keys())
# ---------------------------------------------------
preprocess = False

if preprocess:
    allergen_data = {}
    for allergen, path in allergen_paths.items():
        df = pd.read_csv(path)
        allergen_data[allergen] = df

    # retrieving the most recent year data for each country
    all_dfs_most_recent_values = {}

    for allergen in allergen_data.keys():
        df = allergen_data[allergen]
        most_recent_year = df['Year'].iloc[-1]

        df_most_recent_values = df.groupby(['Code', 'Entity']).apply(lambda x: pd.Series(
            {allergen: x[df.columns[-1]].iloc[-1]}))

        all_dfs_most_recent_values[allergen] = df_most_recent_values

    # concatenating all allergens into one df
    concatenated = pd.concat(all_dfs_most_recent_values.values(), axis=1)

    # imputing the missing data with the given strategy
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = MinMaxScaler()

    for column in concatenated.columns:
        concatenated[column] = imputer.fit_transform(np.array(concatenated[column]).reshape(-1, 1))
        concatenated[column] = scaler.fit_transform(np.array(concatenated[column]).reshape(-1, 1))

    concatenated = concatenated.reset_index()

    # adding continent info
    continents = pd.read_csv(f'{data_path}//continents.csv', keep_default_na=False)
    continents['Continent'].replace({'NA': 'NAM'}, inplace=True)

    concatenated = concatenated.merge(continents, how='left', left_on='Code', right_on='alpha3').drop(
        columns=['alpha2', 'alpha3', 'numeric', 'fips', 'Country', 'Capital', 'Area in km²'])

    concatenated.to_csv(f'{data_path}//concatenated.csv', index=False)
# ------------------------------------------------------------------------------

concatenated = pd.read_csv(f'{data_path}//concatenated.csv')

app = dash.Dash()
server = app.server

# -------------------------------------------------------------------------------

allergen_options = [{"label": str(allergen), "value": str(allergen)} for allergen in list_of_allergens]

app.layout = html.Div([
    html.Div([
        html.Div(
            [
                html.Div([
                    html.P("AllerVis", className="control_label"),
                ],
                    style={'margin': '0px, 0px', 'font-size': '25px',
                           "font-family": "Helvetica", "font-weight": "bold"}
                ),
                html.P("Filter by allergen:", className="control_label"),
                html.Div([
                    dcc.RadioItems(
                        id="allergen_selector",
                        options=[
                            {"label": "All ", "value": "all"},
                            {"label": "Customize ", "value": "custom"},
                        ],
                        value="all",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),
                ],
                    style={'margin': '5px'}),
                dcc.Dropdown(
                    id="allergens",
                    options=allergen_options,
                    multi=True,
                    value=list_of_allergens,
                    className="dcc_control",
                ),
                html.Div([
                    html.Div([
                        html.P("Filter by region:", className="control_label"),
                    ],
                        style={'width': '30%', 'height': '2px', 'display': 'inline-block'}
                    ),
                    html.Div([
                        dcc.Dropdown(
                            id="regions",
                            options=[{"label": "World ", "value": "world"},
                                     {"label": "Europe ", "value": "europe"},
                                     {"label": "Asia ", "value": "asia"},
                                     {"label": "Africa ", "value": "africa"},
                                     {"label": "North America ", "value": "north america"},
                                     {"label": "South America ", "value": "south america"},
                                     {"label": "Oceania ", "value": "oceania"},

                                     ],
                            multi=False,
                            value='world',
                            className="dcc_control",
                        ),
                    ],
                        style={'margin': '5px', 'width': '200px', 'height': '20px',
                               'font-size': "100%", 'display': 'inline-block'}
                    )
                ]),

                html.Div([
                    html.Div([
                        html.P("Select map idiom:", className="control_label"),
                    ],
                        style={'width': '30%', 'height': '2px', 'display': 'inline-block'}
                    ),

                    html.Div([
                        dcc.RadioItems(
                            id="map_idiom_selector",
                            options=[
                                {"label": "Choropleth ", "value": "choropleth"},
                                {"label": "Bubble map ", "value": "bubble"}
                            ],
                            value="choropleth",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                    ],
                        style={'margin': '5px', 'display': 'inline-block'}
                    )
                ]),

                html.P("Select color scheme:", className="control_label"),
                html.Div([
                    dcc.RadioItems(
                        id="color_scheme_selector",
                        options=[
                            {"label": "Sequential ", "value": "sequential"},
                            {"label": "Diverging ", "value": "diverging"},
                            {"label": "Most Prevalent ", "value": "mpa"},
                            {"label": "Least Prevalent ", "value": "lpa"},
                        ],
                        value="sequential",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),
                ],
                    style={'margin': '5px'}),

            ],
            className="pretty_container four columns",
            id="cross-filter-options",
            style={"width": "38%", "padding": 10, "margin": "5px", "background-color": "#f9f9f9",
                   'display': 'inline-block', 'vertical-align': 'top', 'height': '1000',
                   'position': 'relative',
                   "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.05), 0 6px 20px 0 rgba(0, 0, 0, 0.05)",
                   "font-family": "Helvetica"},
        ),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="map_graph",
                               config={'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                                       'displaylogo': False})],
                    id="map_container",
                    className="pretty_container",
                ),
            ],
            id="map_area",
            className="eight columns",
            style={"margin": "5px", "width": "58%", 'display': 'inline-block', 'position': 'relative',
                   "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.05), 0 6px 20px 0 rgba(0, 0, 0, 0.05)"}
        ),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="stack_barchart_graph",
                               config={'modeBarButtonsToRemove': ['lasso2d'],
                                       'displaylogo': False})],
                    id="stack_barchart_container",
                    className="pretty_container",
                ),
            ],
            id="stack_barchart_area",
            className="eight columns",
            style={"margin": "5px",
                   "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.05), 0 6px 20px 0 rgba(0, 0, 0, 0.05)",
                   }
        ),
    ],
        className="row flex-display",
    ),

],
    id="mainContainer",
    style={"padding": 10, "background-color": "#f2f2f2"},
)


# Radio -> multi
@app.callback(
    Output("allergens", "value"),
    [Input("allergen_selector", "value")]
)
def display_status(selector):
    if selector == "all":
        return list_of_allergens
    return []


# -------------------------------------------------------------------------------------------
# Graph
@app.callback(
    Output("stack_barchart_graph", "figure"),
    [Input("allergens", "value"), Input("regions", "value")]
)
def update_plot(selected_allergens, selected_region):
    selected_allergens.sort()
    ascending = True

    concatenated['selected_set'] = concatenated.apply(lambda row: row[selected_allergens].sum(), axis=1)
    concatenated.sort_values('selected_set', ascending=ascending, inplace=True)

    region_concatenated = concatenated

    if selected_region == 'world':
        region_concatenated = concatenated
    elif selected_region == 'europe':
        region_concatenated = concatenated[concatenated['Continent'] == 'EU']
    elif selected_region == 'asia':
        region_concatenated = concatenated[concatenated['Continent'] == 'AS']
    elif selected_region == 'africa':
        region_concatenated = concatenated[concatenated['Continent'] == 'AF']
    elif selected_region == 'north america':
        region_concatenated = concatenated[concatenated['Continent'] == "NAM"]
    elif selected_region == 'south america':
        region_concatenated = concatenated[concatenated['Continent'] == 'SA']
    elif selected_region == 'oceania':
        region_concatenated = concatenated[concatenated['Continent'] == 'OC']

    fig = px.bar(region_concatenated, x='Entity', y=selected_allergens, orientation='v', height=500,
                 labels={'variable': 'Allergen',
                         'Entity': 'Country',
                         },
                 )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(
            l=10,
            r=10,
            b=20,
            t=20,
            pad=4
        ),
    )

    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)

    return fig


# -------------------------------------------------------------------------------------

@app.callback(
    Output("map_graph", "figure"),
    [Input("allergens", "value"),
     Input("regions", "value"),
     Input("map_idiom_selector", "value"),
     Input("color_scheme_selector", "value")
     ]
)
def update_plot(selected_allergens, selected_region, map_idiom, color_scheme):
    concatenated['selected_set'] = concatenated.apply(lambda row: row[selected_allergens].sum(), axis=1)
    concatenated['most_prevalent_allergen'] = concatenated.apply(
        lambda row: row[selected_allergens][row[selected_allergens] == row[selected_allergens].max()].index[0],
        axis=1)
    concatenated['least_prevalent_allergen'] = concatenated.apply(
        lambda row: row[selected_allergens][row[selected_allergens] == row[selected_allergens].min()].index[0],
        axis=1)

    color = 'selected_set'
    color_continuous_scale = px.colors.sequential.Blues
    color_continuous_midpoint = concatenated['selected_set'].mean()

    if color_scheme == 'diverging':
        color_continuous_scale = px.colors.diverging.RdYlGn_r
        color_continuous_midpoint = concatenated['selected_set'].mean()
    elif color_scheme == 'sequential':
        color_continuous_scale = px.colors.sequential.Blues

    elif color_scheme == 'mpa':
        concatenated.sort_values(['most_prevalent_allergen'], inplace=True)
        color = 'most_prevalent_allergen'

    elif color_scheme == 'lpa':

        concatenated.sort_values(['least_prevalent_allergen'], inplace=True)
        color = 'least_prevalent_allergen'

    fig = 0

    if selected_region == 'oceania':
        scope = 'world'
    else:
        scope = selected_region

    if map_idiom == 'choropleth':
        fig = px.choropleth(concatenated,
                            locations='Code',
                            scope=scope,
                            color=color,
                            hover_name="Entity",  # column to add to hover information
                            color_continuous_scale=color_continuous_scale,
                            # color_continuous_midpoint=color_continuous_midpoint,
                            labels={'selected_set': 'Prevalence',
                                    'most_prevalent_allergen': 'Most Prevalent Allergen',
                                    'least_prevalent_allergen': 'Least Prevalent Allergen',
                                    },
                            title=None,
                            height=344
                            )
        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                b=20,
                t=20,
                pad=4
            )
        )

        if selected_region == 'world':
            fig.update_geos(visible=False)

        if selected_region == 'oceania':
            fig.update_geos(visible=False, center=dict(lon=130, lat=-30), projection_scale=3)

    elif map_idiom == 'bubble':

        fig = px.scatter_geo(concatenated,
                             locations='Code',
                             scope=scope,
                             color=color,
                             size='selected_set',
                             hover_name="Entity",  # column to add to hover information
                             color_continuous_scale=color_continuous_scale,
                             labels={'selected_set': 'Prevalence',
                                     'most_prevalent_allergen': 'Most Prevalent Allergen',
                                     'least_prevalent_allergen': 'Least Prevalent Allergen'},
                             title=None,
                             height=339
                             )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                b=20,
                t=20,
                pad=4
            )
        )

        if selected_region == 'oceania':
            fig.update_geos(center=dict(lon=130, lat=-30),
                            projection_scale=3)

        # fig.update_config({'modeBarButtonsToRemove': ['lasso2d']})
    return fig


# ------------------------------------------------------------------------

"""@app.callback(
    Output("map_graph", "figure"), [Input("allergens", "value"), Input("color_scheme_selector", "value")]
)
def update_plot(selected_allergens, color_scheme):
    concatenated['selected_set'] = concatenated.apply(
        lambda row: row[selected_allergens].sum(), axis=1)
    concatenated['most_prevalent_allergen'] = concatenated.apply(
        lambda row: row[selected_allergens][row[selected_allergens] == row[selected_allergens].max()].index[0], axis=1)

    color_continuous_scale = px.colors.diverging.RdYlGn_r
    color_continuous_midpoint = concatenated['selected_set'].mean()
    color = 'selected_set'

    if color_scheme == 'diverging':
        color_continuous_scale = px.colors.diverging.RdYlGn_r
        color_continuous_midpoint = concatenated['selected_set'].mean()
    elif color_scheme == 'sequential':
        color_continuous_scale = px.colors.sequential.Blues
    elif color_scheme == 'mpa':
        color = 'most_prevalent_allergen'

    fig = px.scatter_geo(concatenated,
                         locations='Code',
                         color=color,
                         hover_name="Entity",  # column to add to hover information
                         color_continuous_scale=color_continuous_scale,
                         labels={'selected_set': 'Prevalence'},
                         title=None,
                         height=300

                         )

    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            b=20,
            t=20,
            pad=4
        ),
    )

    return fig
"""

# ---------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run_server()
