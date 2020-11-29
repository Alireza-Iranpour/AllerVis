import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction

import plotly.express as px

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
                style={'margin': '0px, 10px', 'font-size': '30px',
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
                            value="custom",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                            ],
                        style={'margin': '10px'}),
                dcc.Dropdown(
                    id="allergens",
                    options=allergen_options,
                    multi=True,
                    value=list_of_allergens,
                    className="dcc_control",
                ),

                html.P("Select color scheme:", className="control_label"),
                html.Div([
                        dcc.RadioItems(
                            id="color_scheme_selector",
                            options=[
                                {"label": "Sequential ", "value": "sequential"},
                                {"label": "Diverging ", "value": "diverging"},
                            ],
                            value="sequential",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                            ],
                        style={'margin': '10px'}),

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
                    [dcc.Graph(id="map_graph")],
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
                    [dcc.Graph(id="stack_barchart_graph")],
                    id="stack_barchart_container",
                    className="pretty_container",
                ),
            ],
            id="stack_barchart_area",
            className="eight columns",
            style={"margin": "5px", "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.05), 0 6px 20px 0 rgba(0, 0, 0, 0.05)"}
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
    Output("allergens", "value"), [Input("allergen_selector", "value")]
)
def display_status(selector):
    if selector == "all":
        return list_of_allergens
    return []


# Graph
@app.callback(
    Output("stack_barchart_graph", "figure"), [Input("allergens", "value")]
)
def update_plot(selected_allergens):
    ascending = True
    concatenated['selected_set'] = concatenated.apply(lambda row: row[selected_allergens].sum(), axis=1)
    concatenated.sort_values('selected_set', ascending=ascending, inplace=True)
    fig = px.bar(concatenated, x='Entity', y=selected_allergens, orientation='v', height=500)

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


@app.callback(
    Output("map_graph", "figure"), [Input("allergens", "value"), Input("color_scheme_selector", "value")]
)
def update_plot(selected_allergens, color_scheme):

    concatenated['selected_set'] = concatenated.apply(lambda row: row[selected_allergens].sum(), axis=1)

    color_continuous_scale = px.colors.diverging.RdYlGn_r
    color_continuous_midpoint = concatenated['selected_set'].mean()

    if color_scheme == 'diverging':
        color_continuous_scale = px.colors.diverging.RdYlGn_r
        color_continuous_midpoint = concatenated['selected_set'].mean()
    elif color_scheme == 'sequential':
        color_continuous_scale = px.colors.sequential.Blues


    fig = px.choropleth(concatenated,
                        locations='Code',
                        color='selected_set',
                        hover_name="Entity",  # column to add to hover information
                        color_continuous_scale=color_continuous_scale,
                        # color_continuous_midpoint=color_continuous_midpoint,
                        labels={'selected_set': 'consumption'},
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


if __name__ == '__main__':
    app.run_server()
