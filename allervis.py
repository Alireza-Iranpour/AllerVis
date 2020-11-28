import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction


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
# ---------------------------------------------------
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
# ------------------------------------------------------------------------------

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# -------------------------------------------------------------------------------

allergen_options = [
    {"label": str(allergen), "value": str(allergen)} for allergen in allergen_data.keys()
]

app.layout = html.Div([
    html.Div([
        html.Div(
            [
                html.P(
                    "Filter by date (or select range in histogram):",
                    className="control_label",
                ),
                dcc.RangeSlider(
                    id="year_slider",
                    min=1960,
                    max=2017,
                    value=[1990, 2010],
                    className="dcc_control",
                ),
                html.P("Filter by allergen:", className="control_label"),
                dcc.RadioItems(
                    id="allergen_selector",
                    options=[
                        {"label": "All ", "value": "all"},
                        {"label": "Customize ", "value": "custom"},
                    ],
                    value="active",
                    labelStyle={"display": "inline-block"},
                    className="dcc_control",
                ),
                dcc.Dropdown(
                    id="allergens",
                    options=allergen_options,
                    multi=True,
                    value=list(allergen_data.keys()),
                    className="dcc_control",
                ),
                dcc.Checklist(
                    id="lock_selector",
                    options=[{"label": "Lock camera", "value": "locked"}],
                    className="dcc_control",
                    value=[],
                ),
            ],
            className="pretty_container four columns",
            id="cross-filter-options",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.H6(id="well_text"), html.P("No. of Wells")],
                            id="wells",
                            className="mini_container",
                        ),
                        html.Div(
                            [html.H6(id="gasText"), html.P("Gas")],
                            id="gas",
                            className="mini_container",
                        ),
                        html.Div(
                            [html.H6(id="oilText"), html.P("Oil")],
                            id="oil",
                            className="mini_container",
                        ),
                        html.Div(
                            [html.H6(id="waterText"), html.P("Water")],
                            id="water",
                            className="mini_container",
                        ),
                    ],
                    id="info-container",
                    className="row container-display",
                ),
                html.Div(
                    [dcc.Graph(id="count_graph")],
                    id="countGraphContainer",
                    className="pretty_container",
                ),
            ],
            id="right-column",
            className="eight columns",
        ),
    ],
        className="row flex-display",
    ),

],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# Radio -> multi
@app.callback(
    Output("allergens", "value"), [Input("allergen_selector", "value")]
)
def display_status(selector):
    if selector == "all":
        return list(allergen_data.keys())
    elif selector == "active":
        return ["AC"]
    return []


if __name__ == '__main__':
    app.run_server()