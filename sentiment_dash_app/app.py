from dash import Dash, dcc, html, Input, Output, State, html, dash_table, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import snowflake.connector as sc
from urllib.parse import quote
# from azure.identity import ManagedIdentityCredential
# from azure.keyvault.secrets import SecretClient
from utils import *
from config import *

# set snowflake connection

conn_params = {
    'account': ACCOUNT,
    'user': USER,
    # 'private_key_file': private_key_file,
    'private_key': PRIVATE_KEY,
    'warehouse': WAREHOUSE,
    'database': DATABASE,
    'schema': SCHEMA
}

conn = sc.connect(**conn_params)

# load data

reviews_fact_query = 'SELECT * FROM SUSHI.NYC.SUSHI_PLACE_REVIEWS_FACT'
places_dim_query = 'SELECT * FROM SUSHI.NYC.SUSHI_PLACES_DIMENSION'
sentiment_fact_query = 'SELECT * FROM SUSHI.NYC.SUSHI_REVIEW_SENTIMENT_FACT'
sent_metrics_by_loc_query = 'SELECT * FROM SUSHI.NYC.V_SENTIMENT_METRICS_BY_LOCATION'
avg_sent_query = 'SELECT * FROM SUSHI.NYC.V_AVG_SENTIMENT_SCORES'
review_info_query = 'SELECT * FROM SUSHI.NYC.V_ALL_REVIEW_INFO'

# tables
sushi_reviews = pd.read_sql(reviews_fact_query, con=conn)
sushi_places = pd.read_sql(places_dim_query, con=conn)
sushi_sentiment = pd.read_sql(sentiment_fact_query, con=conn)

# views
v_sentiment_metrics_by_loc = pd.read_sql(sent_metrics_by_loc_query, con=conn)
# Ensure LATITUDE and LONGITUDE columns are numeric
v_sentiment_metrics_by_loc['LATITUDE'] = pd.to_numeric(v_sentiment_metrics_by_loc['LATITUDE'], errors='coerce')
v_sentiment_metrics_by_loc['LONGITUDE'] = pd.to_numeric(v_sentiment_metrics_by_loc['LONGITUDE'], errors='coerce')
# get weighted confidence score per location
v_sentiment_metrics_by_loc['WEIGHTED_CONFIDENCE_SCORE'] = weighted_confidence_score(v_sentiment_metrics_by_loc)
# average sentiment scores overall
avg_sent_scores = pd.read_sql(avg_sent_query, con=conn)#.T.rename({0:'Average_Score'}, axis=1).reset_index(names='Sentiment Metric')
# avg_sent_scores.columns = [col.upper() for col in avg_sent_scores.columns]
v_review_info = pd.read_sql(review_info_query, con=conn)

# Default values for dropdowns
default_borough = ["All"]
default_price_level = ["All"]
default_restaurant_type = ['All']
default_rating_count = [0, 100]  # Default range for a slider
default_wc_title = 'Most Frequent Words - Citywide'
default_rating_title = 'Average Citywide Sushi Rating'
agg_positive_rating_title = 'Average Positive Sentiment Score'
agg_neutral_rating_title = 'Average Neutral Sentiment Score'
agg_negative_rating_title = 'Average Negative Sentiment Score'
default_bar_chart = 'All'
default_barchart_title = 'Sushi Sentiment Scores'
second_row_visuals_height = 400
second_row_visuals_width = 800
default_google_href=""
default_google_style={"display": "none"} 


# figure creation

default_map_fig = px.scatter_map(
    v_sentiment_metrics_by_loc,
    lat="LATITUDE",
    lon="LONGITUDE",
    hover_name="DISPLAYNAME_TEXT",  # Shows location name on hover
    custom_data ='PLACE_ID',
    zoom=11,  # Zoom level (adjust for best view)
    height=550,
    width=1200,
    hover_data=['ADDRESS', 'USERRATINGCOUNT']
)

# Add an empty trace for the selected location
default_map_fig.add_trace(
    px.scatter_map(
        pd.DataFrame(),  # Empty dataframe for the selected location trace
        lat=[],
        lon=[],
        hover_name=[]
    ).data[0]
)

# Use OpenStreetMap (OSM) as the base layer
default_map_fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0, "t":0, "l":0, "b":0}
)

# Define dropdown options
column_groups = {
    "NLTK VADER": ["AVG_VADER_NEG", "AVG_VADER_NEU", "AVG_VADER_POS", "AVG_VADER_COMPOUND"],
    "Twitter-roBERTa-base": ["AVG_BASE_NEG", "AVG_BASE_NEU", "AVG_BASE_POS"],
    "Foody Bert": ["AVG_FOODY_NEG", "AVG_FOODY_NEU", "AVG_FOODY_POS"],
}

group_vals = [j for i in column_groups.values() for j in i]

# Convert dictionary to DataFrame
flattened_metrics = []
for key, values in column_groups.items():
    for value in values:
        flattened_metrics.append({"Model": key, "Sentiment Metric": value})

# add to dictionary

column_groups['All'] = group_vals

# Generate dropdown options for PRICE_LEVEL
price_level_options = [
    {"label": "All", "value": "All"},  # Option to show all price levels
    {"label": "Null", "value": "Null"}  # Option to show rows with null price levels
] + [
    {"label": str(price_level), "value": str(price_level)}  # Convert all values to strings
    for price_level in v_sentiment_metrics_by_loc['PRICELEVEL'].dropna().unique()
]

sentiment_model_dict = {
        "NLTK VADER": ['VADER_POS', 'VADER_NEG', 'VADER_NEU'],
        "Twitter-roBERTa-base": ['BASE_POS', 'BASE_NEG', 'BASE_NEU'],
        "Foody Bert": ["FOODY_POS", "FOODY_NEG", "FOODY_NEU"],
        "Special": ['TEXTBLOB_POLARITY', 'TEXTBLOB_SUBJECTIVITY', 'VADER_COMPOUND']
    }

# metrics
avg_rating = v_sentiment_metrics_by_loc['OVERALL_RATING'].mean()
presentation_avg_rating = f"### {round(avg_rating, 2)} ⭐"

pos_avg_rating = sushi_sentiment['BASE_POS'].mean()
presentation_pos_avg_rating = f"### {round(pos_avg_rating, 2)} 😊"

neu_avg_rating = sushi_sentiment['BASE_NEU'].mean()
presentation_neu_avg_rating = f"### {round(neu_avg_rating, 2)} 😐"

neg_avg_rating = sushi_sentiment['BASE_NEG'].mean()
presentation_neg_avg_rating = f"### {round(neg_avg_rating, 2)} 😞"

default_wordcloud_image = wordcloud_viz(sushi_sentiment, gen_whole_wordcloud=True)

rating_count_min = min(v_sentiment_metrics_by_loc['USERRATINGCOUNT'].tolist())
rating_count_max = max(v_sentiment_metrics_by_loc['USERRATINGCOUNT'].tolist())

overall_rating_min = min(v_sentiment_metrics_by_loc['OVERALL_RATING'].tolist())
overall_rating_max = max(v_sentiment_metrics_by_loc['OVERALL_RATING'].tolist())

weighted_conf_min = min(v_sentiment_metrics_by_loc['WEIGHTED_CONFIDENCE_SCORE'].tolist())
weighted_conf_max = max(v_sentiment_metrics_by_loc['WEIGHTED_CONFIDENCE_SCORE'].tolist())


review_data_table = v_review_info.merge(sushi_places[['ID', 'DISPLAYNAME_TEXT', 'ADDRESS', 'BOROUGH']], left_on='PLACE_ID', right_on='ID')
visual_cols = ['REVIEW_ID', 'DISPLAYNAME_TEXT', 'TEXT', 'ADDRESS', 'BOROUGH', 'RATING', 'PUBLISHTIME']
visible_review_data_table = review_data_table[visual_cols]

# Initialize Dash app
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.SUPERHERO, 
                                 dbc.icons.BOOTSTRAP, 
                                 dbc.icons.FONT_AWESOME,
                                 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css']
           )
# Initialize Flask server
server = app.server

# reset buttons for layout
reset_button = dbc.Button(
    "Reset Filters",
    id="reset-button",
    color="danger",  # Red color for emphasis
    className="mb-3"  # Add margin below the button
)

table_reset_button = dbc.Button(
    "Reset Table",
    id="table-reset-button",
    color="danger",  # Red color for emphasis
    className="mb-3"  # Add margin below the button
)

# Define the sidebar (navigation bar)
sidebar = html.Div(
    [
        html.H2("Filters", className="display-4"),
        html.Hr(),
        html.P("Select a Model:", className="lead"),
        dcc.Dropdown(
            id="dropdown-model",
            options=[{"label": key, "value": key} for key in column_groups.keys()],
            value='All',  # Default selection
            clearable=False,
            multi=False
        ),
        # Add more filters or text here if needed
        # Borough Filter
        html.P("Borough", className="lead"),
        dcc.Dropdown(
            id="dropdown-borough",
            options=[{"label": borough, "value": borough} for borough in list(v_sentiment_metrics_by_loc['BOROUGH'].unique()) + ['All']],
            value=["All"],  # Default selection
            clearable=True,
            multi=True
        ),
        # Primary Type Filter
        html.P("Restaurant Focus", className="lead"),
        dcc.Dropdown(
            id="dropdown-restaurant-type",
            options=[{"label": type_, "value": type_} for type_ in list(v_sentiment_metrics_by_loc['PRIMARY_TYPE'].unique()) + ['All']],
            value=["All"],  # Default selection
            clearable=True,
            multi=True
        ),
        # Price Level Filter
        html.P("Price Level", className="lead"),
        dcc.Dropdown(
            id="dropdown-price-level",
            options=price_level_options,
            value=["All"],  # Default selection
            clearable=True,
            multi=True
        ),
        # User Rating Count Filter
        html.P("Count of User Ratings", className="lead"),
        dcc.RangeSlider(
            rating_count_min,
            rating_count_max,
            id="slider-user-rating-count",
            value=[rating_count_min,
                   rating_count_max],  # Default selection
        ),
        # Overall Rating Filter
        html.P("Overall Rating", className="lead"),
        dcc.RangeSlider(
            overall_rating_min,
            overall_rating_max,
            id="slider-overall-rating",
            value=[overall_rating_min,
                   overall_rating_max,],  # Default selection
        ),
        # Weighted Confidence Filter
        html.Span([
        html.P("Rating Confidence Score", className="lead", style={"display": "inline-block", "margin-right": "5px"}),
        html.I(
            className="fas fa-info-circle",
            id="weighted-confidence-tooltip",
            style={"cursor": "pointer", "color": "white", "font-size": "1.2rem", "vertical-align": "middle"}
        ),
    ], style={"display": "inline-flex", "align-items": "center"}),  # Make the whole span inline and aligned,
        dbc.Tooltip(
            "The Rating Confidence Score reflects how reliable the overall rating is, based on the number of reviews. "
            "Higher scores indicate greater confidence in the rating.",
            target="weighted-confidence-tooltip",
            placement="right",
        ),
        dcc.RangeSlider(
            weighted_conf_min,
            weighted_conf_max,
            id="slider-weighted_conf",
            value=[weighted_conf_min,
                   weighted_conf_max,],  # Default selection
        ),
        # Reservable Filter
        html.P("Reservable", className="lead"),
        dcc.Dropdown(
            id="dropdown-reservable",
            options=[
            {'label': 'All', 'value': 'All'},        # Option to include both True and False
            {'label': 'True', 'value': True},       
            {'label': 'False', 'value': False},
            {'label': 'Unknown', 'value': 'Unknown'}        
            ],
            value=["All"],  # Default selection
            clearable=True,
            multi=False
        ),
        # Serves Alcohol Filter
        html.P("Serves Alcohol", className="lead"),
        dcc.Dropdown(
            id="dropdown-alcohol",
            options=[
            {'label': 'All', 'value': 'All'},    
            {'label': 'True', 'value': True},        
            {'label': 'False', 'value': False},
            {'label': 'Unknown', 'value': 'Unknown'}       
            ],
            value=["All"],  # Default selection
            clearable=True,
            multi=False
        ),
        # Wheelchair Accessible Filter
        html.P("Wheelchair Accessibility", className="lead"),
        dcc.Dropdown(
            id="dropdown-wc-accessible-entrance",
            options=[
            {'label': 'All', 'value': 'All'},    
            {'label': 'True', 'value': True},        
            {'label': 'False', 'value': False},
            {'label': 'Unknown', 'value': 'Unknown'}      
            ],
            value=["All"],  # Default selection
            clearable=True,
            multi=False
        )
    ],
    style={"padding":"2rem 1rem", 
        #    "background-color": "#f8f9fa", 
        #    "color":'black', 
           "justify":"left",
           "height":"100vh"},
)

content = html.Div(
    [
        html.H1("Sushi in NYC - Sentiment Analysis", style={"textAlign": "center"}),
        html.H3("🍙 Click on a location to learn more about it 🍣", style={"textAlign": "center"}),
        dcc.Store(id='filter-store', data={}),
        html.Div(
            dbc.Row([
                dbc.Col([
                    reset_button
                ])
            ])
        ),
    # map and cards
    dbc.Container(
        [ 
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='sushi-map', figure=default_map_fig)
            ],
                width=8,  # Set the column width (adjust as needed)
                className="mx-auto p-0"  # Center the column within the row
            )  # Embed the map
        ], justify='center',
           className="g-0"
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(default_rating_title, className="avg-card-title text-center", id='rating-title'),
                                dcc.Markdown(presentation_avg_rating, className="display-3 text-primary text-center", id='loc-rating'),
                            ],
                            className="d-flex flex-column justify-content-center align-items-center"
                        ),
                        className="h-100",
                    ),
                    width=2,
                    className="p-0",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(agg_positive_rating_title, className="pos-card-title text-center", id='pos-rating-title'),
                                dcc.Markdown(presentation_pos_avg_rating, className="display-3 text-primary text-center", id='pos-loc-rating'),
                            ],
                            className="d-flex flex-column justify-content-center align-items-center"
                        ),
                        className="h-100",
                    ),
                    width=2,
                    className="p-0",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(agg_neutral_rating_title, className="neu-card-title text-center", id='neu-rating-title'),
                                dcc.Markdown(presentation_neu_avg_rating, className="display-3 text-primary text-center", id='neu-loc-rating'),
                            ],
                            className="d-flex flex-column justify-content-center align-items-center"
                        ),
                        className="h-100",
                    ),
                    width=2,
                    className="p-0",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(agg_negative_rating_title, className="neg-card-title text-center", id='neg-rating-title'),
                                dcc.Markdown(presentation_neg_avg_rating, className="display-3 text-primary text-center", id='neg-loc-rating'),
                            ],
                            className="d-flex flex-column justify-content-center align-items-center"
                        ),
                        className="h-100",
                    ),
                    width=2,
                    className="p-0",
                )
            ],
            className="g-0",
            justify="center",
        ),
            # Button Row
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "See on Google",
                            id="google-button",
                            color="primary",
                            href=default_google_href,
                            target="_blank",  # Open in a new tab
                            style=default_google_style  # Initially hidden
                        ),
                        width=12,  # The button takes the full width of the row
                        className="d-flex justify-content-center",
                    ),
                    justify="center",
                    className="g-0"
                ),  
    ],
                fluid=True,
                className="p-0"
            ),

        html.Br(),

        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(default_barchart_title, id='barchart-title', className="text-center"),
                            dcc.Graph(id='sentiment-bar-chart', style={"width": "100%"})  # Sentiment bar chart
                        ],
                        width=6,
                        className="pe-3"  # Padding right using Bootstrap utility
                    ),  # Left column (sentiment bar chart)

                    dbc.Col(
                        [
                            html.H4(f"{default_wc_title}", id='wordcloud-title', className="text-center"),
                            html.Img(id='wordcloud-image', src="", style={"width": "100%", "height": "auto"})  # Wordcloud
                        ],
                        width=6,
                        className="ps-3"  # Padding left using Bootstrap utility
                    )  # Right column (Wordcloud)
                ],
                justify="between"
            ),
            className="py-4",  # Padding top and bottom for the container
            fluid=True
        ),

        html.Br(),
        # Dropdown for selecting 3D scatter plot columns
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(  # Ensures vertical stacking
                                    [
                                        html.H4(
                                            "Select Sentiment Model for 3D Review Visualization",
                                            style={"text-align": "center", "display": "block", "margin-bottom": "5px"},
                                        ),
                                        html.H5(
                                            "Click on a point in the Visual to See the Review below",
                                            style={"text-align": "center", "display": "block", "margin-bottom": "5px"},
                                        ),
                                        dcc.Dropdown(
                                            id="scatter-column-selector",
                                            options=[
                                                {"label": k, "value": k} for k in sentiment_model_dict.keys()
                                            ],
                                            value="NLTK VADER",
                                            clearable=False,
                                            style={
                                                "width": "1200px",
                                                "margin": "auto",
                                                "display": "block",
                                                "text-align": "center",
                                            },
                                        ),
                                    ],
                                    style={"width": "100%", "text-align": "center"},
                                ),
                            ],
                            width=12,
                            style={
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "flex-direction": "column",
                            },
                        ),  # Centering Dropdown

                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(  # Group the graph and icon in an inline-flex div
                                        [
                                            dcc.Graph(
                                                id="sentiment-scatter-3d",
                                                style={
                                                    "width": "1180px",  # Slightly smaller than the dropdown to account for icon
                                                    "margin": "auto",
                                                    "display": "inline-block",
                                                    "vertical-align": "middle",
                                                },
                                            ),
                                            html.I(
                                                className="fas fa-info-circle",
                                                id="3d-scatter-viz",
                                                style={
                                                    "cursor": "pointer",
                                                    "color": "white",
                                                    "font-size": "1.6rem",
                                                    "margin-left": "45px",
                                                    "vertical-align": "middle",
                                                },
                                            ),
                                        ],
                                        style={"display": "inline-flex", "align-items": "center", "justify-content": "center", "width": "1200px"},
                                    ),
                                    dbc.Tooltip(
                                        "The NLTK VADER model is a rule-based, lexicon-driven model that uses a pre-defined dictionary of words and their sentiment scores. Hence, the model can fail to recognize sarcasm or context in language."
                                        " Twitter-RoBERTa is a transformer-based model using deep learning, trained on a large corpus of Twitter data."
                                        " This has allowed it to learn contextual relationships and nuance."
                                        " Notice how when visualized, the classifications of NLTK VADER are less confident than RoBERTa's."
                                        " The shape of the outputs for NLTK peak towards neutrality, while for RoBERTa it is less pronounced.",
                                        target="3d-scatter-viz",
                                        placement="bottom",
                                    ),
                                ],
                                style={"display": "flex", "justify-content": "center", "align-items": "center"},
                            ),
                            width=12,
                            style={"display": "flex", "justify-content": "center"},
                        ),
                    ],
                    justify="center",
                ),

                html.Br(),

                dbc.Row(
                    [
                        html.Div(
                            dbc.Row([
                                dbc.Col([
                                    table_reset_button
                                ], style={"display": "flex", "justify-content": "left"})
                            ])
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H4(
                                            "Table of Sushi Reviews",
                                            style={"text-align": "center", "display": "block", "margin-bottom": "5px"}
                                        ),
                                        dash_table.DataTable(
                                            id='review-table',
                                            columns=[{"name": i, "id": i} if i != 'DISPLAYNAME_TEXT' else {"name":'RESTAURANT', "id":i} for i in visible_review_data_table.columns],
                                            data=visible_review_data_table.to_dict('records'),
                                            style_table={
                                                'backgroundColor': 'black',
                                                'color': 'white',
                                                'maxWidth': '5000px',  # Set a maximum width for the table
                                                'width': '1500px',      # Make the table responsive within the maxWidth
                                                'margin': '0 auto'     # Center the table on the page
                                            },
                                            style_header={
                                                'backgroundColor': 'black',
                                                'color': 'white',
                                                'fontWeight': 'bold'
                                            },
                                            style_data={
                                                'backgroundColor': 'black',
                                                'color': 'white'
                                            },
                                            style_cell={
                                                'backgroundColor': 'black',
                                                'color': 'white',
                                                'whiteSpace': 'normal',  # Allow text to wrap
                                                'textAlign': 'center',      # Align text to the left
                                                'maxWidth': '300px',      # Set a maximum width for each column
                                                'overflow': 'hidden',     # Hide overflow
                                                'textOverflow': 'ellipsis' # Add ellipsis for overflow text (optional)
                                            },
                                            style_filter={
                                                'color': 'white',
                                                'backgroundColor': '#222',  # Darker gray to differentiate
                                                #'border': '1px solid #444'
                                            },
                                            page_size=25,  # Shows 25 rows per page
                                            filter_action="native",  # Enables filtering/search
                                            sort_action="native",  # Enables sorting
                                        )
                                    ]
                                )
                            ],
                            style={"display": "flex", "justify-content": "center"}
                            )
                    ]
                )
            ],
            fluid=True,
        )
    ],
    style={"padding": "12px"}, className="my-4"  # Margin top and bottom for the row
)

# Define the app layout
app.layout = dbc.Container(
    fluid=True,
    children=[
    dbc.Row([
        dbc.Col(sidebar, width=2, style={"padding-left": "0"}),  # Left column for navigation bar and filters
        dbc.Col(content, width=10, style={"padding-right": "0"}),  # Right column for visuals
    ], justify='center')
],
style={"padding":"0"})

# callbacks

@app.callback(
    Output('sentiment-bar-chart', 'figure', allow_duplicate=True),
    [Input('dropdown-model', 'value')],
    prevent_initial_call=True
)
def update_sentiment_chart(selected_model):
    # Filter data based on selected model
    model_score_map = pd.DataFrame(flattened_metrics)
    try:
        selected_columns = column_groups[selected_model]
    except KeyError:
        selected_columns = group_vals
    df_melted = avg_sent_scores.melt(value_vars=selected_columns, var_name="Sentiment Metric", value_name="Score") \
         .merge(model_score_map, on='Sentiment Metric')
    if selected_model == "All":
        filtered_df = df_melted  # Show all models
        color_col = "Model"  # Color by Model when "All" is selected
    else:
        filtered_df = df_melted[df_melted["Model"] == selected_model]  # Filter specific model
        color_col = "Model"  # Keep color by Model even when filtering

    # Create the bar chart
    fig = px.bar(
        filtered_df,
        x="Sentiment Metric",
        y="Score",
        color=color_col,  # Keep the color logic consistent
        title="Average Sentiment Score by Model",
        text_auto=True
    )

    # Improve visualization
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    fig.update_layout(yaxis=dict(title="Average Score"), xaxis=dict(title=""), showlegend=True)

    return fig



@app.callback(

    Output("sentiment-bar-chart", "figure"),
    Output("loc-rating", "children"),
    Output("rating-title", "children"),
    Output("pos-loc-rating", "children"),
    Output("pos-rating-title", "children"),
    Output("neu-loc-rating", "children"),
    Output("neu-rating-title", "children"),
    Output("neg-loc-rating", "children"),
    Output("neg-rating-title", "children"),
    Output('google-button', 'href'),
    Output('google-button', 'style'),

    Input("sushi-map", "clickData")
)

def update_detail_graph(click_data):

    if click_data is None: # Handle case where no point is clicked

        return  update_chart_glo(group_vals, flattened_metrics, avg_sent_scores), presentation_avg_rating, default_rating_title, presentation_pos_avg_rating, agg_positive_rating_title, presentation_neu_avg_rating, agg_neutral_rating_title, presentation_neg_avg_rating, agg_negative_rating_title, default_google_href, default_google_style

    selected_lat = click_data["points"][0]["lat"]

    selected_lon = click_data["points"][0]["lon"]

    selected_add = click_data["points"][0]['customdata'][1]

    selected_name = click_data["points"][0]["hovertext"].replace("'", "\\'")

    # Filter data based on selected coordinates (implement your filtering logic here)

    restaurant = v_sentiment_metrics_by_loc.query(F"DISPLAYNAME_TEXT == '{selected_name}' & LATITUDE == {selected_lat} & LONGITUDE == {selected_lon}")
    
    # rating 
    restaurant_rating = "### " + str(restaurant['OVERALL_RATING'].values[0]) + " ⭐"
    restaurant_rating_title = f"{selected_name} Overall Rating"

    # pos rating
    pos_restaurant_rating = "### " + str(round(restaurant['AVG_BASE_POS'].values[0], 2)) + " 😊"
    pos_restaurant_rating_title = f"{selected_name} Avg Pos. Sentiment Score"

    # neu rating
    neu_restaurant_rating = "### " + str(round(restaurant['AVG_BASE_NEU'].values[0], 2)) + " 😐"
    neu_restaurant_rating_title = f"{selected_name} Avg Neu. Sentiment Score"

    # neg rating
    neg_restaurant_rating = "### " + str(round(restaurant['AVG_BASE_NEG'].values[0], 2)) + " 😞"
    neg_restaurant_rating_title = f"{selected_name} Avg Neg. Sentiment Score"

    # prep bar chart

    filter_cols = [col for col in restaurant.columns if 'AVG' in col or col == 'DISPLAYNAME_TEXT']

    filtered_df = restaurant[filter_cols]
    value_vars = [i for i in filter_cols if i != 'DISPLAYNAME_TEXT']
    model_score_map = pd.DataFrame(flattened_metrics)

    melt_df = filtered_df.melt(value_vars=value_vars, var_name="Sentiment Metric", value_name="Score") \
        .merge(model_score_map, on='Sentiment Metric')

    # Create bar chart
    fig = px.bar(melt_df, x="Sentiment Metric", y="Score", 
                 text_auto=True, title=f"Average {selected_name} Model Scores", color="Model")

    # Improve visualization
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    fig.update_layout(yaxis=dict(title="Average Score"), xaxis=dict(title="Metric"), showlegend=True)

    # serve google button
        # Generate the Google Maps URL

    google_loc_name = '+'.join(selected_name.split(' '))
    google_maps_url = f"https://www.google.com/maps/search/?q={google_loc_name}+{selected_add}"

    google_style = {"display": "inline-block"}

    return fig, restaurant_rating, restaurant_rating_title, pos_restaurant_rating, pos_restaurant_rating_title, neu_restaurant_rating, neu_restaurant_rating_title, neg_restaurant_rating, neg_restaurant_rating_title, google_maps_url, google_style

# filters for updating map
@app.callback(
    Output("sushi-map", "figure", allow_duplicate=True),

    Input('dropdown-borough', 'value'),
    Input('dropdown-price-level', 'value'),
    Input('dropdown-restaurant-type', 'value'),
    Input('slider-user-rating-count', 'value'),
    Input('slider-overall-rating', 'value'),
    Input('slider-weighted_conf', 'value'),
    Input('dropdown-reservable', 'value'),
    Input('dropdown-alcohol', 'value'),
    Input('dropdown-wc-accessible-entrance', 'value'),
    prevent_initial_call=True
)

def update_map(
    selected_boroughs, 
    selected_price_levels, 
    selected_restaurant_type, 
    selected_rating_count, 
    selected_overall_rating,
    selected_weighted_conf,
    selected_reservable,
    selected_alcohol,
    selected_wc_access
):

    # Start with the full dataset
    filtered_data = v_sentiment_metrics_by_loc.copy()
    filtered_dim = sushi_places.copy()[['ID', 'RESERVABLE', 'SERVESBEER', 'SERVESWINE', 'WHEELCHAIRACCESSIBLEENTRANCE']]

    # Apply borough filter
    if selected_boroughs and "All" not in selected_boroughs:
        filtered_data = filtered_data[filtered_data['BOROUGH'].isin(selected_boroughs)]

    # Apply price level filter
    if selected_price_levels and "All" not in selected_price_levels:
        # Handle "Null" option
        if "Null" in selected_price_levels:
            # Include rows where PRICE_LEVEL is null
            null_filter = filtered_data['PRICELEVEL'].isna()
            # Include rows with selected price levels (excluding "Null")
            price_level_filter = filtered_data['PRICELEVEL'].isin(
                [level for level in selected_price_levels if level != "Null"]
            )
            # Combine filters
            filtered_data = filtered_data[null_filter | price_level_filter]
        else:
            # Exclude nulls and filter by selected price levels
            filtered_data = filtered_data[filtered_data['PRICELEVEL'].isin(selected_price_levels)]

    # Apply Primary Type filter
    if selected_restaurant_type and "All" not in selected_restaurant_type:
        filtered_data = filtered_data[filtered_data['PRIMARY_TYPE'].isin(selected_restaurant_type)]

    # Apply USERRATINGCOUNT filter using the range slider
    if selected_rating_count:
        lower_bound, upper_bound = selected_rating_count
        filtered_data = filtered_data[
            (filtered_data['USERRATINGCOUNT'] >= lower_bound) &
            (filtered_data['USERRATINGCOUNT'] <= upper_bound)
        ]

    # Apply Overall rating filter using the range slider
    if selected_overall_rating:
        lower_bound, upper_bound = selected_overall_rating
        filtered_data = filtered_data[
            (filtered_data['OVERALL_RATING'] >= lower_bound) &
            (filtered_data['OVERALL_RATING'] <= upper_bound)
        ]

    # Apply Weighted Confidence filter using the range slider
    if selected_weighted_conf:
        lower_bound, upper_bound = selected_weighted_conf
        filtered_data = filtered_data[
            (filtered_data['WEIGHTED_CONFIDENCE_SCORE'] >= lower_bound) &
            (filtered_data['WEIGHTED_CONFIDENCE_SCORE'] <= upper_bound)
        ]

    # Apply reservable filter
    if selected_reservable:
        if selected_reservable == "All":
            pass  # Keep all values if "All" is selected
        elif selected_reservable == "Unknown":
            # Filter for rows where 'RESERVABLE' is NaN
            filtered_dim = filtered_dim[filtered_dim['RESERVABLE'].isna()]
        else:
            # Filter for True or False as usual
            reservable_value = selected_reservable == "True"
            filtered_dim = filtered_dim[filtered_dim['RESERVABLE'] == reservable_value]

    # Apply alcohol filter (checking both beer and wine)
    if selected_alcohol:
        if selected_alcohol == "All":
            pass  # Keep all values if "All" is selected
        elif selected_alcohol == "Unknown":
            # Filter for rows where either 'SERVESBEER' or 'SERVESWINE' is NaN
            filtered_dim = filtered_dim[filtered_dim['SERVESBEER'].isna() | filtered_dim['SERVESWINE'].isna()]
        else:
            alcohol_value = selected_alcohol == "True"
            filtered_dim = filtered_dim[(filtered_dim['SERVESBEER'] == alcohol_value) | (filtered_dim['SERVESWINE'] == alcohol_value)]

    # Apply wc access filter
    if selected_wc_access:
        if selected_wc_access == "All":
            pass  # Keep all values if "All" is selected
        elif selected_wc_access == "Unknown":
            # Filter for rows where 'WHEELCHAIRACCESSIBLEENTRANCE' is NaN
            filtered_dim = filtered_dim[filtered_dim['WHEELCHAIRACCESSIBLEENTRANCE'].isna()]
        else:
            wc_access_value = selected_wc_access == "True"
            filtered_dim = filtered_dim[filtered_dim['WHEELCHAIRACCESSIBLEENTRANCE'] == wc_access_value]


    # Add more filters here as needed
    # Example:
    # if selected_another_filter and "All" not in selected_another_filter:
    #     filtered_data = filtered_data[filtered_data['ANOTHER_COLUMN'].isin(selected_another_filter)]

    data_selections = filtered_data.merge(filtered_dim, on='ID', how='inner')

    # Create the map figure
    map_fig = px.scatter_map(
        data_selections,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="DISPLAYNAME_TEXT",  # Shows location name on hover
        custom_data = 'PLACE_ID',
        zoom=11,  # Zoom level (adjust for best view)
        height=550,
        width=1200,
        hover_data=['ADDRESS', 'USERRATINGCOUNT']
    )

    # Use OpenStreetMap (OSM) as the base layer
    map_fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0, "t":0, "l":0, "b":0}
    )

    return map_fig

# wordcloud callback

@app.callback(
    Output('wordcloud-image', 'src', allow_duplicate=True),  # Update the Wordcloud image
    Output('wordcloud-title', 'children', allow_duplicate=True),
    Input("sushi-map", "clickData"),  # Input that triggers the update
    prevent_initial_call='initial_duplicate'
)
def update_wordcloud(click_data):

    if not click_data:
        # default
        get_wordcloud = wordcloud_viz(sushi_sentiment, gen_whole_wordcloud=True)

        return f"data:image/png;base64,{get_wordcloud}", default_wc_title
    
    else:
        # user select
        selected_lat = click_data["points"][0]["lat"]

        selected_lon = click_data["points"][0]["lon"]

        selected_name = click_data["points"][0]["hovertext"].replace("'", "\\'")

        # Filter data based on selected coordinates (implement your filtering logic here)

        restaurant = v_sentiment_metrics_by_loc.query(F"DISPLAYNAME_TEXT == '{selected_name}' & LATITUDE == {selected_lat} & LONGITUDE == {selected_lon}") \
        .merge(v_review_info[['PLACE_ID', 'WORD_TOKENS']], on='PLACE_ID')

        # Generate Wordcloud based on input_value or other logic
        get_wordcloud = wordcloud_viz(restaurant)
        
        return f"data:image/png;base64,{get_wordcloud}", f'Most Frequent Words - {selected_name}'

# callback for highlighting selected locations

@app.callback(
    Output('sushi-map', 'figure', allow_duplicate=True),  # Update the map figure
    Input('sushi-map', 'clickData'),  # Input from the map's click data.
    prevent_initial_call=True
)
def highlight_selected_location(click_data):
    # Create a copy of the initial map figure
    updated_fig = default_map_fig

    if not click_data:
        # If no location is clicked, return the original map figure
        return updated_fig
    else:
        # Extract custom data from click_data
        try:
            custom_data = click_data['points'][0]['customdata']
        except KeyError:
            pass
        selected_place = custom_data[0]  # First item in customdata (PLACE_ID)
        
        # Fetch the restaurant data for the selected location
        restaurant = v_sentiment_metrics_by_loc[v_sentiment_metrics_by_loc['PLACE_ID'] == selected_place]
        
        # Update the selected location trace
        hover_text = f"{restaurant['DISPLAYNAME_TEXT'].values[0]}<br>{restaurant['ADDRESS'].values[0]}" # create name & address text
        updated_fig.data[-1].lat = [restaurant['LATITUDE'].values[0]]  # Update latitude
        updated_fig.data[-1].lon = [restaurant['LONGITUDE'].values[0]]  # Update longitude
        updated_fig.data[-1].hovertext = [hover_text]  # Update hover text
        updated_fig.data[-1].marker = dict(color='red', size=15)  # Highlight the selected location
        return updated_fig

# callback for resetting filters

@app.callback(
    [
        Output('dropdown-borough', 'value'),  # Reset borough dropdown
        Output('dropdown-price-level', 'value'),  # Reset price level dropdown
        Output('slider-user-rating-count', 'value'),  # Reset rating count slider
        Output('slider-overall-rating', 'value'),
        Output('slider-weighted_conf', 'value'),
        Output('sushi-map', 'figure'),  # Reset map figure
        Output('wordcloud-image', 'src'),  # Reset Wordcloud
        Output('dropdown-restaurant-type', 'value'),
        Output('loc-rating', 'children', allow_duplicate=True),
        Output('rating-title', 'children', allow_duplicate=True),
        Output('wordcloud-title', 'children'),
        Output('dropdown-model', 'value'),
        Output('barchart-title', 'children'),
        Output("pos-loc-rating", "children", allow_duplicate=True),
        Output("pos-rating-title", "children", allow_duplicate=True),
        Output("neu-loc-rating", "children", allow_duplicate=True),
        Output("neu-rating-title", "children", allow_duplicate=True),
        Output("neg-loc-rating", "children", allow_duplicate=True),
        Output("neg-rating-title", "children", allow_duplicate=True),
        Output('dropdown-reservable', 'value'),
        Output('dropdown-alcohol', 'value'),
        Output('dropdown-wc-accessible-entrance', 'value'),
        Output('google-button', 'href', allow_duplicate=True),
        Output('google-button', 'style', allow_duplicate=True),
    ],
    [Input('reset-button', 'n_clicks')],  # Triggered by the reset button
    prevent_initial_call=True  # Prevent the callback from firing on initial load
)
def reset_filters(n_clicks):
    # Return default values for all inputs and outputs
    return (
        default_borough,  # Reset borough dropdown
        default_price_level,  # Reset price level dropdown
        [rating_count_min, rating_count_max],  # Reset rating count slider
        [overall_rating_min, overall_rating_max],
        [weighted_conf_min, weighted_conf_max],
        default_map_fig,  # Reset map figure
        f"data:image/png;base64,{default_wordcloud_image}",  # Reset Wordcloud
        default_restaurant_type,
        presentation_avg_rating,
        default_rating_title,
        default_wc_title,
        default_bar_chart,
        default_barchart_title,
        presentation_pos_avg_rating,
        agg_positive_rating_title, 
        presentation_neu_avg_rating, 
        agg_neutral_rating_title, 
        presentation_neg_avg_rating, 
        agg_negative_rating_title,
        None,
        None,
        None,
        default_google_href, 
        default_google_style
    )

# callback for 3d scatter & table interactions
@app.callback(
    Output('review-table', 'data'),
    [Input('sentiment-scatter-3d', 'clickData'),
     Input('table-reset-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_or_reset_table(click_data, n_clicks):
    # Get the context of the triggered input
    ctx = callback_context

    # Check which input triggered the callback
    if not ctx.triggered:
        return visible_review_data_table.to_dict('records')
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # If the reset button was clicked
    if triggered_id == 'table-reset-button':
        return visible_review_data_table.to_dict('records')
    
    # If a data point was clicked in the 3D scatter plot
    if triggered_id == 'sentiment-scatter-3d' and click_data:
        point_index = click_data['points'][0]['pointNumber']
        clicked_row = visible_review_data_table.iloc[point_index]
        new_df = pd.DataFrame([clicked_row])
        return new_df.to_dict('records')
    
    # Fallback: return the original data
    return visible_review_data_table.to_dict('records')


# Callback to update the 3D scatter plot w dropdown
@app.callback(
    Output("sentiment-scatter-3d", "figure"),
    Input("scatter-column-selector", "value")
)
def update_3d_scatter(selected_option):
    if not selected_option:
        selected_option = 'NLTK VADER'

    if selected_option == 'Special':
        title = 'Subjectivity, Polarity, Compound Score'
    else:
        title = 'Positive, Negative, Neutral'

    cols = sentiment_model_dict[selected_option]  # Get selected columns
    fig = px.scatter_3d(sushi_sentiment, x=cols[0], y=cols[1], z=cols[2], 
                        color=cols[0], 
                        title=f"3D Scatter Plot - {title}",
                        height=900,
                        width=1200,
                        custom_data=['REVIEW_ID']
                        )
    
    # Expanding the grid by modifying axis properties


    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[sushi_sentiment[cols[0]].min() - 0.5, sushi_sentiment[cols[0]].max() + 0.5], showgrid=True),
            yaxis=dict(range=[sushi_sentiment[cols[1]].min() - 0.5, sushi_sentiment[cols[1]].max() + 0.5], showgrid=True),
            zaxis=dict(range=[sushi_sentiment[cols[2]].min() - 0.5, sushi_sentiment[cols[2]].max() + 0.5], showgrid=True),
            aspectmode="cube",  # Keeps all axes equally scaled
        ),
        legend=dict(
        x=1.3,  # Moves legend outside the plot area
        y=1,  # Keeps it aligned at the top
        bgcolor="rgba(255,255,255,0.7)",  # Optional: Adds a semi-transparent background
    )
    )

    return fig

# # callback for filter changes via map clicks - filter store
# @app.callback(
#     Output('filter-store', 'data'),  # Update the filter store
#     [
#         Input('dropdown-borough', 'value'),  # Borough filter
#         Input('dropdown-price-level', 'value'),  # Price level filter
#         Input('dropdown-restaurant-type', 'value'),
#         Input('slider-user-rating-count', 'value'),  # Rating count slider
#         Input('slider-overall-rating', 'value')
#     ]
# )
# def update_filter_store(selected_boroughs, selected_price_levels, selected_restaurant_type, selected_rating_count, selected_overall_rating):
#     # Store the current filter values in a dictionary
#     filter_data = {
#         'boroughs': selected_boroughs,
#         'price_levels': selected_price_levels,
#         'primary_type': selected_restaurant_type,
#         'rating_count': selected_rating_count,
#         'overall_rating': selected_overall_rating
#     }
#     return filter_data



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 80))
    app.run(debug=False, port=port, host='0.0.0.0')







