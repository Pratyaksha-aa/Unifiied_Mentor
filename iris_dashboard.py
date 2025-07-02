import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Load dataset
df = pd.read_csv("Iris.csv")

# Initialize app
app = Dash(__name__)
server = app.server  # for deployment

# App layout
app.layout = html.Div([
    html.H1("🌸 Iris Dataset Dashboard", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id='species-dropdown',
        options=[{'label': sp, 'value': sp} for sp in df['Species'].unique()],
        value='Iris-setosa',
        clearable=False,
        style={'width': '50%', 'margin': 'auto'}
    ),

    dcc.Graph(id='scatter-plot'),

    html.Div([
        dcc.Graph(id='box-sepal-length'),
        dcc.Graph(id='box-petal-length'),
        dcc.Graph(id='box-sepal-width'),
        dcc.Graph(id='box-petal-width')
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px', 'padding': '20px'})
])

# Callbacks
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('species-dropdown', 'value')
)
def update_scatter(selected_species):
    filtered_df = df[df['Species'] == selected_species]
    fig = px.scatter(
        filtered_df,
        x='SepalLengthCm',
        y='PetalLengthCm',
        size='PetalWidthCm',
        color='SepalWidthCm',
        title=f"{selected_species} - Sepal vs Petal",
        labels={'SepalLengthCm': 'Sepal Length (cm)', 'PetalLengthCm': 'Petal Length (cm)'},
        template='plotly_dark'
    )
    return fig

@app.callback(
    Output('box-sepal-length', 'figure'),
    Output('box-petal-length', 'figure'),
    Output('box-sepal-width', 'figure'),
    Output('box-petal-width', 'figure'),
    Input('species-dropdown', 'value')
)
def update_boxplots(selected_species):
    filtered_df = df[df['Species'] == selected_species]
    fig1 = px.box(filtered_df, y="SepalLengthCm", title="Sepal Length")
    fig2 = px.box(filtered_df, y="PetalLengthCm", title="Petal Length")
    fig3 = px.box(filtered_df, y="SepalWidthCm", title="Sepal Width")
    fig4 = px.box(filtered_df, y="PetalWidthCm", title="Petal Width")
    return fig1, fig2, fig3, fig4

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

