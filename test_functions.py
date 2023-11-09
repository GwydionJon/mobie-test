def dash_plot(ds):
    from dash import Dash, dcc, html, Input, Output
    import plotly.express as px

    app = Dash("test")


    app.layout = html.Div([
        html.H4('Cell Stacks'),
        dcc.Graph(id="graph"),
        html.P("Slice:"),
        dcc.Slider(
            id='slices',
            min=0,
            max=ds.shape[2],
            step=1,
            value=1
        )
    ])


    @app.callback(
    Output("graph", "figure"), 
    Input("slices", "value"))
    def filter_heatmap(slice):
        ds_slice = ds[:,:,slice] # replace with your own data source
        fig = px.imshow(ds_slice)
        return fig


    app.run_server(debug=True, port = 8083)


def plot_3d(ds):
    import plotly.graph_objects as go
    import numpy as np
    shape_len = ds.shape[0]
    X, Y, Z = np.mgrid[-shape_len:shape_len:38j, -shape_len:shape_len:38j, -shape_len:shape_len:38j]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=ds[:].flatten(),
        isomin=1,
        isomax=450,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=10, # needs to be a large number for good volume rendering
        ))
    fig.show()