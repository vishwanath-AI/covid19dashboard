import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from dash.dependencies import Input, Output
import operator
import functools
import plotly.express as px
from datetime import datetime, timedelta

def to_unix_time(dt):
    epoch =  datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Coronavirus Infections'
server = app.server


# server = Flask(__name__, static_folder='static')
# app =  Dash(server=server)

# @server.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(server.root_path, 'static'),
#                                'favicon.ico', mimetype='image/vnd.microsoft.icon')

url_death="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
url_recovered="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"

url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"

df_death=pd.read_csv(url_death)
df_recov=pd.read_csv(url_recovered)
df=pd.read_csv(url)
#df = pd.read_csv('data/coronavirus.csv', index_col=False, header=0)
df.rename(columns = {'Province/State':'State'}, inplace = True)
df_H=df[df['State']=='Hubei']
df_death.rename(columns = {'Province/State':'State'}, inplace = True)
df_H_d=df_death[df_death['State']=='Hubei']
df_recov.rename(columns = {'Province/State':'State'}, inplace = True)
df_H_r=df_recov[df_recov['State']=='Hubei']


df_U=df[df['Country/Region']=='US']
df_U.loc['date_infections'] = df_U.sum()


df_U_d=df_death[df_death['Country/Region']=='US']
df_U_d.loc['date_infections'] = df_U_d.sum()

df_U_r=df_recov[df_recov['Country/Region']=='US']
df_U_r.loc['date_infections'] = df_U_r.sum()


cols_to_sum = [col for col in df.columns if '/20' in col] 
#d = datetime. datetime. today()

df['total_cases'] = df[datetime.strftime(datetime.now() - timedelta(1), '%-m/%-d/%y')]
#df['total_cases'] = df[cols_to_sum].sum(axis=1)  # assigned to a column

scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\
[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\
[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]


countries=df['Country/Region'].unique().tolist()[:2]

trace=[]
colorway = ['#f3cec9', '#D2691E', '#FF6347', '#BC8F8F', '#9ACD32', '#006400', '#182844','#8B0000','#FFD700','#00FF00','#808000','#4169E1','#BA55D3','#708090','#D2B48C','#4682B4','#F5DEB3','#FFE4E1','#DB7093','#DA70D6','#B0E0E6','#00FA9A','#FF7F50','#F08080','#BDB76B']
#layout = go.Layout(colorway=colorway)
i=0
for country in countries:
    a=df[df['Country/Region']==country]


    
    tracex = go.Bar(
    y=(a['State'].tolist()),
    x=(a['total_cases'].tolist()),
    name=country,
    
    )
    trace.append(tracex)
    #i=i+1
  

    
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )



df_1=df.groupby(['Country/Region']).sum()
df_1.reset_index(inplace=True)
df1 = df_1[(df_1['total_cases']>30) & (df_1['total_cases']<60000)]
df2 = df_1[(df_1['total_cases']<2)]
df3=df[df['Country/Region']=='Mainland China']
df3.reset_index(inplace=True)


x = pd.date_range(start = "2020-01-22", end = datetime.now(), freq = "D")
x = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in x]

y_index = pd.date_range(start = datetime.strftime(datetime(2020, 1, 22),'%-m/%-d/%y'), end = datetime.strftime(datetime.now()- timedelta(1),'%-m/%-d/%y'), freq = "D")
y_index = [datetime.strftime(date,'%-m/%-d/%y') for date in y_index]


y=[df_H[str(d)].item() for d in y_index]
y_d=[df_H_d[str(d)].item() for d in y_index]
y_r=[df_H_r[str(d)].item() for d in y_index]



y1=[df_U.sum()[str(d)] for d in y_index]
y1_d=[df_U_d.sum()[str(d)] for d in y_index]
y1_r=[df_U_r.sum()[str(d)] for d in y_index]


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}
def gen_traces(selected_name, trace_name, trace_color):  
    if selected_name is None:
        return trace
    # else:

    #     return trace_mgr_dict[selected_name]


# Create a Dash layout
app.layout = html.Div([
    html.H1(children='Corono Virus Infected Cases Globally'),

    html.Div(children='''
        A Visualization of the Infection Increase from 01/22/2020 till Today
        '''),
    html.Div(children='''
    '''),

    dcc.Graph(
        id='example-map',
        
        figure= go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = df['Long'],
        lat = df['Lat'],
        text = df['State'].astype(str),
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Bluered',#'Blues',
            cmin = 0,
            color = df['total_cases'],
            cmax = df['total_cases'].max(),
            colorbar_title="Total Infected Cases"
        )),layout=dict(height=800, width=1600, title="Total Infected Cases of Corona Virus from 01/22/2020 till Today"))),

    html.Br(),

    # html.Div(children='''
    #     The data used here is from https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv
    # '''),
    html.Div(children='''
    '''),
    html.Div(children='''
        Click on the below tabs to see plots for Mainland China and outside China
        '''),

    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Mainland China', id='tab1', value='tab-1-example', style=tab_style, 
        selected_style=tab_selected_style),


        dcc.Tab(label='All Other Countries', id='tab2',value='tab-2-example', style=tab_style, 
        selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-example')
])



@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])

 
def render_content(tab):
    
    if tab == 'tab-1-example':
        return html.Div(children=[


    html.Br(),
    html.Br(),


    html.Div(children='''
    '''),

    dcc.Graph(
        id='example-scatter',

        figure=go.Figure(data = [go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Confirmed Cases',
        marker_color='rgba(152, 0, 0, .8)'),go.Scatter(
        x=x,
        y=y_d,
        name='Deaths Caused',
        mode='lines+markers',
        marker_color='rgb(231, 99, 250)'),
        go.Scatter(
        x=x,
        y=y_r,
        name='Recovered Cases',
        mode='lines+markers',
        marker_color='rgb(17, 157, 255)')],

        layout = go.Layout(height=600,xaxis = dict(
                   range = [to_unix_time(datetime(2020, 1, 21)),
                            to_unix_time(datetime.now())],
                            title="Trend of Infected Cases of Corona Virus from 01/22/2020 till Today in Hubei - the Most INFECTED City")
    ))),



    html.Div(children='''
    '''),

        dcc.Graph(
        id='example-graph1',
        
        figure=go.Figure(data=[go.Bar(
            x=df3['State'].tolist(), y=df3['total_cases'].tolist(),
            text=df3['total_cases'].tolist(),
            textposition='auto',
            #layout=go.Layout(barmode='group',title="Coronavirus Infection across Mainland China")
            #title='Coronavirus Infection across Mainland China'
        )],layout=go.Layout(height=600,title="Coronavirus Infections across Mainland China"))

    ),

    html.Br(),
    html.Br(),

    html.Div(children='''
    '''),

    
        dcc.Graph(
                id='example-graph3',
                figure={
                    'data': [                          
                        go.Pie(                                        
                            labels=list(df3['State']),
                            values=list(df3['total_cases']),  
                            hoverinfo='label+value+percent'#, textinfo='label+percent'                                     
                            )                              
                        ],
                    'layout':{
                            'showlegend':True,
                            'title':'All Infected States in Mainland China',
                            'height':800, 
                            'width':1500,
                            'annotations':[
                                # ...                                
                            ]
                    }  
                }
            )


])
    
    elif tab == 'tab-2-example':
        return html.Div(children=[
            html.Div(children='''
        ''',style={'padding': 10}),


        html.Div(children='''
        '''),

        dcc.Graph(
        id='example-graph1',
        
        figure=go.Figure(data=[go.Bar(
            x=df1['Country/Region'].tolist(), y=df1['total_cases'].tolist(),
            text=df1['total_cases'].tolist(),
            textposition='auto',
            
        )],layout=go.Layout(height=600,title="Coronavirus Infections across Countries outside Mainland China"))
        #figure.update_layout(title_text="Coronavirus Infection across Countries outside Mainland China")
        ),

        html.Div(children='''
    '''),

        dcc.Graph(
            id='example-scatter',
            figure=go.Figure(data = [go.Scatter(
            x=x,
            y=y1,
            name='Confirmed Cases',
            mode='lines+markers',
            marker_color='rgba(152, 0, 0, .8)'), go.Scatter(
            x=x,
            y=y1_d,
            name='Death Caused',
            mode='lines+markers',
            marker_color='rgb(231, 99, 250)'),go.Scatter(
            x=x,
            y=y1_r,
            name='Recovered Cases',
            mode='lines+markers',
            marker_color='rgb(17, 157, 255)')],

            layout = go.Layout(title="Trend of Infected Cases of Corona Virus from 01/22/2020 till Today in United States",height=600,xaxis = dict(
                    range = [to_unix_time(datetime(2020, 1, 21)),
                                to_unix_time(datetime.now())]
                                )
        ))),

        
        
            html.Div(children='''
            '''),

            
            dcc.Graph(
                id='example-graph3',
                figure={
                    'data': [                          
                        go.Pie(                                        
                            labels=list(df1['Country/Region']),
                            values=list(df1['total_cases']),  
                            hoverinfo='label+value+percent'#, textinfo='label+percent'                                     
                            )                              
                        ],
                    'layout':{
                            'showlegend':True,
                            'title':'Highest Infected Countries across the World outside Mainland China',
                            'height':800, 
                            'width':1500,
                            'annotations':[
                                # ...                                
                            ]
                    }  
                }
            ),
            dcc.Graph(
                id='example-graph2',
                figure={
                    'data': [                          
                        go.Pie(                                        
                            labels=list(df2['Country/Region']),
                            values=list(df2['total_cases']),  
                            hoverinfo='label+value+percent'#, textinfo='label+percent'                                     
                            )                              
                        ],
                    'layout':{
                            'showlegend':True,
                            'title':'Least Infected Countries across the World outside Mainland China',
                            'height':800, 
                            'width':1500,
                            'annotations':[
                                # ...                                
                            ]
                    }  
                }
            )
        
    
        
])

 



if __name__ == '__main__':
    app.run_server(debug=True)

