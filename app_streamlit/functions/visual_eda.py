import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def eda_boxplots(df_demanda, df_generation, df_co2):
    df_demanda=df_demanda[['fecha', 'valor_demanda_MW']]

    df_generation = df_generation[(df_generation['energia'] == 'Generación total') | (df_generation['tipo_tecnología'] == 'Generación total')]
    df_generation.reset_index()
    df_generation=df_generation[['fecha', 'valor_generacion_MW']]

    df_co2 = df_co2[~(df_co2['energia'].isin(['tCO2 eq./MWh', 'Total tCO2 eq.']))]
    df_co2 = df_co2.groupby('fecha', as_index=False)['valor'].sum()

    fig_box_demanda=px.box(df_demanda, x='valor_demanda_MW', title='Valores demanda energética', labels={'valor_demanda_MW': 'Demanda (MW)'})
    fig_box_generation=px.box(df_generation, x='valor_generacion_MW', title='Valores generación energética', labels={'valor_generacion_MW': 'Generacion (MW)'})
    fig_box_co2=px.box(df_co2, x='valor', title='Valores emisiones de CO2', labels={'valor': 'Valores (T/CO2)'})

    st.plotly_chart(fig_box_demanda)
    st.plotly_chart(fig_box_generation)
    st.plotly_chart(fig_box_co2)

def eda_relations(df_demanda, df_generation, df_co2):
    df_demanda=df_demanda[['fecha', 'valor_demanda_MW']]

    df_generation = df_generation[(df_generation['energia'] == 'Generación total') | (df_generation['tipo_tecnología'] == 'Generación total')]
    df_generation.reset_index()
    df_generation=df_generation[['fecha', 'valor_generacion_MW']]

    df_co2 = df_co2[~(df_co2['energia'].isin(['tCO2 eq./MWh', 'Total tCO2 eq.']))]
    df_co2 = df_co2.groupby('fecha', as_index=False)['valor'].sum()

    df_relations = pd.merge(df_demanda, df_generation, on='fecha', how='inner')
    df_relations = pd.merge(df_relations, df_co2, on='fecha', how='inner')

    fig_demanda_generacion=px.scatter(df_relations, x='valor_demanda_MW', y='valor_generacion_MW', labels={'valor_demanda_MW':'Demanda(MW)', 'valor_generacion_MW': 'Generacion(MW)'})
    fig_demanda_co2=px.scatter(df_relations, x='valor_demanda_MW', y='valor', labels={'valor_demanda_MW':'Demanda(MW)', 'valor': 'Valores(T/CO2)'})
    fig_generacion_co2=px.scatter(df_relations, x='valor_generacion_MW', y='valor', labels={'valor_generacion_MW':'Generacion(MW)', 'valor': 'Valores(T/CO2)'})

    st.plotly_chart(fig_demanda_generacion)
    st.plotly_chart(fig_demanda_co2)
    st.plotly_chart(fig_generacion_co2)


def eda_demanda_ano_2020_z(df_demanda):

    df_demanda['fecha']=pd.to_datetime(df_demanda['fecha'])
    df_demanda['year']=df_demanda['fecha'].dt.year
    df_demanda_2020=df_demanda[df_demanda['year']==2020]

    z=3

    mean=np.mean(df_demanda_2020['valor_demanda_MW'])
    std=np.std(df_demanda_2020['valor_demanda_MW'])

    lim_sup=mean+z*std
    lim_inf=mean-z*std

    fig_2020=px.histogram(df_demanda_2020, x='valor_demanda_MW')
    fig_2020.add_vline(x=lim_sup, annotation_text='limite superior', line_color='green')
    fig_2020.add_vline(x=lim_inf,annotation_text='limite inferior', line_color='red')
    st.plotly_chart(fig_2020)

def eda_demanda_ano_2020_t(df_demanda):

    df_demanda['fecha']=pd.to_datetime(df_demanda['fecha'])
    df_demanda['year']=df_demanda['fecha'].dt.year
    df_demanda_2020=df_demanda[df_demanda['year']==2020]

    h=1.5

    q1=np.quantile(df_demanda_2020['valor_demanda_MW'], 0.25)
    q3=np.quantile(df_demanda_2020['valor_demanda_MW'], 0.75)

    IQR=q3-q1

    lim_sup_dem=q3+h*IQR
    lim_inf_dem=q1-h*IQR

    fig=px.histogram(df_demanda_2020, x='valor_demanda_MW')
    fig.add_vline(x=lim_sup_dem, annotation_text='limite superior', line_color='green')
    fig.add_vline(x=lim_inf_dem,annotation_text='limite inferior', line_color='red')
    st.plotly_chart(fig)


def eda_anos_atipicos(df_demanda):
    df_demanda_year=df_demanda.groupby('year', as_index=False)['valor_demanda_MW'].sum()
    df_demanda_year=df_demanda_year[~(df_demanda_year['year']==2024)]

    #Z-Score

    z=3

    mean=np.mean(df_demanda_year['valor_demanda_MW'])
    std=np.std(df_demanda_year['valor_demanda_MW'])

    lim_sup=mean+z*std
    lim_inf=mean-z*std

    #Tukey

    h=1.5

    q1=np.quantile(df_demanda_year['valor_demanda_MW'], 0.25)
    q3=np.quantile(df_demanda_year['valor_demanda_MW'], 0.75)

    IQR=q3-q1

    lim_sup_dem=q3+h*IQR
    lim_inf_dem=q1-h*IQR

    fig_z=px.histogram(df_demanda_year, x='year', y='valor_demanda_MW', nbins=14)
    fig_z.add_hline(y=lim_sup, annotation_text='limite superior', line_color='green')
    fig_z.add_hline(y=lim_inf,annotation_text='limite inferior', line_color='red')
    st.plotly_chart(fig_z)

    fig_t=px.histogram(df_demanda_year, x='year', y='valor_demanda_MW', nbins=14)
    fig_t.add_hline(y=lim_sup_dem, annotation_text='limite superior', line_color='green')
    fig_t.add_hline(y=lim_inf_dem,annotation_text='limite inferior', line_color='red')
    st.plotly_chart(fig_t)

def eda_anos_atipicos_dbscan(df_demanda):
    df_demanda_year=df_demanda.groupby('year', as_index=False)['valor_demanda_MW'].sum()
    df_demanda_year=df_demanda_year[~(df_demanda_year['year']==2024)]
    dbscan = DBSCAN(eps=4e6, min_samples=2)
    df_demanda_year['cat'] = dbscan.fit_predict(df_demanda_year[['valor_demanda_MW']])

    df_demanda_year['color'] = df_demanda_year['cat'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')

    # Visualizar con Plotly
    fig = px.scatter(df_demanda_year, x='year', y='valor_demanda_MW', color='color',
                 title="Detección de Outliers en la Demanda Anual de Energía",
                 labels={'valor_demanda_MW': 'Demanda (MW)', 'year': 'Año'})
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig)