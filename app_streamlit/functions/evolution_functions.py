import pandas as pd
import plotly.express as px
import streamlit as st


def crecimiento_anual_demanda(df_demanda):
    df_demanda['fecha'] = pd.to_datetime(df_demanda['fecha'])
    df_demanda['year'] = df_demanda['fecha'].dt.year

    # Calcular la demanda anual (suma o promedio)
    demanda_anual = df_demanda.groupby('year')['valor_demanda_MW'].sum().reset_index()
    demanda_anual.rename(columns={'valor_demanda_MW': 'total_demanda_anual'}, inplace=True)

    # Calcular el crecimiento porcentual
    demanda_anual['%_crecimiento'] = demanda_anual['total_demanda_anual'].pct_change() * 100

    fig = px.bar(demanda_anual, x='year', y='%_crecimiento', title='Crecimiento (%) de demanda por Año',
                 labels={'%_crecimiento': 'Crecimiento (%)', 'year': 'Año'})

    st.plotly_chart(fig)


def crecimiento_anual_generacion(df_generation):
    df_generation = df_generation[(df_generation['energia'] == 'Generación total') | (df_generation['tipo_tecnología'] == 'Generación total')]
    df_generation.reset_index()
    df_generation = df_generation[['fecha', 'valor_generacion_MW']]

    df_generation['fecha'] = pd.to_datetime(df_generation['fecha'])
    df_generation['year'] = df_generation['fecha'].dt.year

    # Calcular la demanda anual (suma o promedio)
    generacion_anual = df_generation.groupby('year')['valor_generacion_MW'].sum().reset_index()
    generacion_anual.rename(columns={'valor_generacion_MW': 'total_generacion_anual'}, inplace=True)

    # Calcular el crecimiento porcentual
    generacion_anual['%_crecimiento'] = generacion_anual['total_generacion_anual'].pct_change() * 100

    fig = px.bar(generacion_anual, x='year', y='%_crecimiento', title='Crecimiento (%) de generación por año',
                 labels={'%_crecimiento': 'Crecimiento (%)', 'year': 'Año'})

    st.plotly_chart(fig)


def crecimiento_anual_emisiones(df_co2):
    df_co2 = df_co2[~(df_co2['energia'].isin(['tCO2 eq./MWh', 'Total tCO2 eq.']))]
    df_co2 = df_co2.groupby('fecha', as_index=False)['valor'].sum()

    df_co2['fecha'] = pd.to_datetime(df_co2['fecha'])
    df_co2['year'] = df_co2['fecha'].dt.year

    # Calcular la demanda anual (suma o promedio)
    emisiones_anual = df_co2.groupby('year')['valor'].sum().reset_index()
    emisiones_anual.rename(columns={'valor': 'total_emisiones_anual'}, inplace=True)

    # Calcular el crecimiento porcentual
    emisiones_anual['%_crecimiento'] = emisiones_anual['total_emisiones_anual'].pct_change() * 100

    fig = px.bar(emisiones_anual, x='year', y='%_crecimiento', title='Crecimiento (%) de emisiones por año',
                 labels={'%_crecimiento': 'Crecimiento (%)', 'year': 'Año'})

    st.plotly_chart(fig)

def crecimiento_anual_importaciones(df_exchanges):

    df_exchanges = df_exchanges[df_exchanges['tipo_transaccion'] == 'Importación']
    df_exchanges['fecha'] = pd.to_datetime(df_exchanges['fecha'])
    df_exchanges['year'] = df_exchanges['fecha'].dt.year

    importaciones_anuales = df_exchanges.groupby(['year'])['valor_MW'].sum().reset_index()
    importaciones_anuales.rename(columns={'valor_MW': 'total_importaciones_anual'}, inplace=True)

    importaciones_anuales['%_crecimiento'] = importaciones_anuales['total_importaciones_anual'].pct_change() * 100

    fig = px.bar(importaciones_anuales, x='year', y='%_crecimiento',
                     title='Crecimiento (%) de importaciones por año',
                     labels={'%_crecimiento': 'Crecimiento (%)', 'year': 'Año'})

    st.plotly_chart(fig)

def crecimiento_anual_exportaciones(df_exchanges):

    df_exchanges = df_exchanges[df_exchanges['tipo_transaccion'] == 'Exportación']
    df_exchanges['fecha'] = pd.to_datetime(df_exchanges['fecha'])
    df_exchanges['year'] = df_exchanges['fecha'].dt.year

    exportaciones_anuales = df_exchanges.groupby(['year'])['valor_MW'].sum().reset_index()
    exportaciones_anuales.rename(columns={'valor_MW': 'total_exportaciones_anual'}, inplace=True)

    exportaciones_anuales['%_crecimiento'] = exportaciones_anuales['total_exportaciones_anual'].pct_change() * 100

    fig=px.bar(exportaciones_anuales, x='year', y='%_crecimiento',title='Crecimiento (%) de exportaciones por año', labels={'%_crecimiento': 'Crecimiento (%)', 'year': 'Año'})

    st.plotly_chart(fig)

def crecimiento_anual_balance(df_demanda, df_generation):

    df_demanda['fecha'] = pd.to_datetime(df_demanda['fecha'])
    df_generation['fecha'] = pd.to_datetime(df_generation['fecha'])

    df_generation_balance = df_generation[(df_generation['energia'] == 'Generación total') | (df_generation['tipo_tecnología'] == 'Generación total')]
    df_generation_balance = df_generation_balance.drop(columns=['energia', 'tipo_tecnología'])
    df_generation_balance = df_generation_balance.reset_index(drop=True)

    df_saldo_balance = pd.merge(df_demanda, df_generation_balance, on='fecha', how='inner')
    df_saldo_balance = df_saldo_balance[['fecha', 'valor_demanda_MW', 'valor_generacion_MW']]
    df_saldo_balance['balance'] = df_saldo_balance['valor_generacion_MW'] - df_saldo_balance['valor_demanda_MW']
    df_saldo_balance['fecha']=pd.to_datetime(df_saldo_balance['fecha'])
    df_saldo_balance['year']=df_saldo_balance['fecha'].dt.year

    saldo_anual = df_saldo_balance.groupby(['year'])['balance'].sum().reset_index()
    saldo_anual.rename(columns={'balance': 'total_saldo_balance_anual'}, inplace=True)

    saldo_anual['%_crecimiento'] = saldo_anual['total_saldo_balance_anual'].pct_change() * 100

    fig=px.bar(saldo_anual, x='year', y='%_crecimiento',title='Crecimiento (%) del saldo energético por año', labels={'%_crecimiento': 'Crecimiento (%)', 'year': 'Año'})

    st.plotly_chart(fig)

