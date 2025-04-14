import pandas as pd

def calcular_crecimiento_5_anos(df_generation):

    generacion_anual = df_generation.groupby('year')['valor_generacion_MW'].sum().reset_index()

    # Seleccionar los últimos 5 años
    ultimos_5_anios = generacion_anual.tail(5)

    # Calcular el crecimiento porcentual
    valor_inicial = ultimos_5_anios.iloc[0]['valor_generacion_MW']
    valor_reciente = ultimos_5_anios.iloc[-1]['valor_generacion_MW']
    crecimiento_porcentual = ((valor_reciente - valor_inicial) / valor_inicial) * 100

    return round(crecimiento_porcentual, 2)


def calcular_crecimiento_demanda(df_demanda):
# Asegurarse de que 'fecha' es de tipo datetime
    df_demanda['fecha'] = pd.to_datetime(df_demanda['fecha'])

    # Calcular la demanda máxima por año
    df_demanda['year'] = df_demanda['fecha'].dt.year
    demanda_maxima_anual = df_demanda.groupby('year')['valor_demanda_MW'].max().reset_index()

    # Seleccionar los últimos 5 años
    ultimos_5_anos = demanda_maxima_anual.tail(5)

    # Calcular el crecimiento porcentual
    valor_inicial = ultimos_5_anos.iloc[0]['valor_demanda_MW']
    valor_reciente = ultimos_5_anos.iloc[-1]['valor_demanda_MW']
    crecimiento_porcentual = ((valor_reciente - valor_inicial) / valor_inicial) * 100

    return round(crecimiento_porcentual,2)


def calculo_crecimiento_co2(df_co2):

    emisiones_maxima_anual = df_co2.groupby('year')['valor'].max().reset_index()

    # Seleccionar los últimos 5 años
    ultimos_5_anos = emisiones_maxima_anual.tail(5)

    # Calcular el crecimiento porcentual
    valor_inicial = ultimos_5_anos.iloc[0]['valor']
    valor_reciente = ultimos_5_anos.iloc[-1]['valor']
    crecimiento_porcentual = ((valor_reciente - valor_inicial) / valor_inicial) * 100

    return round(crecimiento_porcentual,2)
