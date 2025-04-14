from modulefinder import Module

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

def preprocess_data(df_demanda, df_exchanges, df_generation):
    # Eliminar columnas innecesarias
    df_demanda = df_demanda.drop(columns=["fecha_extraccion"])
    df_exchanges = df_exchanges.drop(columns=["fecha_extraccion"])
    df_generation = df_generation.drop(columns=["fecha_extraccion"])

    # Filtrar y limpiar df_generation
    df_generation_filtered = df_generation[
        (df_generation['energia'] == 'Generación total') | (df_generation['tipo_tecnología'] == 'Generación total')]
    df_generation_filtered = df_generation_filtered.drop(columns=['energia', 'tipo_tecnología'])
    df_generation_filtered = df_generation_filtered.reset_index(drop=True)

    # Filtrar df_exchanges
    df_exchanges_filtered = df_exchanges[(df_exchanges['tipo_transaccion'] == 'saldo')]
    df_exchanges_agg = df_exchanges_filtered.groupby("fecha", as_index=False)["valor_MW"].sum()

    # Merge de los DataFrames
    df_merge_test = df_demanda.merge(df_exchanges_agg, on="fecha", how="left")
    df_merge_test = df_merge_test.rename(columns={"valor_MW": "saldo_intercambios"})
    df_merge_test = df_merge_test.merge(df_generation_filtered, on="fecha", how="left")

    # Interpolación de valores NaN'S en 'saldo_intercambios'
    df_merge_test['saldo_intercambios'] = df_merge_test['saldo_intercambios'].interpolate(method='linear')

    # Crear nuevas columnas basadas en la fecha
    df_merge_test['fecha'] = pd.to_datetime(df_merge_test['fecha'], format='%Y-%m-%d')
    df_merge_test['año'] = df_merge_test['fecha'].dt.year
    df_merge_test['mes'] = df_merge_test['fecha'].dt.month
    df_merge_test['dia'] = df_merge_test['fecha'].dt.day
    df_merge_test['dia_semana'] = df_merge_test['fecha'].dt.weekday
    df_merge_test['es_fin_de_semana'] = df_merge_test['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
    df_merge_test = df_merge_test.drop(columns=["fecha"])

    return df_merge_test


def escalador(df, T=7, target_column="valor_demanda_MW", scaler_filename="models/scaler.pkl"):

    # Seleccionar las columnas a escalar, excluyendo la columna objetivo
    columnas_a_escalar = df.drop(columns=[target_column]).columns
    valores = df[columnas_a_escalar].values
    objetivo = df[target_column].values

    # Seleccionar las columnas a escalar, excluyendo la columna objetivo
    valores = valores
    objetivo = objetivo

    # Cargar el escalador desde el archivo pickle
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    # Aplicar el escalador a los valores y al objetivo
    valores_escalados = scaler.fit_transform(valores)
    objetivo_escalado = scaler.fit_transform(objetivo.reshape(-1, 1))

    # Crear listas para las secuencias de entrada y salida
    X = []
    y = []
    
    # Generar ventanas deslizantes
    for t in range(len(df) - T):
        # Toma valores de X de t en t con stride de 1
        x = valores_escalados[t : t + T]
        X.append(x)
        
        # Toma los valores de t en t
        y_ = objetivo_escalado[t + T]
        y.append(y_)
    
    # Convertir listas a arrays de numpy
    X = np.array(X)  # Dimensión: (samples, timesteps, features)
    y = np.array(y)  # Dimensión: (samples, 1)
    
    return X, y

def train_test_split_data(valores_escalados, objetivo_escalado, train_ratio=0.8):
    # Calcular el tamaño del conjunto de entrenamiento
    train_size = int(len(valores_escalados) * train_ratio)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test = valores_escalados[:train_size], valores_escalados[train_size:]
    y_train, y_test = objetivo_escalado[:train_size], objetivo_escalado[train_size:]

    return X_train, X_test, y_train, y_test


def modelo_neuronal_rnn(X_test, y_test, scaler_filename="models/scaler.pkl", model_filename="models/rnn_model.pkl"):
    # Cargar el escalador preentrenado desde el archivo pickle
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    # Cargar el modelo RNN preentrenado desde el archivo pickle
    with open(model_filename, "rb") as f:
        model_rnn = pickle.load(f)

    # Realizar predicciones
    predictions_scaled = model_rnn.predict(X_test)

    # Asegurar que el escalador reciba datos en el formato correcto para la transformación inversa
    predictions = scaler.inverse_transform(predictions_scaled)
    expected = scaler.inverse_transform(y_test)

    # Mostrar predicciones
    for i in range(len(y_test)):
        print(f"Real: {expected[i]} | Predicción: {predictions[i]}")

    # Crear un DataFrame para las gráficas
    df = pd.DataFrame({
        'Fecha': range(len(expected)),  # Asumiendo que cada índice es una fecha secuencial
        'Real': expected.flatten(),
        'Predicción': predictions.flatten()})

    # Graficar con plotly
    fig_rnn = px.line(df, x='Fecha', y=['Real', 'Predicción'], labels={'Fecha': 'Tiempo', 'value': 'Valor'},
                       title="Predicciones vs Valores Reales")
    return st.plotly_chart(fig_rnn)


def modelo_neuronal_lstm(X_test, y_test, scaler_filename="models/scaler.pkl", model_filename="models/lstm_model.pkl"):
    # Cargar el escalador preentrenado desde el archivo pickle
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    # Cargar el modelo LSTM preentrenado desde el archivo pickle
    with open(model_filename, "rb") as f:
        model_lstm = pickle.load(f)

    # Realizar predicciones
    predictions_scaled = model_lstm.predict(X_test)

    # Asegurar que el escalador reciba datos en el formato correcto para la transformación inversa
    predictions = scaler.inverse_transform(predictions_scaled)
    expected = scaler.inverse_transform(y_test)

    # Mostrar predicciones
    for i in range(len(y_test)):
        print(f"Real: {expected[i]} | Predicción: {predictions[i]}")

    # Crear un DataFrame para las gráficas
    df = pd.DataFrame({
        'Fecha': range(len(expected)),  # Asumiendo que cada índice es una fecha secuencial
        'Real': expected.flatten(),
        'Predicción': predictions.flatten()})

    # Graficar con plotly
    fig_lstm = px.line(df, x='Fecha', y=['Real', 'Predicción'], labels={'Fecha': 'Tiempo', 'value': 'Valor'},
                  title="Predicciones vs Valores Reales")
    return st.plotly_chart(fig_lstm)

def modelo_neuronal_gru(X_test, y_test, scaler_filename="models/scaler.pkl", model_filename="models/gru_model.pkl"):
    # Cargar el escalador preentrenado desde el archivo pickle
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    # Cargar el modelo LSTM preentrenado desde el archivo pickle
    with open(model_filename, "rb") as f:
        model_lstm = pickle.load(f)

    # Realizar predicciones
    predictions_scaled = model_lstm.predict(X_test)

    # Asegurar que el escalador reciba datos en el formato correcto para la transformación inversa
    predictions = scaler.inverse_transform(predictions_scaled)
    expected = scaler.inverse_transform(y_test)

    # Mostrar predicciones
    for i in range(len(y_test)):
        print(f"Real: {expected[i]} | Predicción: {predictions[i]}")

    # Crear un DataFrame para las gráficas
    df = pd.DataFrame({
        'Fecha': range(len(expected)),  # Asumiendo que cada índice es una fecha secuencial
        'Real': expected.flatten(),
        'Predicción': predictions.flatten()})

    # Graficar con plotly
    fig_gru = px.line(df, x='Fecha', y=['Real', 'Predicción'], labels={'Fecha': 'Tiempo', 'value': 'Valor'},
                  title="Predicciones vs Valores Reales")
    return st.plotly_chart(fig_gru)

def predict_7_days_rnn(
        scaler_filename="models/scaler.pkl",
        model_filename="models/rnn_model.pkl",
        last_sequence=None):

    # Cargar el scaler y el modelo
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    #return predictions
    if last_sequence.ndim == 3:
        last_sequence = last_sequence[0]  

    if last_sequence is None or last_sequence.ndim != 2:
        raise ValueError("`last_sequence` debe ser un array 2D con forma (T, n_features).")
    
    predictions_scaled = []
    input_sequence = last_sequence.reshape(7,7)

    for _ in range(7):  # Predecir 7 días
        # Redimensionar la secuencia para cumplir con el formato del modelo
        input_sequence_reshaped = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

        # Realizar la predicción
        prediction_scaled = model.predict(input_sequence_reshaped)[0, 0]  # Extraer el valor escalar
        predictions_scaled.append(prediction_scaled)

        # Actualizar la secuencia de entrada
        # Desplazar los timesteps anteriores y añadir la nueva predicción como una característica adicional
        new_timestep = np.zeros(input_sequence.shape[1])
        new_timestep[0] = prediction_scaled  # Suponiendo que la predicción corresponde a la primera característica
        input_sequence = np.vstack((input_sequence[1:], new_timestep))
    
    # Invertir la escala de las predicciones
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    # Crear un DataFrame para las predicciones
    days = list(range(1, 8))  # Días 1 al 7
    predictions_df = pd.DataFrame({
        "Día": days,
        "Demanda (MW)": predictions.flatten()})

    # Crear el gráfico con plotly.express
    fig_rnn = px.line(
        predictions_df,
        x="Día",
        y="Demanda (MW)",
        title="Predicción de 7 días de demanda",
        markers=True,
        template="plotly_white")

    return st.plotly_chart(fig_rnn)


def predict_7_days_lstm(
        scaler_filename="models/scaler.pkl",
        model_filename="models/lstm_model.pkl",
        last_sequence=None):

    # Cargar el scaler y el modelo
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    #return predictions
    if last_sequence.ndim == 3:
        last_sequence = last_sequence[0]  

    if last_sequence is None or last_sequence.ndim != 2:
        raise ValueError("`last_sequence` debe ser un array 2D con forma (T, n_features).")
    
    predictions_scaled = []
    input_sequence = last_sequence.reshape(7,7)

    for _ in range(7):  # Predecir 7 días
        # Redimensionar la secuencia para cumplir con el formato del modelo
        input_sequence_reshaped = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

        # Realizar la predicción
        prediction_scaled = model.predict(input_sequence_reshaped)[0, 0]  # Extraer el valor escalar
        predictions_scaled.append(prediction_scaled)

        # Actualizar la secuencia de entrada
        # Desplazar los timesteps anteriores y añadir la nueva predicción como una característica adicional
        new_timestep = np.zeros(input_sequence.shape[1])
        new_timestep[0] = prediction_scaled  # Suponiendo que la predicción corresponde a la primera característica
        input_sequence = np.vstack((input_sequence[1:], new_timestep))
    
    # Invertir la escala de las predicciones
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    # Crear un DataFrame para las predicciones
    days = list(range(1, 8))  # Días 1 al 7
    predictions_df = pd.DataFrame({
        "Día": days,
        "Demanda (MW)": predictions.flatten()})

    # Crear el gráfico con plotly.express
    fig_lstm = px.line(
        predictions_df,
        x="Día",
        y="Demanda (MW)",
        title="Predicción de 7 días de demanda",
        markers=True,
        template="plotly_white")

    return st.plotly_chart(fig_lstm)


def predict_7_days_gru(
        scaler_filename="models/scaler.pkl",
        model_filename="models/gru_model.pkl",
        last_sequence=None):
    # Cargar el scaler y el modelo
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # return predictions
    if last_sequence.ndim == 3:
        last_sequence = last_sequence[0]

    if last_sequence is None or last_sequence.ndim != 2:
        raise ValueError("`last_sequence` debe ser un array 2D con forma (T, n_features).")

    predictions_scaled = []
    input_sequence = last_sequence.reshape(7, 7)

    for _ in range(7):  # Predecir 7 días
        # Redimensionar la secuencia para cumplir con el formato del modelo
        input_sequence_reshaped = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

        # Realizar la predicción
        prediction_scaled = model.predict(input_sequence_reshaped)[0, 0]  # Extraer el valor escalar
        predictions_scaled.append(prediction_scaled)

        # Actualizar la secuencia de entrada
        # Desplazar los timesteps anteriores y añadir la nueva predicción como una característica adicional
        new_timestep = np.zeros(input_sequence.shape[1])
        new_timestep[0] = prediction_scaled  # Suponiendo que la predicción corresponde a la primera característica
        input_sequence = np.vstack((input_sequence[1:], new_timestep))

    # Invertir la escala de las predicciones
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    # Crear un DataFrame para las predicciones
    days = list(range(1, 8))  # Días 1 al 7
    predictions_df = pd.DataFrame({
        "Día": days,
        "Demanda (MW)": predictions.flatten()})

    # Crear el gráfico con plotly.express
    fig_gru = px.line(
        predictions_df,
        x="Día",
        y="Demanda (MW)",
        title="Predicción de 7 días de demanda",
        markers=True,
        template="plotly_white")

    return st.plotly_chart(fig_gru)


def model_prophet(df):
    # Preparación del df para procesar mediante Prophet
    df = df.rename(columns={"año": "year", "mes": "month", "dia": "day"})
    df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])
    df_prophet = df[['valor_demanda_MW', 'fecha']]
    df_prophet = df_prophet.rename(columns={'valor_demanda_MW': 'y', 'fecha': 'ds'})
    df_prophet = df_prophet[['ds', 'y']]

    # Llamada y entrenamiento del modelo
    model = Prophet()
    model.fit(df_prophet)

    # Predicciones del modelo
    future = model.make_future_dataframe(periods=31)
    forecast = model.predict(future)

    # Gráfico 1: Predicción vs Datos Reales
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines+markers', name='Datos Reales'))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicciones'))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', name='Confianza Superior', line=dict(dash='dot')))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', name='Confianza Inferior', line=dict(dash='dot')))
    fig1.update_layout(title="Predicciones vs Datos Reales", xaxis_title="Fecha", yaxis_title="Demanda (MW)")
    st.plotly_chart(fig1)

    # Gráfico 2: Errores a lo largo del tiempo
    df_errors = forecast[['ds', 'yhat']].merge(df_prophet, on='ds', how='left')
    df_errors['error'] = abs(df_errors['y'] - df_errors['yhat'])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_errors['ds'], y=df_errors['error'], mode='lines+markers', name='Error Absoluto'))
    fig2.update_layout(title="Errores en Predicciones", xaxis_title="Fecha", yaxis_title="Error Absoluto (MW)")
    st.plotly_chart(fig2)

    # Gráfico 3: Comparación por Granularidad
    granularities = ['D', 'W', 'ME']  # Día, Semana, Mes
    for gran in granularities:
        df_gran = df_errors.resample(gran, on='ds').mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df_gran.index, y=df_gran['y'], name='Datos Reales'))
        fig3.add_trace(go.Bar(x=df_gran.index, y=df_gran['yhat'], name='Predicciones'))
        fig3.update_layout(title=f"Predicciones y Datos Reales - Granularidad {gran}",
                           xaxis_title="Fecha", yaxis_title="Demanda Promedio (MW)", barmode='group')
        st.plotly_chart(fig3)

    # Gráfico 4: Componentes del modelo Prophet
    st.write("Componentes del modelo Prophet")
    fig4 = model.plot_components(forecast)
    st.pyplot(fig4)

    # Gráfico 5: Histograma de errores
    fig5 = px.histogram(df_errors, x='error', nbins=20, title="Histograma de Errores", labels={'error': 'Error Absoluto'})
    st.plotly_chart(fig5)

    # Gráfico 6: Distribución de errores (Densidad)
    #fig6 = px.density_contour(df_errors, x='ds', y='error', title="Distribución de Errores", labels={'ds': 'Fecha', 'error': 'Error Absoluto'})
    #st.plotly_chart(fig6)

    # Gráfico 7: Acumulación de Predicciones y Reales
    #df_cumulative = df_errors.copy()
    #df_cumulative['real_cumsum'] = df_cumulative['y'].cumsum()
    #df_cumulative['pred_cumsum'] = df_cumulative['yhat'].cumsum()
    #fig7 = go.Figure()
    #fig7.add_trace(go.Scatter(x=df_cumulative['ds'], y=df_cumulative['real_cumsum'], mode='lines', name='Acumulado Real'))
    #fig7.add_trace(go.Scatter(x=df_cumulative['ds'], y=df_cumulative['pred_cumsum'], mode='lines', name='Acumulado Predicción'))
    #fig7.update_layout(title="Demanda Acumulada: Reales vs Predicciones", xaxis_title="Fecha", yaxis_title="Demanda Acumulada (MW)")
    #st.plotly_chart(fig7)


    for period in [7, 14, 30]:
            last_n_days = df_prophet.iloc[-period:]

            # Próximos N días predichos
            next_n_days = forecast.iloc[len(df_prophet):len(df_prophet) + period]

            # Crear gráfico
            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(x=last_n_days['ds'], y=last_n_days['y'], mode='lines+markers', name='Últimos Días '
                                                                                                        'Reales)', line=dict(color='blue')))
            fig8.add_trace(go.Scatter(x=next_n_days['ds'], y=next_n_days['yhat'], mode='lines+markers', name='Próximos Días (Predicción)', line=dict(color='red')))

            title = f"Comparación Últimos {period} Días vs Próximos {period} Días"
            fig8.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Demanda (MW)")
            st.plotly_chart(fig8)

def visual_loss_rnn(history_filename='models/history_rnn.json'):

    with open(history_filename, "r") as f:
        history_rnn = json.load(f)

        df_rnn = pd.DataFrame(history_rnn)

    return st.plotly_chart(px.line(df_rnn, y=['loss', 'val_loss'], title='Función de pérdida (MSE)', labels={'value':'Error(MSE)', 'index':'Epochs'}))

def visual_loss_lstm(history_filename='models/history_lstm.json'):

    with open(history_filename, "r") as f:
         history_lstm = json.load(f)

         df_lstm = pd.DataFrame(history_lstm)

    return st.plotly_chart(px.line(df_lstm, y=['loss', 'val_loss'],title='Función de pérdida (MSE)',labels={'value':'Error(MSE)', 'index':'Epochs'}))

def visual_loss_gru(history_filename='models/history_gru.json'):

    with open(history_filename, "r") as f:
        history_gru = json.load(f)

        df_gru = pd.DataFrame(history_gru)

    return st.plotly_chart(px.line(df_gru, y=['loss', 'val_loss'], title='Función de pérdida (MSE)', labels={'value':'Error(MSE)', 'index':'Epochs'}))