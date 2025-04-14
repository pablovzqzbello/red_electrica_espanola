import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import requests


def last_update():
    user = "root"
    password = "shintomaruyukio"
    database = "red_electrica_espanola"
    host = "localhost"
    engine = create_engine(f'mysql+pymysql://{user}:{password}@localhost/{database}')
    query = "SELECT MAX(fecha) AS fecha_mas_reciente FROM balance_energia"
    try:
        with engine.connect() as connection:
            print("Conexión exitosa a la base de datos")
            last_update_date = pd.read_sql_query(sql = query, con=connection)
            last_update_date = last_update_date["fecha_mas_reciente"].iloc[0]
            #last_update_date = last_update_date.strftime("%Y-%m-%d")
            return last_update_date
    except SQLAlchemyError as e:
        print(f"Error de conexión: {e}")


def extract_balance(start_day, end_day, time_trunc='day'):
    all_data = []
    
    url = 'https://apidatos.ree.es/es/datos/balance/balance-electrico'
        
    params = {
    'start_date': f'{start_day}', 
    'end_date': f'{end_day}', 
    'time_trunc': time_trunc
    }

    response = requests.get(url, params=params)
        
    if response.status_code != 200:
        print(f"Error fetching data for day {start_day}: {response.status_code}")
        
    balance_data = response.json()
    content_data = balance_data.get('included', [])[0].get('attributes', {}).get('content', [])

    data_list = []

    for item in content_data:
        type_name = item['type'] 
        values = item.get('attributes', {}).get('values', [])
            
        for value in values:
            value['type'] = type_name
            data_list.append(value)

    all_data.extend(data_list)  

    df_balance = pd.DataFrame(all_data) 
    df_balance['fecha_extraccion'] = pd.Timestamp.now()
    df_balance["fecha_extraccion"]= df_balance["fecha_extraccion"].dt.floor("s")
    df_balance.rename(columns={'datetime':'fecha', 'value':'valor_balance_GW', 'percentage':'porcentaje','type':'energia'},inplace=True)
    df_balance.drop(['porcentaje'], axis=1, inplace=True)
    df_balance['fecha']=df_balance['fecha'].str.split('T').str[0]
    df_balance['fecha'] = pd.to_datetime(df_balance['fecha'], errors="coerce").dt.date
    return df_balance


def extract_demand(start_day, end_day, category='demanda', widget='evolucion', time_trunc='day'):
    all_data = []

    url = f"https://apidatos.ree.es/es/datos/{category}/{widget}"

    params = {
    'start_date': f'{start_day}', 
    'end_date': f'{end_day}', 
    'time_trunc': time_trunc
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        all_data.append(data)
    else:
        print(f"Error fetching data for day {start_day}: {response.status_code}")

    demanda = []

    for entry in all_data:
        included = entry.get('included', [])
        for item in included:
            values = item.get('attributes', {}).get('values', [])
            for value in values:
                relevant = {'datetime': value.get('datetime'),
                            'demand_value': value.get('value'),
                            'percentage': value.get('percentage')}
                demanda.append(relevant)

    df_demanda = pd.DataFrame(demanda)
    df_demanda['fecha_extraccion'] = pd.Timestamp.now()
    df_demanda["fecha_extraccion"] = df_demanda["fecha_extraccion"].dt.floor("s")
    df_demanda.rename(columns={'datetime': 'fecha', 'demand_value': 'valor_demanda_MW', 'percentage': 'porcentaje'},
                      inplace=True)
    df_demanda.drop(['porcentaje'], axis=1, inplace=True)
    df_demanda['fecha'] = df_demanda['fecha'].str.split('T').str[0]
    df_demanda['fecha'] = pd.to_datetime(df_demanda['fecha'], errors="coerce").dt.date
    return df_demanda


def extract_exchange(start_day, end_day, time_trunc='day', widget='todas-fronteras-fisicos'):
    all_lines = []

    url = f'https://apidatos.ree.es/es/datos/intercambios/{widget}'

    params = {'start_date': f'{start_day}',
              'end_date': f'{end_day}',
              'time_trunc': time_trunc}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data for day {start_day}: {response.status_code}")
        
    exchange_data = response.json()

    lines = []

    for country in exchange_data.get('included', []):
        country_name = country.get('id')

        if 'content' in country.get('attributes', {}):
            for content in country['attributes']['content']:
                trade_type = content.get('attributes', {}).get('title')
                values = content.get('attributes', {}).get('values', [])

                for item in values:
                    line = {'country': country_name,
                            'type': trade_type,
                            'value': item.get('value'),
                            'percentage': item.get('percentage'),
                            'datetime': item.get('datetime')}
                    lines.append(line)

    all_lines.extend(lines)
    df_exchanges = pd.DataFrame(all_lines)
    df_exchanges['fecha_extraccion'] = pd.Timestamp.now()
    df_exchanges["fecha_extraccion"] = df_exchanges["fecha_extraccion"].dt.floor("s")
    df_exchanges.rename(
        columns={'datetime': 'fecha', 'value': 'valor_GW', 'percentage': 'porcentaje', 'type': 'tipo_transaccion',
                 'country': 'pais'}, inplace=True)
    df_exchanges.drop(['porcentaje'], axis=1, inplace=True)
    df_exchanges['fecha'] = df_exchanges['fecha'].str.split('T').str[0]
    df_exchanges['fecha'] = pd.to_datetime(df_exchanges['fecha'], errors="coerce").dt.date
    
    
    return df_exchanges


def extract_generation(start_day, end_day, time_trunc='day'):
    all_gen_df = []

    
    url = 'https://apidatos.ree.es/es/datos/generacion/estructura-generacion'

    params = {
    'start_date': f'{start_day}', 
    'end_date': f'{end_day}', 
    'time_trunc': time_trunc
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data for day {start_day}: {response.status_code}")

    generation_data = response.json()

    gen_df = []

    for included_data in generation_data.get('included', []):
        values = included_data.get('attributes', {}).get('values', [])

        df_gen = pd.DataFrame(values)

        df_gen['type'] = included_data.get('type')
        df_gen['id'] = included_data.get('id')
        df_gen['groupId'] = included_data.get('groupId')
        df_gen['title'] = included_data.get('attributes', {}).get('title')
        df_gen['description'] = included_data.get('attributes', {}).get('description')
        df_gen['color'] = included_data.get('attributes', {}).get('color')
        df_gen['technology_type'] = included_data.get('attributes', {}).get('type')

        gen_df.append(df_gen)

    all_gen_df.extend(gen_df)

    df_generation = pd.concat(all_gen_df, ignore_index=True)

    df_generation = df_generation[
        ['datetime', 'value', 'percentage', 'type', 'id', 'groupId', 'title', 'description', 'color',
         'technology_type']]
    df_generation['fecha_extraccion'] = pd.Timestamp.now()
    df_generation["fecha_extraccion"] = df_generation["fecha_extraccion"].dt.floor("s")
    df_generation.rename(
        columns={'datetime': 'fecha', 'value': 'valor_generacion_GW', 'percentage': 'porcentaje', 'type': 'energia',
                 'technology_type': 'tipo_tecnología'}, inplace=True)
    df_generation.drop(['porcentaje', 'title', 'groupId', 'id', 'description', 'color'], axis=1, inplace=True)
    df_generation['fecha'] = df_generation['fecha'].str.split('T').str[0]
    df_generation['fecha'] = pd.to_datetime(df_generation['fecha'], errors="coerce").dt.date
    return df_generation


def emisiones_co2(start_day, end_day, time_trunc='day'):
    all_gen_df_co2 = []

    url = 'https://apidatos.ree.es/es/datos/generacion/no-renovables-detalle-emisiones-CO2'

    params = {'start_date': f'{start_day}',
              'end_date': f'{end_day}',
              'time_trunc': time_trunc}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data for day {start_day}: {response.status_code}")

    generation_data_co2 = response.json()

    gen_df_co2 = []

    for included_data in generation_data_co2.get('included', []):
        values = included_data.get('attributes', {}).get('values', [])

        df_gen = pd.DataFrame(values)

        df_gen['type'] = included_data.get('type')
        df_gen['id'] = included_data.get('id')
        df_gen['groupId'] = included_data.get('groupId')
        df_gen['title'] = included_data.get('attributes', {}).get('title')
        df_gen['description'] = included_data.get('attributes', {}).get('description')
        df_gen['color'] = included_data.get('attributes', {}).get('color')
        df_gen['technology_type'] = included_data.get('attributes', {}).get('type')

        gen_df_co2.append(df_gen)

    all_gen_df_co2.extend(gen_df_co2)

    df_generation_co2 = pd.concat(all_gen_df_co2, ignore_index=True)

    df_generation_co2 = df_generation_co2[
        ['datetime', 'value', 'percentage', 'type', 'id', 'groupId', 'title', 'description', 'color',
         'technology_type']]
    df_generation_co2.drop(['id', 'groupId', 'title', 'description', 'percentage', 'color', 'technology_type'], axis=1,
                           inplace=True)
    df_generation_co2['fecha_extraccion'] = pd.Timestamp.now()
    df_generation_co2["fecha_extraccion"] = df_generation_co2["fecha_extraccion"].dt.floor("s")
    df_generation_co2.rename(columns={'datetime': 'fecha', 'value': 'valor', 'percentage': 'porcentaje', 'type': 'energia'}, inplace=True)
    df_generation_co2['fecha'] = df_generation_co2['fecha'].str.split('T').str[0]
    df_generation_co2['fecha'] = pd.to_datetime(df_generation_co2['fecha'], errors="coerce").dt.date
    return df_generation_co2



def insert_data(df_balance, df_demanda, df_generation):
    user = "root"
    password = "shintomaruyukio"
    database = "red_electrica_espanola"
    host = "localhost"
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
    with engine.connect() as connection:
        
        df_balance.to_sql('balance_energia', con=connection, if_exists='append', index=False)
        df_generation.to_sql('generacion_energia', con=connection, if_exists='append', index=False)
        df_demanda.to_sql('demanda_energia', con=connection, if_exists='append', index=False)
        #df_exchanges.to_sql('transacciones_energia', con=connection, if_exists='replace', index=False)
        #Desde finales de septiembre no hay actualizaciones de exchanges, se deja comentado hasta nueva orden
        


def actualizar_bdd():

    start_day = last_update()
    
    end_day = datetime.today().strftime('%Y-%m-%d')

    if start_day != end_day:
        df_balance = extract_balance(start_day, end_day)
        df_generation = extract_generation(start_day, end_day)
        df_demanda = extract_demand(start_day, end_day)
        df_emisiones_co2 = emisiones_co2(start_day, end_day)
        
        insert_data(df_balance, df_demanda, df_generation)
        print("La base de datos ha sido actualizada.")
    else:
        print("La base de datos está al día.")




