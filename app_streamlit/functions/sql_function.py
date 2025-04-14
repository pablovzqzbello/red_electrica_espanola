
import nbimporter
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import mysql.connector
import mysql
from mysql.connector import Error
from functions.extraction_data import *
import unicodedata
import pandas as pd

load_dotenv()

config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }



def create_db():
    engine_no_db = create_engine(f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}/')
    try:
        with engine_no_db.connect() as connection: 
            print("Conexión establecida.")
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {config['database']}"))
            connection.execute(text(f"USE {config['database']}"))
            print("Base de datos creada y en uso")
    except Error as e:
        print(f"Error: {e}")



def crear_tablas():
    try:
        connection = mysql.connector.connect(**config)

        if connection.is_connected():
            print("Conexión exitosa a la base de datos")

            # Crear un cursor
            cursor = connection.cursor()

            # Consulta para crear la tabla
            crear_tablas = """
            CREATE TABLE IF NOT EXISTS demanda_energia (
                fecha DATE PRIMARY KEY,
                valor_demanda_MW FLOAT,
                fecha_extraccion DATETIME);
                
            CREATE TABLE IF NOT EXISTS balance_energia (
                fecha DATE,
                valor_balance_MW FLOAT,
                energia VARCHAR(50),
                fecha_extraccion DATETIME,
                PRIMARY KEY (fecha, energia),
                FOREIGN KEY (fecha) REFERENCES demanda_energia(fecha) ON UPDATE CASCADE ON DELETE CASCADE); 

            CREATE TABLE IF NOT EXISTS transacciones_energia (
                pais VARCHAR(50),
                tipo_transaccion VARCHAR(20),
                valor_MW FLOAT,
                fecha DATE,
                fecha_extraccion DATETIME,
                PRIMARY KEY (pais, tipo_transaccion, fecha),
                FOREIGN KEY (fecha) REFERENCES demanda_energia(fecha) ON UPDATE CASCADE ON DELETE CASCADE);

            CREATE TABLE IF NOT EXISTS generacion_energia (
                fecha DATE,
                valor_generacion_MW FLOAT,
                energia VARCHAR(50),
                tipo_tecnología VARCHAR(50),
                fecha_extraccion DATETIME,
                PRIMARY KEY (fecha, energia, tipo_tecnología),
                FOREIGN KEY (fecha) REFERENCES demanda_energia(fecha) ON UPDATE CASCADE ON DELETE CASCADE);

            CREATE TABLE IF NOT EXISTS emisiones_co2 (
                fecha DATE,
                valor_emisiones FLOAT,
                energia VARCHAR(50),
                fecha_extraccion DATETIME,
                PRIMARY KEY (fecha, energia),
                FOREIGN KEY (fecha) REFERENCES demanda_energia(fecha) ON UPDATE CASCADE ON DELETE CASCADE);

            """

            # Ejecutar la consulta
            cursor.execute(crear_tablas)
            print("Tablas creadas exitosamente.")
    
    except Error as e:
        print(f"Error: {e}")
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Conexión a la base de datos cerrada.")


def insert_data(df_balance, df_generation, df_exchanges, df_emisiones_co2, df_demanda):
    
    engine = create_engine(f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}/{config["database"]}')
    
    try:
        
        with engine.connect() as connection:
            
            print("Conexión exitosa a la base de datos")

            df_balance.to_sql('balance_energia', con=engine, if_exists='replace', index=False)
            print("Datos de balance insertados correctamente.")
            
            df_exchanges.to_sql('transacciones_energia', con=engine, if_exists='replace', index=False)
            print("Datos de intercambios insertados correctamente.")

            df_generation.to_sql('generacion_energia', con=engine, if_exists='replace', index=False)
            print("Datos de generación insertados correctamente.")

            df_emisiones_co2.to_sql('emisiones_co2', con=engine, if_exists='replace', index=False)
            print("Datos de emisiones insertados correctamente.")
    
            df_demanda.to_sql('demanda_energia', con=engine, if_exists='replace', index=False)
            print("Datos de demanda insertados correctamente.")


    except Exception as e:
        print(f"Error al insertar datos: {e}")


def extract_data(query):
    load_dotenv()
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    database = os.getenv('DB_NAME')

    # Crear la cadena de conexión con el formato correcto
    engine = create_engine(f'mysql+pymysql://{user}:{password}@localhost/{database}')

    try:
        with engine.connect() as connection:
            print("Conexión exitosa a la base de datos")
            df=pd.read_sql_query(sql=query, con=connection)
            return df

    except SQLAlchemyError as e:
        print(f"Error de conexión: {e}")



