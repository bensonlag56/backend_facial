import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

def create_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def insert_usuario(nombre, apellido, codigo_unico, email, requisitoriado, ruta_modelo):
    connection = create_connection()
    if connection is None:
        return False
    try:
        cursor = connection.cursor()
        sql = """
        INSERT INTO usuarios (nombre, apellido, codigo_unico, email, requisitoriado, modelo_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (nombre, apellido, codigo_unico, email, requisitoriado, ruta_modelo)
        cursor.execute(sql, values)
        connection.commit()
        return True
    except Error as e:
        print(f"Error inserting user: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_all_usuarios():
    connection = create_connection()
    if connection is None:
        return []
    try:
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT * FROM usuarios"
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"Error fetching users: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def delete_usuario(codigo_unico):
    connection = create_connection()
    if connection is None:
        return False
    try:
        cursor = connection.cursor()
        sql = "DELETE FROM usuarios WHERE codigo_unico = %s"
        cursor.execute(sql, (codigo_unico,))
        connection.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"Error deleting user: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
