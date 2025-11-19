"""
Agente de tiempo con LangChain + LangSmith

Idea general:
- A partir del nombre de una ciudad, buscamos sus coordenadas en un CSV local (worldcities.csv).
- Usamos esas coordenadas para consultar la API de Open-Meteo.
- Exponemos todo como un tool de LangChain para que el agente pueda invocarlo.
- Activamos LangSmith para ver el RunTree completo: llamada del agente, uso del tool,
  y todo el flujo bien trazado.

Requisitos:
- Fichero worldcities.csv en el mismo directorio.
- dev.env con al menos:
    OPENAI_API_KEY=...
    LANGSMITH_API_KEY=...        # o directamente LANGCHAIN_API_KEY
    # Opcional, pero recomendado:
    # LANGCHAIN_PROJECT=agente-meteo-demo
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent


# ==============================
#  CONFIGURACIÓN INICIAL
# ==============================

# Cargamos variables de entorno desde dev.env:
# aquí metes tu OPENAI_API_KEY, LANGSMITH_API_KEY, etc.
load_dotenv("dev.env")


# ==============================
#  CARGA DEL CSV DE CIUDADES
# ==============================

# Cargamos el CSV de ciudades una sola vez al inicio.
# Si el CSV no está en el mismo directorio, toca ajustar la ruta.
df_cities = pd.read_csv("Data/worldcities.csv", sep=",")


# ==============================
#  FUNCIONES AUXILIARES
# ==============================

def get_coords(city_name: str):
    """
    Busca latitud y longitud de una ciudad en el CSV worldcities.

    Por qué lo hacemos así:
    - Evitamos depender de otra API de geocoding: todo sale de un fichero local.
    - pandas nos permite filtrar rápido por nombre de ciudad.

    Estrategia:
    - Intentamos primero con 'city' (tal cual viene en el CSV, con acentos).
    - Si no hay match, probamos con 'city_ascii' (versión "plana" del nombre).
    - Si aun así no hay coincidencia, devolvemos (None, None) y dejamos que
      el agente explique el problema al usuario.
    """
    df = df_cities

    # Normalizamos a minúsculas; así 'Madrid' == 'madrid'.
    matches = df[df["city"].str.casefold() == city_name.casefold()]

    if matches.empty:
        # Plan B: city_ascii suele ser la versión sin acentos ni caracteres raros.
        matches = df[df["city_ascii"].str.casefold() == city_name.casefold()]

    if matches.empty:
        return None, None

    # Cogemos la primera coincidencia; para algo más serio, podríamos gestionar
    # ciudades duplicadas (varios "San José", etc.), pero aquí vamos a lo práctico.
    row = matches.iloc[0]
    lat = float(row["lat"])
    lng = float(row["lng"])
    return lat, lng


def build_open_meteo_url(lat: float, lon: float) -> str:
    """
    Construye la URL para consultar la API de Open-Meteo con los parámetros
    que nos interesan.

    Lo separamos en una función por higiene:
    - Si mañana queremos añadir precipitación, nubosidad, etc., basta tocar aquí.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = (
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,wind_speed_10m"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    return base + params


# ==============================
#  TOOL DE LANGCHAIN
# ==============================

@tool
def get_weather(city: str) -> str:
    """
    Tool que usará el agente para obtener el tiempo de una ciudad.

    Flujo interno:
    1. Recibe el nombre de la ciudad como cadena.
    2. Busca coordenadas en el CSV (get_coords).
    3. Construye la URL de Open-Meteo (build_open_meteo_url).
    4. Hace la petición HTTP con requests.
    5. Devuelve el JSON de respuesta como STRING.

    Detalle importante:
    - Siempre devolvemos un string, incluso en caso de error.
      Para los errores, retornamos un pequeño JSON con un campo "error"
      para que el agente lo pueda entender y explicárselo al usuario.
    """
    lat, lon = get_coords(city)

    if lat is None or lon is None:
        # En lugar de lanzar excepción, devolvemos un JSON con mensaje de error.
        return json.dumps(
            {"error": f"No encuentro la ciudad '{city}' en el CSV de ciudades."},
            ensure_ascii=False,
        )

    url = build_open_meteo_url(lat, lon)

    try:
        # Timeout razonable para no quedarnos colgados si la API tarda.
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        # Devolvemos JSON con detalle del error para que el modelo pueda explicarlo.
        return json.dumps(
            {
                "error": "Error llamando a Open-Meteo",
                "detail": str(e),
                "url": url,
            },
            ensure_ascii=False,
        )

    # Si todo va bien, devolvemos el cuerpo de la respuesta tal cual (texto JSON).
    return resp.text


# ==============================
#  PROMPT DEL SISTEMA
# ==============================

SYSTEM_PROMPT = """
Eres un asistente que informa sobre el tiempo de forma clara, cercana y en español.

Dispones de una herramienta llamada `get_weather` que:
- Recibe el nombre de una ciudad.
- Devuelve un JSON (como texto) de la API open-meteo con información del tiempo,
  por ejemplo:
    - current.temperature_2m
    - current.wind_speed_10m
    - hourly.time
    - hourly.temperature_2m
    - hourly.relative_humidity_2m
    - hourly.wind_speed_10m

Tu trabajo es:

1. Cuando el usuario pregunte por el tiempo en una ciudad, llama a la herramienta `get_weather`.

2. Lee el JSON devuelto (viene como una cadena de texto).

3. A partir de ese JSON, responde en ESPAÑOL y de forma CLARA y AMIGABLE. Tu respuesta debe tener dos partes:

   A) SITUACIÓN ACTUAL
      - Indica la ciudad (si está clara por el contexto).
      - Di la temperatura actual en grados Celsius (`current.temperature_2m`).
      - Di la velocidad del viento actual (`current.wind_speed_10m`).
      - Añade una breve interpretación: si hace frío, calor, brisa ligera, viento moderado, etc.

   B) RESUMEN PRÓXIMAS 24 HORAS (SIN LISTAR HORA POR HORA)
      - Usa los datos de `hourly` para aproximadamente las próximas 24 horas.
      - Resume el rango de temperaturas (mínima y máxima aproximadas).
      - Comenta si la tendencia general es que la temperatura suba, baje o se mantenga estable.
      - Indica si se espera viento fuerte o condiciones destacables (mucho frío, mucho calor, etc.).
      - NO enumeres cada hora por separado. Solo un resumen general y, como mucho, algún aviso:
        por ejemplo, “de madrugada refresca bastante” o “a última hora de la tarde aumenta el viento”.

4. No muestres el JSON ni la URL de la API, el usuario no necesita ver eso.

5. Si el JSON tiene un campo 'error', explícale al usuario lo que ha pasado de forma sencilla
   y, si procede, sugiere que pruebe con otra ciudad o revise el nombre.
"""


# ==============================
#  CREACIÓN DEL AGENTE
# ==============================

# Aquí montamos el agente usando create_agent de LangChain.
# Con LangSmith activado, verás:
# - Un run para el agente.
# - Un sub-run para el modelo de lenguaje.
# - Un sub-run para el tool `get_weather`, incluyendo sus argumentos.
agent = create_agent(
    model="openai:gpt-5-mini",   # Cambia el modelo si quieres algo más potente.
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
)


# ==============================
#  PRUEBA RÁPIDA (ENTRY POINT)
# ==============================

if __name__ == "__main__":
    # Pregunta de prueba; si quieres, puedes sustituir esto por un input() clásico.
    user_question = "Qué tiempo hace en Madrid ahora mismo?"

    # Para que en LangSmith puedas localizar fácilmente esta ejecución,
    # añadimos algunos tags en la configuración del run.
    run_config = {
        "tags": ["agente-meteo", "demo-local", "madrid"],
        # "metadata": {"origen": "script_local"},  # Por si quieres meter más contexto.
    }

    # Llamamos al agente. En LangSmith verás toda la película:
    # - Entrada del usuario.
    # - Llamada al modelo.
    # - Invocación del tool get_weather con { "city": "Madrid" }.
    # - De nuevo el modelo interpretando el JSON devuelto.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_question}]},
        config=run_config,
    )

    print("=== RESULTADO RAW DEL AGENTE ===")
    print(result)

    # En muchas versiones de LangChain, result trae una estructura tipo:
    # {"messages": [...]}  y el último mensaje es la respuesta final.
    try:
        last_msg = result["messages"][-1]
        content = last_msg.get("content") if isinstance(last_msg, dict) else last_msg.content
        print("\n=== RESPUESTA DEL AGENTE ===")
        print(content)
    except Exception:
        # Si la estructura es distinta, al menos el RAW ya está impreso arriba.
        pass
