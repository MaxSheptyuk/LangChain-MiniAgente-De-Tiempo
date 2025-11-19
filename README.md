# LangChain-MiniAgente-De-Tiempo
Una demo rápida sobre como podemos usar LangChain con Open AI construyendo un  mini agente que proporciona info básica sobre el tempo y consejos adicionales.


Pequeño experimento de **mini agente de tiempo** construido con:

- 🧠 **LangChain** (agent + tools)
- 📡 **Open-Meteo** (API pública de datos meteorológicos)
- 👀 **LangSmith** (trazas completas del RunTree)
- 🗺️ **CSV local de ciudades** (`worldcities.csv`) para resolver coordenadas

La idea es sencilla:

> Le preguntas en castellano por el tiempo en una ciudad  
> (“¿Qué tiempo hace en Madrid ahora mismo?”)  
> y el mini agente se encarga de:
> 1. Buscar las coordenadas de la ciudad en un CSV local.
> 2. Consultar la API pública de Open-Meteo.
> 3. Devolver una respuesta amigable:
>    - situación actual
>    - resumen general de las próximas 24 horas (sin listar cada hora).
> 4. Registrar todas las trazas en LangSmith para poder inspeccionar el **RunTree**:
>    modelo → agente → tool `get_weather` → vuelta al modelo.

---

## ✨ Características

- ✅ **Resolución de ciudades vía CSV local** (`worldcities.csv`), sin depender de APIs de geocoding.
- ✅ **Integración con Open-Meteo** (API pública, sin API key).
- ✅ **Agente LangChain** que decide cuándo usar el tool `get_weather`.
- ✅ **Tool instrumentado**: argumentos y respuestas visibles en LangSmith.
- ✅ **Respuestas en castellano**:
  - Situación actual: temperatura, viento y breve interpretación (“hace fresco”, “brisa ligera”…).
  - Resumen de las próximas 24 horas: rango de temperaturas, tendencia y avisos generales.
- ✅ Código sencillo, pensado para jugar con:
  - prompts,
  - observabilidad,
  - y patrones agent + tools.

---

## 🧱 Stack técnico

- Python 3.11+
- [LangChain](https://python.langchain.com/)
- [langchain-openai](https://github.com/langchain-ai/langchain-openai)
- [LangSmith](https://smith.langchain.com/)
- [Open-Meteo](https://open-meteo.com/)
- pandas
- requests
- python-dotenv

---

## 📁 Estructura del proyecto

```text
LangChain-MiniAgente-De-Tiempo/
├─ src/
│  └─ weather_agent.py      # Script principal del agente
├─ data/
│  └─ worldcities.csv       # CSV con ciudades y coordenadas
├─ dev.env.example          # Plantilla de variables de entorno (sin secretos)
├─ requirements.txt         # Dependencias del proyecto
└─ README.md
