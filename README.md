# Agente de análisis de datos con Gemini

Agente en Python que responde preguntas en lenguaje natural sobre un CSV de ventas.
El modelo (Google Gemini) no calcula nada: decide qué herramienta usar, y las cifras
las calcula pandas. Así las respuestas son verificables.

## Qué hace

Ejemplos de preguntas:

- "¿Qué ciudad tiene más ingresos?"
- "Dame el top 3 de productos por ingresos."
- "¿Qué porcentaje de los ingresos supone Calzado?"
- "¿Cuál es la mediana y la desviación típica de los ingresos por venta?"
- "¿Hay correlación entre unidades e ingresos?"
- "Dibuja un gráfico de barras de ingresos por ciudad."

Recuerda la conversación, así que se pueden hacer preguntas de seguimiento
("¿y en Madrid?").

## Cómo funciona

```
Pregunta -> Gemini decide herramienta(s) -> Python/pandas las ejecuta -> Gemini redacta la respuesta
```

Bucle de agente manual con *function calling* (sin automatismos del SDK):

- 8 herramientas: `describir_datos`, `calcular`, `calcular_por_mes`, `ranking`,
  `porcentaje_del_total`, `resumen_estadistico`, `correlacion`, `grafico`.
- Los docstrings de las funciones son las descripciones que lee el modelo.
- Los errores de las herramientas se devuelven como datos, para que el modelo se corrija solo.
- Límite de pasos, control de velocidad (plan gratuito) y reintentos ante 429/500/503.

## Evaluación

`evaluacion.py` lanza 24 preguntas cuya respuesta correcta se calcula con pandas
(*ground truth*) y comprueba si la respuesta del agente contiene el valor correcto.

Limitaciones conocidas: la comprobación es por coincidencia de texto, el LLM no es
determinista y las preguntas son sencillas. No se prueban preguntas sin respuesta
posible.

## Instalación

Requiere Python 3.13 y una API key gratuita de [Google AI Studio](https://aistudio.google.com/).

```bash
git clone https://github.com/mauroredondo/agente-datos.git
cd agente-datos
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=tu_clave" > .env     
python generar_datos.py                    # crea ventas.csv (datos sintéticos)
python agente.py                           # modo conversación
python evaluacion.py                       # evaluación (~2-3 min)
```

## Estructura

| Archivo | Función |
|---|---|
| `agente.py` | Bucle del agente y llamadas a Gemini |
| `herramientas.py` | Herramientas de análisis con pandas |
| `evaluacion.py` | Evaluación con respuestas de referencia |
| `generar_datos.py` | Genera el dataset sintético |

## Próximos pasos

- Repetir la evaluación varias veces y dar intervalo de confianza.
- Añadir preguntas sin respuesta posible para medir alucinaciones.
- Permitir cargar cualquier CSV.
