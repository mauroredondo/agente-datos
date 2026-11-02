# Agente de análisis de datos con Gemini

Agente en Python que responde preguntas en lenguaje natural sobre **cualquier CSV**.
El modelo (Google Gemini) no calcula nada: decide qué herramienta usar, y las cifras
las calcula pandas. Así las respuestas son verificables.

## Qué hace

Ejemplos de preguntas (sobre el dataset de ventas de demostración):

- "¿Qué ciudad tiene más ingresos?"
- "Dame el top 3 de productos por ingresos."
- "¿Qué porcentaje de los ingresos supone Calzado?"
- "¿Cuál es la mediana y la desviación típica de los ingresos por venta?"
- "¿Hay correlación entre unidades e ingresos?"
- "Dibuja un gráfico de barras de ingresos por ciudad."

Recuerda la conversación, así que se pueden hacer preguntas de seguimiento
("¿y en Madrid?").

## Usar tu propio CSV

```bash
python agente.py ruta/a/tus_datos.csv
```

Al cargar el archivo, el agente:

- Detecta el separador (`,` `;` tabulador) y la codificación.
- Convierte automáticamente las columnas de texto a números (admite coma decimal,
  como `1,5`) o a fechas (formato día/mes/año) cuando casi todos sus valores lo permiten.
- Informa al modelo de las columnas disponibles, y las herramientas por mes
  usan la primera columna de fecha que encuentren.

Sin argumentos, usa el dataset de demostración `ventas.csv`.

## Cómo funciona

```
Pregunta -> Gemini decide herramienta(s) -> Python/pandas las ejecuta -> Gemini redacta la respuesta
```

Bucle de agente manual con *function calling* (sin automatismos del SDK):

- 8 herramientas: `describir_datos`, `calcular`, `calcular_por_mes`, `ranking`,
  `porcentaje_del_total`, `resumen_estadistico`, `correlacion`, `grafico`.
- Los docstrings de las funciones son las descripciones que lee el modelo.
- Los errores de las herramientas se devuelven como datos, para que el modelo se corrija solo.
- Límite de pasos, control de velocidad (plan gratuito), reintentos ante 429/500/503
  y timeout en las llamadas.

## Evaluación

`evaluacion.py` lanza 24 preguntas sobre `ventas.csv` cuya respuesta correcta se calcula
con pandas (*ground truth*) y comprueba si la respuesta del agente contiene el valor correcto.
Resultado: **X/Y correctas** (rellenar tras ejecutarla).

Limitaciones conocidas: la comprobación es por coincidencia de texto, el LLM no es
determinista, las preguntas son sencillas y la evaluación solo cubre el dataset de
demostración (no los CSV propios). No se prueban preguntas sin respuesta posible.

## Instalación

Requiere Python 3.13 y una API key gratuita de [Google AI Studio](https://aistudio.google.com/).

```bash
git clone https://github.com/mauroredondo/agente-datos.git
cd agente-datos
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=tu_clave" > .env     # nunca subas este archivo
python generar_datos.py                    # crea ventas.csv (datos sintéticos)
python agente.py                           # dataset de demostración
python agente.py tus_datos.csv             # o tu propio CSV
python evaluacion.py                       # evaluación (~2-3 min)
```

## Estructura

| Archivo | Función |
|---|---|
| `agente.py` | Bucle del agente y llamadas a Gemini |
| `herramientas.py` | Carga de CSV y herramientas de análisis con pandas |
| `evaluacion.py` | Evaluación con respuestas de referencia |
| `generar_datos.py` | Genera el dataset sintético de demostración |

## Próximos pasos

- Repetir la evaluación varias veces y dar intervalo de confianza.
- Añadir preguntas sin respuesta posible para medir alucinaciones.
- Permitir cargar archivos Excel y varios CSV a la vez.
