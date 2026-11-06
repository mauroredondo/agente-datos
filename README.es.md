# Agente de análisis de datos con Gemini

🇬🇧 [English](README.md) | 🇪🇸 Español

Agente en Python que responde preguntas en lenguaje natural sobre un archivo CSV. El modelo (Google Gemini) no calcula nada: decide qué herramienta usar, y las cifras las calcula pandas. Así, cada número de una respuesta se puede comprobar.

## Qué hace

Ejemplos de preguntas (con el dataset de ventas incluido):

- "¿Qué ciudad tiene más ingresos?"
- "Dame el top 3 de productos por ingresos."
- "¿Qué porcentaje de los ingresos supone Calzado?"
- "¿Cuál es la mediana y la desviación típica de los ingresos por venta?"
- "¿Hay correlación entre unidades e ingresos?"
- "Dibuja un gráfico de barras de ingresos por ciudad."

Recuerda la conversación, así que se pueden hacer preguntas de seguimiento ("¿y en Madrid?"). Funciona con cualquier CSV, no solo con el de ejemplo.

## Usar tu propio CSV

```bash
python agente.py ruta/a/tus_datos.csv
```

Al cargar el archivo, el agente:

- Detecta el separador (`,` `;` tabulador) y la codificación (UTF-8 y, si falla, Latin-1).
- Convierte las columnas de texto a números (admite coma decimal, como `1,5`) o a fechas (formato día/mes/año) cuando casi todos sus valores lo permiten.
- Informa al modelo de las columnas disponibles. La herramienta por mes usa la primera columna de fecha que encuentre.

Sin argumentos, usa el dataset de ejemplo, `ventas.csv`.

## Cómo funciona

```
Pregunta -> Gemini decide herramienta(s) -> Python/pandas las ejecuta -> Gemini redacta la respuesta
```

Es un bucle de agente escrito a mano con *function calling* (el modo automático del SDK está desactivado):

- **8 herramientas:** `describir_datos`, `calcular`, `calcular_por_mes`, `ranking`, `porcentaje_del_total`, `resumen_estadistico`, `correlacion`, `grafico`.
- Los docstrings de las funciones son las descripciones que lee el modelo.
- Los errores de las herramientas se devuelven al modelo como datos, para que se corrija solo (columna que no existe, operación no válida, filtro sin filas).
- Un límite de pasos (6) evita bucles. Cada llamada tiene un timeout de 60 segundos, respeta el límite del plan gratuito (15 peticiones por minuto) y se reintenta ante 429, 500, 503 y 504.
- El modelo es `gemini-flash-latest`, un alias que Google puede mover a una versión más nueva. Si necesitas resultados reproducibles, fija un identificador de modelo concreto en `agente.py`.

## Evaluación

`evaluacion.py` lanza 24 preguntas cuya respuesta correcta se calcula con pandas (*ground truth*) y comprueba que la respuesta del agente contiene ese valor. También lanza 5 preguntas que **no se pueden** responder con las columnas del dataset ("¿Cuál es el beneficio total?", "¿Cuántas devoluciones hubo en Madrid?"). En esas, lo correcto es decir que no están los datos, y dar una cifra cuenta como alucinación. Además cuenta las respuestas que mencionan una moneda, ya que el dataset no tiene columna de moneda.

```bash
python evaluacion.py                        # una ejecución
python evaluacion.py --repeticiones 5       # cada pregunta 5 veces, con intervalos de confianza
python evaluacion.py --completa             # además, unas instrucciones mínimas, para comparar
python evaluacion.py --sin-imposibles       # sin las preguntas sin respuesta posible
```

Cada respuesta se guarda en `resultados.csv` según llega (no se sube a GitHub). Al terminar escribe `resumen_evaluacion.md` con la fecha, el modelo, el porcentaje de acierto y dos intervalos de confianza al 95 %:

- **Wilson**, que trata cada respuesta como independiente.
- **Bootstrap por preguntas**, que remuestrea las preguntas. Con varias repeticiones es el más honesto, porque las respuestas repetidas a una misma pregunta no son independientes.

Con `--completa`, los dos prompts se comparan con un contraste exacto de signos sobre las preguntas en las que sus tasas de acierto difieren. Las preguntas que fallan por un error de la API (sobrecarga, tiempo de espera, cuota) se cuentan aparte y nunca como fallos.

Limitaciones de la evaluación:

- La comprobación es por coincidencia de texto, así que una respuesta correcta redactada de forma inesperada cuenta como fallo. En las preguntas sin respuesta posible busca frases como "no existe" o "no hay", así que conviene leer también `resultados.csv`.
- El modelo no es determinista y las preguntas son sencillas, así que una nota alta no significa que el agente sea fiable con otras más difíciles.
- La evaluación solo cubre el dataset de ejemplo, no tus propios CSV.
- El plan gratuito tiene una cuota de peticiones. Si se agota, la ejecución termina con errores de API y sin resultado.
- Aquí todavía no se publica ninguna cifra de acierto: el resumen se genera ejecutando el script con tu propia clave.

## Tests

Las herramientas y el bucle del agente tienen tests unitarios que no necesitan clave ni internet. El bucle se prueba con un Gemini falso que devuelve respuestas guionizadas.

```bash
python -m unittest discover -s tests -t .
```

Los tests cubren cada herramienta frente a pandas y sus casos de error (columna que no existe, operación no válida, filtro sin filas, valores `NaN`, CSV con separador `;`, coma decimal y Latin-1), el bucle del agente (llamadas a herramientas, errores devueltos como datos, límite de pasos, memoria, respuestas vacías del modelo) y la estadística que usa la evaluación.

## Instalación

Requiere Python 3.13 y una API key gratuita de [Google AI Studio](https://aistudio.google.com/).

```bash
git clone https://github.com/mauroredondo/agente-datos.git
cd agente-datos
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=tu_clave" > .env   # git ignora .env
python generar_datos.py                 # crea ventas.csv (datos sintéticos)
python agente.py                        # conversación con el dataset de ejemplo
python agente.py ruta/al/archivo.csv    # conversación con tu propio CSV
```

`requirements-lock.txt` tiene las versiones exactas utilizadas.

## Estructura

| Archivo | Función |
|---|---|
| `agente.py` | Bucle del agente y llamadas a Gemini (ritmo, reintentos, límite de pasos) |
| `herramientas.py` | Las 8 herramientas de pandas y el cargador de CSV |
| `evaluacion.py` | Evaluación con preguntas de referencia y sin respuesta posible, repeticiones e intervalos de confianza |
| `generar_datos.py` | Genera el dataset sintético de ventas (300 filas, semilla fija) |
| `tests/` | Tests unitarios de las herramientas y del bucle del agente |
| `pruebas/hola.py` | Prueba de conexión de una sola llamada para comprobar la clave |

## Próximos pasos

- Ejecutar la evaluación con varias repeticiones y publicar las cifras con sus intervalos.
- Comparar modelos (por ejemplo uno más pequeño) con el mismo contraste pareado.
- Añadir una herramienta SQL (SQLite) y guardarraíles que rechacen preguntas ambiguas o ajenas al tema.
- Sustituir la coincidencia de texto de las preguntas sin respuesta posible por una comprobación más estricta.
- Admitir archivos Excel y varios CSV a la vez.

## Tecnologías

Python 3.13 · Google Gemini (function calling) · pandas · matplotlib · unittest

## Qué aprendí

- Construir un bucle de agente a mano: esquemas de herramientas a partir de docstrings, ejecución de las llamadas, errores devueltos como datos y límite de pasos.
- Mantener al modelo lejos de la aritmética, para que las respuestas se puedan verificar con pandas.
- Evaluar un sistema con LLM mediante preguntas con respuesta de referencia, y dejar claro lo que la evaluación no puede demostrar.
- Probar las partes deterministas sin la API, y el bucle del agente con un modelo falso.

## Autor

**Mauro Redondo Beltrán** · Grado en Matemáticas (Universidad de Salamanca) · Máster en Inteligencia Artificial (La Salle - URL)
