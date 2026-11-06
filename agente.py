import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

import sys

import herramientas
from herramientas import (cargar_csv, describir_datos, calcular, calcular_por_mes, ranking,
                          porcentaje_del_total, resumen_estadistico, correlacion, grafico)
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options=types.HttpOptions(timeout=60_000))  # ms: evita esperas infinitas

MODEL = "gemini-flash-lite-latest"
MAX_PASOS = 6  # límite para que el agente no se quede en bucle

HERRAMIENTAS = {f.__name__: f for f in (describir_datos, calcular, calcular_por_mes, ranking,
                                        porcentaje_del_total, resumen_estadistico, correlacion, grafico)}

def crear_instrucciones() -> str:
    """Instrucciones del sistema con las columnas del dataset cargado en ese momento."""
    return (
        "Eres un analista de datos. Responde siempre en español. "
        "Para cualquier cifra debes usar las herramientas: no inventes números ni monedas. "
        f"El dataset tiene estas columnas: {', '.join(herramientas.df.columns)}. "
        "Solo dibuja un gráfico si el usuario lo pide explícitamente."
    )


INSTRUCCIONES = crear_instrucciones()

ULTIMA_LLAMADA = 0.0
INTERVALO_MIN = 4.5  # el plan gratuito permite 15 peticiones por minuto



def crear_config(instrucciones: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=instrucciones,
        tools=list(HERRAMIENTAS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def llamar_modelo(contents, config):
    """Llama a Gemini respetando el límite por minuto y reintentando si falla (429, 500, 503, 504)."""
    global ULTIMA_LLAMADA
    for intento in range(4):
        pausa = INTERVALO_MIN - (time.time() - ULTIMA_LLAMADA)
        if pausa > 0:
            time.sleep(pausa)
        ULTIMA_LLAMADA = time.time()
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code in (429, 500, 503, 504) and intento < 3:
                espera = 30 if e.code == 429 else 2 ** (intento + 1)
                print(f"  [Gemini ocupado ({e.code}), reintento en {espera}s]")
                time.sleep(espera)
            else:
                raise


def preguntar(pregunta: str, instrucciones: str = INSTRUCCIONES, historial: list | None = None) -> str:
    """Responde una pregunta. Si se pasa `historial` (lista), se va ampliando y el agente recuerda
    los turnos anteriores; sin él, cada pregunta empieza de cero."""
    contents = historial if historial is not None else []
    inicio = len(contents)  # para deshacer este turno si el modelo no responde
    contents.append(types.Content(role="user", parts=[types.Part(text=pregunta)]))
    config = crear_config(instrucciones)
    for paso in range(MAX_PASOS):
        resp = llamar_modelo(contents, config)

        # Gemini puede devolver una respuesta sin contenido (por ejemplo, bloqueada por seguridad).
        # Se descarta el turno entero para no dejar el historial con una pregunta sin respuesta.
        candidato = resp.candidates[0] if resp.candidates else None
        if candidato is None or candidato.content is None or not candidato.content.parts:
            del contents[inicio:]
            motivo = getattr(candidato, "finish_reason", None)
            detalle = f" (motivo: {getattr(motivo, 'name', motivo)})" if motivo else ""
            return f"El modelo no ha devuelto ninguna respuesta{detalle}. Prueba a reformular la pregunta."

        llamadas = resp.function_calls
        contents.append(candidato.content)  # lo que dijo o pidió el modelo
        if not llamadas:  # el modelo ya tiene respuesta final
            return resp.text or ""

        resultados = []
        for llamada in llamadas:
            argumentos = dict(llamada.args or {})  # una herramienta sin argumentos puede venir con args=None
            print(f"  -> herramienta: {llamada.name}({argumentos})")
            funcion = HERRAMIENTAS.get(llamada.name)
            if funcion is None:
                resultado = {"error": f"La herramienta {llamada.name} no existe"}
            else:
                try:
                    resultado = funcion(**argumentos)
                except Exception as e:
                    resultado = {"error": f"Argumentos no válidos para {llamada.name}: {e}"}
            resultados.append(types.Part.from_function_response(name=llamada.name, response=resultado))
        contents.append(types.Content(role="user", parts=resultados))

    return "No he conseguido responder en el número máximo de pasos."


if __name__ == "__main__":
    if len(sys.argv) > 1:  # python agente.py ruta/al/archivo.csv
        try:
            cargar_csv(sys.argv[1])
        except Exception as e:
            sys.exit(f"No se pudo cargar el CSV: {e}")
        INSTRUCCIONES = crear_instrucciones()
        print(f"Cargado {sys.argv[1]} ({len(herramientas.df)} filas)")
    print("Agente de datos (escribe 'salir' para terminar)")
    historial = []  # memoria de la conversación
    try:
        while True:
            pregunta = input("\nPregunta: ").strip()
            if not pregunta:
                continue
            if pregunta.lower() in ("salir", "exit", "quit"):
                break
            n = len(historial)
            try:
                print(preguntar(pregunta, instrucciones=INSTRUCCIONES, historial=historial))
            except errors.APIError as e:
                del historial[n:]  # descarta el turno incompleto
                print(f"Error de la API ({e.code}): inténtalo de nuevo.")
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nHasta luego.")