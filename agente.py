import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from herramientas import (df, describir_datos, calcular, calcular_por_mes, ranking,
                          porcentaje_del_total, resumen_estadistico, correlacion, grafico)
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-flash-latest"
MAX_PASOS = 6  # límite para que el agente no se quede en bucle

HERRAMIENTAS = {f.__name__: f for f in (describir_datos, calcular, calcular_por_mes, ranking,
                                        porcentaje_del_total, resumen_estadistico, correlacion, grafico)}

INSTRUCCIONES = (
    "Eres un analista de datos. Responde siempre en español. "
    "Para cualquier cifra debes usar las herramientas: no inventes números ni monedas. "
    f"El dataset tiene estas columnas: {', '.join(df.columns)}. "
    "Solo dibuja un gráfico si el usuario lo pide explícitamente."
)

ULTIMA_LLAMADA = 0.0
INTERVALO_MIN = 4.5  # el plan gratuito permite 15 peticiones por minuto



def crear_config(instrucciones: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=instrucciones,
        tools=list(HERRAMIENTAS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def llamar_modelo(contents, config):
    """Llama a Gemini respetando el límite por minuto y reintentando si falla (429, 500, 503)."""
    global ULTIMA_LLAMADA
    for intento in range(4):
        pausa = INTERVALO_MIN - (time.time() - ULTIMA_LLAMADA)
        if pausa > 0:
            time.sleep(pausa)
        ULTIMA_LLAMADA = time.time()
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code in (429, 500, 503) and intento < 3:
                espera = 30 if e.code == 429 else 2 ** (intento + 1)
                print(f"  [Gemini ocupado ({e.code}), reintento en {espera}s]")
                time.sleep(espera)
            else:
                raise


def preguntar(pregunta: str, instrucciones: str = INSTRUCCIONES) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=pregunta)])]
    config = crear_config(instrucciones)
    for paso in range(MAX_PASOS):
        resp = llamar_modelo(contents, config)
        llamadas = resp.function_calls

        if not llamadas:  # el modelo ya tiene respuesta final
            return resp.text

        contents.append(resp.candidates[0].content)  # lo que pidió el modelo
        resultados = []
        for llamada in llamadas:
            print(f"  -> herramienta: {llamada.name}({dict(llamada.args)})")
            funcion = HERRAMIENTAS.get(llamada.name)
            if funcion is None:
                resultado = {"error": f"La herramienta {llamada.name} no existe"}
            else:
                try:
                    resultado = funcion(**llamada.args)
                except Exception as e:
                    resultado = {"error": f"Argumentos no válidos para {llamada.name}: {e}"}
        contents.append(types.Content(role="user", parts=resultados))

    return "No he conseguido responder en el número máximo de pasos."


if __name__ == "__main__":
    print("Agente de datos (escribe 'salir' para terminar)")
    try:
        while True:
            pregunta = input("\nPregunta: ").strip()
            if not pregunta:
                continue
            if pregunta.lower() in ("salir", "exit", "quit"):
                break
            print(preguntar(pregunta))
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nHasta luego.")