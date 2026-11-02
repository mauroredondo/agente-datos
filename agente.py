import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from herramientas import describir_datos, calcular, grafico

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-flash-lite-latest"
MAX_PASOS = 6  # límite para que el agente no se quede en bucle

HERRAMIENTAS = {f.__name__: f for f in (describir_datos, calcular, grafico)}

INSTRUCCIONES = (
    "Eres un analista de datos. Responde siempre en español. "
    "Para cualquier cifra debes usar las herramientas: no inventes números. "
    "Empieza por describir_datos si no conoces las columnas. "
    "Si dibujas un gráfico, indica el nombre del archivo generado."
)

config = types.GenerateContentConfig(
    system_instruction=INSTRUCCIONES,
    tools=list(HERRAMIENTAS.values()),
    # desactivamos la automática para escribir nosotros el bucle
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def llamar_modelo(contents):
    """Llama a Gemini y reintenta si el servidor está saturado (429, 500, 503)."""
    for intento in range(4):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code in (429, 500, 503) and intento < 3:
                espera = 2 ** (intento + 1)
                print(f"  [Gemini ocupado ({e.code}), reintento en {espera}s]")
                time.sleep(espera)
            else:
                raise


def preguntar(pregunta: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=pregunta)])]

    for paso in range(MAX_PASOS):
        resp = llamar_modelo(contents)
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
                resultado = funcion(**llamada.args)
            resultados.append(
                types.Part.from_function_response(name=llamada.name, response={"result": resultado})
            )
        contents.append(types.Content(role="user", parts=resultados))

    return "No he conseguido responder en el número máximo de pasos."


if __name__ == "__main__":
    print("Agente de datos (escribe 'salir' para terminar)")
    while True:
        pregunta = input("\nPregunta: ").strip()
        if pregunta.lower() in ("salir", "exit"):
            break
        print(preguntar(pregunta))