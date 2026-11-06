"""Tests del bucle del agente con un Gemini falso: no hacen ninguna llamada a la API.

Se ejecutan desde la carpeta raíz del proyecto:
    python -m unittest discover -s tests -t .
"""
import os

# Clave falsa, definida antes de importar agente: load_dotenv no pisa variables ya definidas,
# así que estos tests nunca usan tu clave real aunque exista un .env.
os.environ["GEMINI_API_KEY"] = "clave-falsa-para-tests"

import unittest
from unittest import mock

from google.genai import types

import agente
import herramientas as h
from tests.test_herramientas import datos_prueba


class RespuestaFalsa:
    """Imita lo mínimo de la respuesta de Gemini que usa preguntar()."""

    def __init__(self, texto=None, llamadas=None):
        partes = []
        if texto:
            partes.append(types.Part(text=texto))
        for nombre, args in (llamadas or []):
            partes.append(types.Part(function_call=types.FunctionCall(name=nombre, args=args)))
        self.candidates = [types.Candidate(content=types.Content(role="model", parts=partes))]
        self.function_calls = [p.function_call for p in partes if p.function_call]
        self.text = texto


def modelo_falso(*respuestas):
    """Sustituye llamar_modelo: devuelve las respuestas en orden."""
    return mock.patch.object(agente, "llamar_modelo", side_effect=list(respuestas))


class TestBucleDelAgente(unittest.TestCase):
    def setUp(self):
        self._df_original = h.df
        h.df = datos_prueba()

    def tearDown(self):
        h.df = self._df_original

    def test_respuesta_directa(self):
        historial = []
        with modelo_falso(RespuestaFalsa("Hola")):
            self.assertEqual(agente.preguntar("¿Qué tal?", historial=historial), "Hola")
        self.assertEqual(len(historial), 2)  # pregunta + respuesta

    def test_ejecuta_la_herramienta_y_devuelve_el_resultado_al_modelo(self):
        historial = []
        llamada = RespuestaFalsa(llamadas=[("calcular", {"columna": "ingresos", "operacion": "suma"})])
        with modelo_falso(llamada, RespuestaFalsa("Son 210")):
            respuesta = agente.preguntar("¿Ingresos totales?", historial=historial)
        self.assertEqual(respuesta, "Son 210")
        self.assertEqual(len(historial), 4)  # pregunta, llamada, resultado de la herramienta, respuesta
        resultado = historial[2].parts[0].function_response.response
        self.assertEqual(resultado, {"resultado": 210.0})

    def test_herramienta_inexistente_devuelve_error_al_modelo(self):
        historial = []
        with modelo_falso(RespuestaFalsa(llamadas=[("hackear", {})]), RespuestaFalsa("No puedo")):
            agente.preguntar("hazlo", historial=historial)
        resultado = historial[2].parts[0].function_response.response
        self.assertIn("no existe", resultado["error"])

    def test_argumentos_invalidos_devuelven_error_al_modelo(self):
        historial = []
        llamada = RespuestaFalsa(llamadas=[("calcular", {"columna": "ingresos", "operacion": "suma",
                                                         "inventado": 1})])
        with modelo_falso(llamada, RespuestaFalsa("Vale")):
            agente.preguntar("x", historial=historial)
        resultado = historial[2].parts[0].function_response.response
        self.assertIn("Argumentos no válidos", resultado["error"])

    def test_error_de_herramienta_llega_al_modelo_como_dato(self):
        historial = []
        llamada = RespuestaFalsa(llamadas=[("calcular", {"columna": "beneficio", "operacion": "suma"})])
        with modelo_falso(llamada, RespuestaFalsa("Esa columna no existe")):
            agente.preguntar("x", historial=historial)
        self.assertIn("error", historial[2].parts[0].function_response.response)

    def test_limite_de_pasos(self):
        bucle = mock.patch.object(agente, "llamar_modelo",
                                  side_effect=lambda *a, **k: RespuestaFalsa(llamadas=[("describir_datos", {})]))
        with bucle as falso:
            respuesta = agente.preguntar("x", historial=[])
        self.assertIn("máximo de pasos", respuesta)
        self.assertEqual(falso.call_count, agente.MAX_PASOS)

    def test_memoria_entre_preguntas(self):
        historial = []
        with modelo_falso(RespuestaFalsa("uno"), RespuestaFalsa("dos")):
            agente.preguntar("primera", historial=historial)
            agente.preguntar("segunda", historial=historial)
        self.assertEqual(len(historial), 4)

    def test_sin_historial_cada_pregunta_empieza_de_cero(self):
        # Se mide el tamaño de lo que recibe el modelo en el momento de cada llamada, porque
        # la lista se sigue ampliando después
        recibidos = []
        respuestas = iter([RespuestaFalsa("uno"), RespuestaFalsa("dos")])

        def falso(contents, config):
            recibidos.append(len(contents))
            return next(respuestas)

        with mock.patch.object(agente, "llamar_modelo", side_effect=falso):
            agente.preguntar("primera")
            agente.preguntar("segunda")
        self.assertEqual(recibidos, [1, 1])  # cada una solo ve su propia pregunta

    def test_respuesta_sin_candidatos_no_rompe_y_no_ensucia_el_historial(self):
        # Gemini puede devolver una respuesta vacía (por ejemplo, bloqueada por seguridad)
        vacia = RespuestaFalsa("x")
        vacia.candidates, vacia.function_calls, vacia.text = [], [], None
        historial = []
        with modelo_falso(RespuestaFalsa("uno"), vacia, RespuestaFalsa("tres")):
            agente.preguntar("primera", historial=historial)
            respuesta = agente.preguntar("segunda", historial=historial)
            self.assertIn("no ha devuelto ninguna respuesta", respuesta)
            self.assertEqual(len(historial), 2)  # solo queda el primer turno, no la pregunta huérfana
            self.assertEqual(agente.preguntar("tercera", historial=historial), "tres")
        self.assertEqual(len(historial), 4)

    def test_respuesta_sin_contenido(self):
        sin_partes = RespuestaFalsa()
        sin_partes.text = None
        historial = []
        with modelo_falso(sin_partes):
            self.assertIn("no ha devuelto", agente.preguntar("x", historial=historial))
        self.assertEqual(historial, [])

    def test_herramienta_sin_argumentos_con_args_none(self):
        # Una herramienta sin parámetros puede llegar con args=None
        llamada = RespuestaFalsa(llamadas=[("describir_datos", None)])
        historial = []
        with modelo_falso(llamada, RespuestaFalsa("Tiene 6 filas")):
            agente.preguntar("¿Qué hay?", historial=historial)
        self.assertEqual(historial[2].parts[0].function_response.response["filas"], 6)


if __name__ == "__main__":
    unittest.main()
