"""Tests de la parte estadística y del flujo de evaluacion.py, con un agente falso (sin API).

Se ejecutan desde la carpeta raíz del proyecto:
    python -m unittest discover -s tests -t .
"""
import os

os.environ["GEMINI_API_KEY"] = "clave-falsa-para-tests"  # antes de importar: nunca se usa la clave real

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import evaluacion as ev


class TestIntervaloWilson(unittest.TestCase):
    def test_mitad(self):
        bajo, alto = ev.intervalo_wilson(5, 10)
        self.assertAlmostEqual(bajo, 0.2366, places=3)
        self.assertAlmostEqual(alto, 0.7634, places=3)

    def test_ningun_acierto(self):
        bajo, alto = ev.intervalo_wilson(0, 10)
        self.assertEqual(bajo, 0.0)
        self.assertAlmostEqual(alto, 0.2775, places=3)

    def test_todos_aciertos(self):
        bajo, alto = ev.intervalo_wilson(10, 10)
        self.assertAlmostEqual(bajo, 0.7225, places=3)
        self.assertEqual(alto, 1.0)

    def test_sin_datos(self):
        bajo, alto = ev.intervalo_wilson(0, 0)
        self.assertTrue(bajo != bajo and alto != alto)  # NaN

    def test_mas_datos_intervalo_mas_estrecho(self):
        a = ev.intervalo_wilson(5, 10)
        b = ev.intervalo_wilson(50, 100)
        self.assertLess(b[1] - b[0], a[1] - a[0])


class TestBootstrap(unittest.TestCase):
    def test_tasas_iguales(self):
        self.assertEqual(ev.intervalo_bootstrap([1.0] * 10), (1.0, 1.0))

    def test_contiene_la_media(self):
        tasas = [1.0] * 8 + [0.0] * 2
        bajo, alto = ev.intervalo_bootstrap(tasas)
        self.assertLessEqual(bajo, 0.8)
        self.assertGreaterEqual(alto, 0.8)

    def test_es_reproducible(self):
        tasas = [1.0, 0.0, 0.5, 1.0]
        self.assertEqual(ev.intervalo_bootstrap(tasas), ev.intervalo_bootstrap(tasas))

    def test_una_sola_pregunta(self):
        bajo, alto = ev.intervalo_bootstrap([1.0])
        self.assertTrue(bajo != bajo and alto != alto)


class TestPruebaSignos(unittest.TestCase):
    def test_a_siempre_mejor(self):
        p, gana_a, gana_b = ev.prueba_signos([1.0] * 5, [0.0] * 5)
        self.assertEqual((gana_a, gana_b), (5, 0))
        self.assertAlmostEqual(p, 2 / 32)

    def test_sin_diferencias(self):
        self.assertEqual(ev.prueba_signos([1.0, 0.0], [1.0, 0.0]), (1.0, 0, 0))

    def test_mixto(self):
        p, gana_a, gana_b = ev.prueba_signos([1, 1, 1, 0], [0, 0, 0, 1])
        self.assertEqual((gana_a, gana_b), (3, 1))
        self.assertAlmostEqual(p, 2 * (1 + 4) / 16)

    def test_empates_no_cuentan(self):
        p, gana_a, gana_b = ev.prueba_signos([1, 1, 1], [1, 1, 0])
        self.assertEqual((gana_a, gana_b), (1, 0))
        self.assertAlmostEqual(p, 1.0)


class TestComprobaciones(unittest.TestCase):
    def test_rechaza(self):
        self.assertTrue(ev.rechaza("No existe ninguna columna de beneficio en el dataset."))
        self.assertTrue(ev.rechaza("Lo siento, no dispongo de esa información."))
        self.assertFalse(ev.rechaza("El beneficio total es de 5.000."))

    def test_contiene_no_confunde_cifras(self):
        self.assertTrue(ev.contiene(ev.normalizar("Hay 300 filas"), "300"))
        self.assertFalse(ev.contiene(ev.normalizar("Hay 3000 filas"), "300"))

    def test_normalizar_quita_separador_de_miles(self):
        self.assertTrue(ev.contiene(ev.normalizar("Total: 65.361 €"), "65361"))

    def test_tasas_ignoran_errores_de_api(self):
        filas = [{"pregunta": "p", "ok": True}, {"pregunta": "p", "ok": False},
                 {"pregunta": "p", "ok": None}, {"pregunta": "q", "ok": None}]
        self.assertEqual(ev.tasas_por_pregunta(filas), {"p": 0.5})  # q no tiene ninguna repetición válida


class TestFlujoDeEvaluacion(unittest.TestCase):
    """Ejecuta main() con un agente falso que contesta bien a todo."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._raiz = ev.RAIZ
        ev.RAIZ = Path(self._tmp.name)  # que no toque el resultados.csv real

    def tearDown(self):
        ev.RAIZ = self._raiz
        self._tmp.cleanup()

    @staticmethod
    def agente_falso(fallar_en=None):
        correctas = {p: " ".join(str(alternativas[0] if isinstance(alternativas, list) else alternativas)
                                 for alternativas in esperados)
                     for p, esperados in ev.PREGUNTAS}

        def falso(pregunta, instrucciones=None, historial=None):
            if fallar_en and pregunta == fallar_en:
                raise RuntimeError("503 simulado")
            if pregunta in correctas:
                return f"Respuesta: {correctas[pregunta]}"
            return "No existe esa información en el dataset."
        return falso

    def ejecutar(self, argumentos, **kw):
        with mock.patch.object(ev, "preguntar", side_effect=self.agente_falso(**kw)):
            ev.main(argumentos)
        return pd.read_csv(ev.RAIZ / "resultados.csv"), (ev.RAIZ / "resumen_evaluacion.md").read_text("utf-8")

    def test_una_version_una_repeticion(self):
        filas, texto = self.ejecutar([])
        self.assertEqual(len(filas), len(ev.PREGUNTAS) + len(ev.PREGUNTAS_IMPOSIBLES))
        self.assertTrue(filas.ok.all())
        self.assertIn("100.0 %", texto)

    def test_repeticiones_y_comparacion(self):
        filas, texto = self.ejecutar(["--completa", "--repeticiones", "2"])
        total = (len(ev.PREGUNTAS) + len(ev.PREGUNTAS_IMPOSIBLES)) * 2 * 2
        self.assertEqual(len(filas), total)
        self.assertEqual(sorted(filas.repeticion.unique()), [1, 2])
        self.assertIn("contraste de signos", texto)

    def test_sin_imposibles(self):
        filas, texto = self.ejecutar(["--sin-imposibles"])
        self.assertEqual(len(filas), len(ev.PREGUNTAS))
        self.assertNotIn("## Preguntas sin respuesta posible", texto)

    def test_un_error_de_api_no_cuenta_como_fallo(self):
        pregunta = ev.PREGUNTAS[0][0]
        filas, texto = self.ejecutar(["--sin-imposibles"], fallar_en=pregunta)
        self.assertEqual(filas.ok.isna().sum(), 1)
        self.assertEqual(int((filas.ok == False).sum()), 0)  # noqa: E712
        self.assertIn("| 23 | 23 |", texto)  # 23 válidas, 23 aciertos

    def test_repeticiones_invalidas(self):
        with self.assertRaises(SystemExit):
            ev.main(["--repeticiones", "0"])


if __name__ == "__main__":
    unittest.main()
