"""Tests de las herramientas de análisis. No usan la API de Gemini: son deterministas.

Se ejecutan desde la carpeta raíz del proyecto:
    python -m unittest discover -s tests -t .
(necesitan que exista ventas.csv, porque herramientas.py lo carga al importarse:
 ejecuta antes `python generar_datos.py`)
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import herramientas as h


def datos_prueba() -> pd.DataFrame:
    """Dataset pequeño con resultados fáciles de comprobar a mano."""
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2025-01-10", "2025-01-20", "2025-02-05",
                                 "2025-02-15", "2025-03-01", "2025-03-02"]),
        "ciudad": ["Madrid", "Madrid", "Sevilla", "Sevilla", "Madrid", "Valencia"],
        "producto": ["A", "B", "A", "B", "A", "C"],
        "unidades": [1, 2, 3, 4, 5, 6],
        "ingresos": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


class BaseHerramientas(unittest.TestCase):
    """Sustituye el dataset activo por uno de prueba y lo restaura al terminar."""

    def setUp(self):
        self._df_original = h.df
        h.df = datos_prueba()

    def tearDown(self):
        h.df = self._df_original


class TestCalcular(BaseHerramientas):
    def test_suma_total(self):
        self.assertEqual(h.calcular("ingresos", "suma"), {"resultado": 210.0})

    def test_recuento(self):
        self.assertEqual(h.calcular("ingresos", "recuento"), {"resultado": 6.0})

    def test_media_por_ciudad(self):
        res = h.calcular("ingresos", "media", agrupar_por="ciudad")["resultado"]
        self.assertEqual(res, {"Madrid": 26.67, "Sevilla": 35.0, "Valencia": 60.0})

    def test_filtro_ignora_mayusculas(self):
        res = h.calcular("ingresos", "suma", filtro_columna="ciudad", filtro_valor="madrid")
        self.assertEqual(res, {"resultado": 80.0})

    def test_operacion_invalida(self):
        self.assertIn("error", h.calcular("ingresos", "raiz"))

    def test_columna_inexistente_lista_las_columnas(self):
        res = h.calcular("beneficio", "suma")
        self.assertIn("error", res)
        self.assertIn("ingresos", res["error"])

    def test_filtro_sin_valor(self):
        self.assertIn("error", h.calcular("ingresos", "suma", filtro_columna="ciudad"))

    def test_filtro_sin_filas_lista_valores_posibles(self):
        res = h.calcular("ingresos", "suma", filtro_columna="ciudad", filtro_valor="Bilbao")
        self.assertIn("error", res)
        self.assertIn("Madrid", res["error"])

    def test_columna_de_texto_devuelve_error_no_excepcion(self):
        self.assertIn("error", h.calcular("ciudad", "media"))

    def test_nan_se_convierte_en_none(self):
        # La desviación de un grupo con una sola fila es NaN, que no es JSON válido
        res = h.calcular("ingresos", "desviacion", agrupar_por="ciudad")["resultado"]
        self.assertIsNone(res["Valencia"])

    def test_nan_sin_agrupar(self):
        res = h.calcular("ingresos", "desviacion", filtro_columna="ciudad", filtro_valor="Valencia")
        self.assertEqual(res, {"resultado": None})


class TestCalcularPorMes(BaseHerramientas):
    def test_suma_por_mes(self):
        res = h.calcular_por_mes("ingresos", "suma")["resultado"]
        self.assertEqual(res, {"2025-01": 30.0, "2025-02": 70.0, "2025-03": 110.0})

    def test_con_filtro(self):
        res = h.calcular_por_mes("ingresos", "suma", filtro_columna="ciudad",
                                 filtro_valor="Sevilla")["resultado"]
        self.assertEqual(res, {"2025-02": 70.0})

    def test_sin_columna_de_fecha(self):
        h.df = h.df.drop(columns=["fecha"])
        self.assertIn("error", h.calcular_por_mes("ingresos", "suma"))

    def test_columna_que_no_es_fecha(self):
        self.assertIn("error", h.calcular_por_mes("ingresos", "suma", columna_fecha="ciudad"))


class TestRanking(BaseHerramientas):
    def test_mayores(self):
        res = h.ranking("ingresos", "suma", "ciudad", n=2)["resultado"]
        self.assertEqual(list(res.items()), [("Madrid", 80.0), ("Sevilla", 70.0)])

    def test_menores(self):
        res = h.ranking("ingresos", "suma", "ciudad", n=1, orden="menor")["resultado"]
        self.assertEqual(res, {"Valencia": 60.0})

    def test_orden_invalido(self):
        self.assertIn("error", h.ranking("ingresos", "suma", "ciudad", orden="primero"))

    def test_n_debe_ser_positivo(self):
        for n in (0, -1):
            with self.subTest(n=n):
                self.assertIn("error", h.ranking("ingresos", "suma", "ciudad", n=n))

    def test_n_no_numerico(self):
        self.assertIn("error", h.ranking("ingresos", "suma", "ciudad", n="tres"))

    def test_n_decimal_del_modelo(self):
        # Gemini envía los números como float (2.0): se acepta si es entero
        res = h.ranking("ingresos", "suma", "ciudad", n=2.0)["resultado"]
        self.assertEqual(len(res), 2)


class TestPorcentaje(BaseHerramientas):
    def test_total_cero_no_devuelve_nan(self):
        h.df = pd.DataFrame({"g": ["a", "b"], "v": [0.0, 0.0]})
        res = h.porcentaje_del_total("v", "g")["porcentajes"]
        self.assertEqual(res, {"a": None, "b": None})

    def test_porcentajes_por_ciudad(self):
        res = h.porcentaje_del_total("ingresos", "ciudad")["porcentajes"]
        self.assertEqual(res, {"Madrid": 38.1, "Sevilla": 33.3, "Valencia": 28.6})

    def test_suman_aproximadamente_100(self):
        res = h.porcentaje_del_total("ingresos", "producto")["porcentajes"]
        self.assertAlmostEqual(sum(res.values()), 100.0, delta=0.3)


class TestResumenEstadistico(BaseHerramientas):
    def test_coincide_con_pandas(self):
        res = h.resumen_estadistico("ingresos")["resultado"]
        serie = h.df["ingresos"]
        self.assertEqual(res["recuento"], 6)
        self.assertEqual(res["media"], round(serie.mean(), 2))
        self.assertEqual(res["desviacion"], round(serie.std(), 2))  # muestral (ddof=1)
        self.assertEqual(res["q1"], 22.5)
        self.assertEqual(res["mediana"], 35.0)
        self.assertEqual(res["q3"], 47.5)

    def test_por_grupos(self):
        res = h.resumen_estadistico("ingresos", agrupar_por="ciudad")["resultado"]
        self.assertEqual(res["Madrid"]["recuento"], 3)
        self.assertIsNone(res["Valencia"]["desviacion"])  # un solo valor: sin desviación

    def test_columna_no_numerica(self):
        self.assertIn("error", h.resumen_estadistico("ciudad"))


class TestCorrelacion(BaseHerramientas):
    def test_pearson_perfecta(self):
        res = h.correlacion("unidades", "ingresos")
        self.assertEqual(res["correlacion"], 1.0)
        self.assertEqual(res["n"], 6)

    def test_spearman(self):
        # Antes fallaba con "No module named 'scipy'", que pandas exige para este método
        self.assertEqual(h.correlacion("unidades", "ingresos", metodo="spearman")["correlacion"], 1.0)

    def test_spearman_con_relacion_monotona_no_lineal(self):
        h.df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25000]})
        self.assertEqual(h.correlacion("x", "y", metodo="spearman")["correlacion"], 1.0)
        self.assertLess(h.correlacion("x", "y", metodo="pearson")["correlacion"], 1.0)

    def test_spearman_coincide_con_scipy_si_esta_instalado(self):
        try:
            from scipy.stats import spearmanr
        except ImportError:
            self.skipTest("scipy no está instalado")
        h.df = pd.DataFrame({"x": [3, 1, 4, 1, 5, 9, 2, 6], "y": [2, 7, 1, 8, 2, 8, 1, 8]})
        esperado = round(float(spearmanr(h.df["x"], h.df["y"])[0]), 2)
        self.assertEqual(h.correlacion("x", "y", metodo="spearman")["correlacion"], esperado)

    def test_con_filtro(self):
        # Solo las filas de Madrid: unidades 1, 2, 5 e ingresos 10, 20, 50 (relación perfecta)
        res = h.correlacion("unidades", "ingresos", filtro_columna="ciudad", filtro_valor="Madrid")
        self.assertEqual(res["correlacion"], 1.0)
        self.assertEqual(res["n"], 3)

    def test_filtro_sin_filas(self):
        res = h.correlacion("unidades", "ingresos", filtro_columna="ciudad", filtro_valor="Bilbao")
        self.assertIn("error", res)

    def test_metodo_invalido(self):
        self.assertIn("error", h.correlacion("unidades", "ingresos", metodo="kendall"))

    def test_columna_no_numerica(self):
        self.assertIn("error", h.correlacion("ciudad", "ingresos"))


class TestGrafico(BaseHerramientas):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self._raiz, self._carpeta = h.RAIZ, h.CARPETA_GRAFICOS
        h.RAIZ = Path(self._tmp.name)
        h.CARPETA_GRAFICOS = h.RAIZ / "graficos"

    def tearDown(self):
        h.RAIZ, h.CARPETA_GRAFICOS = self._raiz, self._carpeta
        self._tmp.cleanup()
        super().tearDown()

    def test_crea_el_archivo(self):
        res = h.grafico("barras", "ciudad", "ingresos", archivo="por_ciudad.png")
        self.assertEqual(res, {"archivo": "graficos/por_ciudad.png"})
        self.assertTrue((h.CARPETA_GRAFICOS / "por_ciudad.png").is_file())

    def test_no_sale_de_la_carpeta_de_graficos(self):
        h.grafico("lineas", "ciudad", "ingresos", archivo="../../fuera.png")
        self.assertTrue((h.CARPETA_GRAFICOS / "fuera.png").is_file())
        self.assertFalse((h.RAIZ / "fuera.png").exists())

    def test_tipo_invalido(self):
        self.assertIn("error", h.grafico("tarta", "ciudad", "ingresos"))


class TestCargarCsv(BaseHerramientas):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.carpeta = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def test_separador_punto_y_coma_coma_decimal_y_fechas(self):
        ruta = self.carpeta / "europeo.csv"
        ruta.write_text("fecha;producto;importe\n01/02/2025;A;1,5\n02/02/2025;B;2,25\n", encoding="utf-8")
        datos = h.cargar_csv(ruta)
        self.assertTrue(pd.api.types.is_numeric_dtype(datos["importe"]))
        self.assertEqual(list(datos["importe"]), [1.5, 2.25])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(datos["fecha"]))
        self.assertEqual(datos["fecha"].iloc[0].month, 2)  # dd/mm/aaaa, no mm/dd
        self.assertIs(h.df, datos)

    def test_codificacion_latin1(self):
        ruta = self.carpeta / "latin1.csv"
        ruta.write_bytes("producto,unidades\nPantalón,3\n".encode("latin-1"))
        self.assertEqual(h.cargar_csv(ruta)["producto"].iloc[0], "Pantalón")

    def test_archivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            h.cargar_csv(self.carpeta / "no_existe.csv")

    def test_solo_cabecera(self):
        ruta = self.carpeta / "vacio.csv"
        ruta.write_text("a;b\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            h.cargar_csv(ruta)


if __name__ == "__main__":
    unittest.main()
