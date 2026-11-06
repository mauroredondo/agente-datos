import argparse
import math
import random
import re
from datetime import date
from pathlib import Path

import pandas as pd
from agente import preguntar, INSTRUCCIONES, MODEL

RAIZ = Path(__file__).parent
df = pd.read_csv(RAIZ / "ventas.csv", parse_dates=["fecha"])


def num(x):
    return str(int(round(x)))


PREGUNTAS = [
    ("¿Cuántas filas tiene el dataset?", [num(len(df))]),
    ("¿Cuáles son los ingresos totales?", [num(df.ingresos.sum())]),
    ("¿Qué ciudad tiene más ingresos?", [df.groupby("ciudad").ingresos.sum().idxmax()]),
    ("¿Qué ciudad tiene menos ingresos?", [df.groupby("ciudad").ingresos.sum().idxmin()]),
    ("¿Cuántas unidades se vendieron en total?", [num(df.unidades.sum())]),
    ("¿Qué producto vendió más unidades en total?", [df.groupby("producto").unidades.sum().idxmax()]),
    ("¿Qué categoría tiene mayores ingresos?", [df.groupby("categoria").ingresos.sum().idxmax()]),
    ("¿Cuáles son los ingresos totales de Barcelona?", [num(df[df.ciudad == "Barcelona"].ingresos.sum())]),
    ("¿Cuál es el mayor número de unidades en una sola venta?", [num(df.unidades.max())]),
    ("¿Cuántas ventas hay de la categoría Calzado?", [num((df.categoria == "Calzado").sum())]),
]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
mes_top = df.groupby(df.fecha.dt.to_period("M")).ingresos.sum().idxmax()
mes_zap = (df[df.producto == "Zapatillas"].groupby(df.fecha.dt.to_period("M")).unidades.sum().idxmax())

PREGUNTAS += [
    ("¿Cuántas unidades se vendieron en Madrid?", [num(df[df.ciudad == "Madrid"].unidades.sum())]),
    ("¿Qué producto genera más ingresos en Sevilla?",
     [df[df.ciudad == "Sevilla"].groupby("producto").ingresos.sum().idxmax()]),
    ("¿Cuál es el mes con más ingresos?", [[str(mes_top), MESES[mes_top.month - 1]]]),
    ("Dame el top 3 de productos por ingresos.",
     list(df.groupby("producto").ingresos.sum().nlargest(3).index)),
    ("¿Qué porcentaje de los ingresos totales supone Calzado? Redondea a un entero.",
     [num(df[df.categoria == "Calzado"].ingresos.sum() / df.ingresos.sum() * 100)]),
    ("¿Cuáles son los 2 productos con menos unidades vendidas?",
     list(df.groupby("producto").unidades.sum().nsmallest(2).index)),
    ("¿Cuántas ventas hay en Valencia?", [num((df.ciudad == "Valencia").sum())]),
    ("¿En qué mes se vendieron más unidades de Zapatillas?", [[str(mes_zap), MESES[mes_zap.month - 1]]]),
]


def dec(x, k=1):
    s = f"{round(x, k):.{k}f}"
    return [s, s.replace(".", ",")]  # acepta 112.5 y 112,5


PREGUNTAS += [
    ("¿Cuál es la mediana de los ingresos por venta? Redondea a un decimal.", [dec(df.ingresos.median())]),
    ("¿Cuál es la desviación típica de las unidades por venta? Redondea a un decimal.", [dec(df.unidades.std())]),
    ("¿Cuál es la media de ingresos por venta en Sevilla? Redondea a un decimal.",
     [dec(df[df.ciudad == "Sevilla"].ingresos.mean())]),
    ("¿Qué ciudad tiene la mediana de ingresos por venta más alta?",
     [df.groupby("ciudad").ingresos.median().idxmax()]),
    ("¿Cuál es la correlación de Pearson entre unidades e ingresos? Redondea a 2 decimales.",
     [dec(df.unidades.corr(df.ingresos), 2)]),
    ("¿Cuál es el tercer cuartil (Q3) de los ingresos por venta? Redondea a un decimal.",
     [dec(df.ingresos.quantile(0.75))]),
]

# Preguntas que NO se pueden responder con las columnas del dataset (fecha, producto, categoria,
# ciudad, unidades, precio_unitario, ingresos). Lo correcto es decir que no hay datos; dar una
# cifra es una alucinación.
PREGUNTAS_IMPOSIBLES = [
    "¿Cuál es el beneficio total de las ventas?",
    "¿Cuántas devoluciones hubo en Madrid?",
    "¿Cuál es la edad media de los clientes?",
    "¿Cuál es el stock actual de Gorras?",
    "¿Qué valoración media dieron los clientes a las Botas?",
]

# Heurística: una respuesta "rechaza" la pregunta si admite que no hay datos. Es una comprobación
# por texto, así que puede equivocarse en las dos direcciones; los resultados deben revisarse.
FRASES_RECHAZO = (
    "no existe", "no existen", "no hay", "no dispone", "no dispongo", "no tiene", "no tengo",
    "no contiene", "no incluye", "no cuenta con", "no figura", "no puedo", "no se puede",
    "no es posible", "no está disponible", "no se encuentra", "carece",
)


def alternativas(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


INSTRUCCIONES_B = "Responde en español."  # versión mínima, para comparar


def normalizar(texto: str) -> str:
    texto = texto.lower()
    return re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", texto)  # quita separadores de miles


def contiene(norm: str, esperado) -> bool:
    """Busca el valor como palabra/número completo: '300' no coincide dentro de '3000'."""
    patron = rf"(?<![\w.,]){re.escape(str(esperado).lower())}(?![\w]|[.,]\d)"
    return re.search(patron, norm) is not None


def rechaza(texto: str) -> bool:
    """True si la respuesta admite que no tiene los datos para contestar."""
    t = texto.lower()
    return any(frase in t for frase in FRASES_RECHAZO)


def menciona_moneda(texto: str) -> bool:
    return "€" in texto or re.search(r"\beuros?\b", texto.lower()) is not None


# --- Estadística ---------------------------------------------------------------------------

def intervalo_wilson(aciertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción (95 % con z = 1,96).
    Supone que los n ensayos son independientes."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = aciertos / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centro - margen), min(1.0, centro + margen)


def intervalo_bootstrap(tasas: list[float], remuestreos: int = 2000, semilla: int = 0,
                        confianza: float = 0.95) -> tuple[float, float]:
    """Intervalo bootstrap (percentiles) de la tasa media de acierto, remuestreando PREGUNTAS con
    reemplazo. Tiene en cuenta que las repeticiones de una misma pregunta no son independientes."""
    n = len(tasas)
    if n < 2:
        return (float("nan"), float("nan"))
    azar = random.Random(semilla)
    medias = sorted(sum(azar.choice(tasas) for _ in range(n)) / n for _ in range(remuestreos))
    alfa = (1 - confianza) / 2
    return medias[int(alfa * remuestreos)], medias[int((1 - alfa) * remuestreos) - 1]


def prueba_signos(tasas_a: list[float], tasas_b: list[float]) -> tuple[float, int, int]:
    """Contraste exacto de signos (bilateral) para comparar dos versiones sobre las mismas preguntas.
    Solo cuentan las preguntas en las que las tasas de acierto difieren. Devuelve
    (p-valor, preguntas donde gana A, preguntas donde gana B)."""
    gana_a = sum(a > b for a, b in zip(tasas_a, tasas_b))
    gana_b = sum(b > a for a, b in zip(tasas_a, tasas_b))
    n = gana_a + gana_b
    if n == 0:
        return 1.0, 0, 0
    k = min(gana_a, gana_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, gana_a, gana_b


def tasas_por_pregunta(filas: list[dict]) -> dict[str, float]:
    """Tasa de acierto de cada pregunta entre sus repeticiones válidas (sin contar los errores de API)."""
    por_pregunta: dict[str, list[bool]] = {}
    for f in filas:
        if f["ok"] is not None:
            por_pregunta.setdefault(f["pregunta"], []).append(bool(f["ok"]))
    return {p: sum(v) / len(v) for p, v in por_pregunta.items()}


# --- Ejecución -----------------------------------------------------------------------------

def evaluar(instrucciones: str, etiqueta: str, previas: list[dict] | None = None,
            preguntas=None, tipo: str = "respuesta posible", repeticion: int = 1) -> list[dict]:
    previas = previas or []
    preguntas = PREGUNTAS if preguntas is None else preguntas
    filas = []
    for pregunta, esperados in preguntas:
        try:
            respuesta = preguntar(pregunta, instrucciones)
        except Exception as e:
            respuesta = f"ERROR: {e}"
        if respuesta.startswith("ERROR"):
            ok = None  # no evaluable
        elif tipo == "sin respuesta":
            ok = rechaza(respuesta)
        else:
            norm = normalizar(respuesta)
            ok = all(any(contiene(norm, a) for a in alternativas(e)) for e in esperados)
        filas.append({"version": etiqueta, "tipo": tipo, "repeticion": repeticion, "pregunta": pregunta,
                      "esperado": esperados if esperados is not None else "(no hay datos)",
                      "ok": ok, "respuesta": respuesta, "inventa_moneda": menciona_moneda(respuesta)})
        pd.DataFrame(previas + filas).to_csv(RAIZ / "resultados.csv", index=False)  # guarda el progreso
    validas = [f for f in filas if f["ok"] is not None]
    aciertos = sum(f["ok"] for f in validas)
    print(f"\n>>> {etiqueta} ({tipo}, repetición {repeticion}): {aciertos}/{len(validas)} válidas "
          f"({len(filas) - len(validas)} con error)\n")
    return filas


def porcentaje(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f} %"  # x != x: NaN


def resumen(filas: list[dict], repeticiones: int) -> str:
    """Resumen en Markdown de toda la evaluación, con intervalos de confianza."""
    lineas = [
        "# Resumen de la evaluación", "",
        f"- Fecha: {date.today().isoformat()}",
        f"- Modelo: `{MODEL}` (es un alias: Google puede cambiar la versión que hay detrás)",
        f"- Repeticiones de cada pregunta: {repeticiones}", "",
    ]
    versiones = list(dict.fromkeys(f["version"] for f in filas))
    for tipo, titulo in (("respuesta posible", "Preguntas con respuesta en los datos"),
                         ("sin respuesta", "Preguntas sin respuesta posible (acierto = rechaza contestar)")):
        del_tipo = [f for f in filas if f["tipo"] == tipo]
        if not del_tipo:
            continue
        lineas += [f"## {titulo}", "",
                   "| Versión | Válidas | Aciertos | Acierto | IC 95 % Wilson | IC 95 % bootstrap por preguntas "
                   "| Errores de API | Mencionan moneda |", "|---|---|---|---|---|---|---|---|"]
        tasas_version = {}
        for v in versiones:
            fv = [f for f in del_tipo if f["version"] == v]
            validas = [f for f in fv if f["ok"] is not None]
            aciertos = sum(f["ok"] for f in validas)
            n = len(validas)
            w = intervalo_wilson(aciertos, n)
            tasas = tasas_por_pregunta(fv)
            tasas_version[v] = tasas
            b = intervalo_bootstrap(list(tasas.values()))
            lineas.append(f"| {v} | {n} | {aciertos} | {porcentaje(aciertos / n if n else float('nan'))} "
                          f"| {porcentaje(w[0])} – {porcentaje(w[1])} | {porcentaje(b[0])} – {porcentaje(b[1])} "
                          f"| {len(fv) - n} | {sum(f['inventa_moneda'] for f in fv)} |")
        if len(versiones) == 2:
            comunes = [p for p in tasas_version[versiones[0]] if p in tasas_version[versiones[1]]]
            p, gana_a, gana_b = prueba_signos([tasas_version[versiones[0]][q] for q in comunes],
                                              [tasas_version[versiones[1]][q] for q in comunes])
            lineas += ["", f"Comparación {versiones[0]} frente a {versiones[1]} (contraste de signos por pregunta): "
                           f"gana la primera en {gana_a}, la segunda en {gana_b}, p = {p:.3f}."]
        lineas.append("")
    lineas += [
        "Notas: el IC de Wilson trata cada respuesta como independiente; el bootstrap remuestrea preguntas, "
        "que es la forma de reconocer que las repeticiones de una misma pregunta no lo son. Los errores de la "
        "API (sobrecarga, tiempo de espera) no cuentan como fallos. Las preguntas sin respuesta posible se "
        "puntúan por texto (heurística): conviene revisar `resultados.csv`.", ""]
    return "\n".join(lineas)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evalúa el agente con preguntas de respuesta conocida.")
    ap.add_argument("--completa", action="store_true",
                    help="evalúa también unas instrucciones mínimas y compara las dos versiones")
    ap.add_argument("--repeticiones", type=int, default=1, metavar="N",
                    help="repite cada pregunta N veces (por defecto 1); con N>1 se pueden dar intervalos fiables")
    ap.add_argument("--sin-imposibles", action="store_true",
                    help="no incluye las preguntas sin respuesta posible")
    args = ap.parse_args(argv)
    if args.repeticiones < 1:
        ap.error("--repeticiones debe ser al menos 1")

    versiones = [("A (completas)", INSTRUCCIONES)]
    if args.completa:
        versiones.append(("B (mínimas)", INSTRUCCIONES_B))

    filas: list[dict] = []
    for repeticion in range(1, args.repeticiones + 1):
        for etiqueta, instrucciones in versiones:
            filas += evaluar(instrucciones, etiqueta, previas=filas, preguntas=PREGUNTAS,
                             tipo="respuesta posible", repeticion=repeticion)
            if not args.sin_imposibles:
                filas += evaluar(instrucciones, etiqueta, previas=filas,
                                 preguntas=[(p, None) for p in PREGUNTAS_IMPOSIBLES],
                                 tipo="sin respuesta", repeticion=repeticion)
    pd.DataFrame(filas).to_csv(RAIZ / "resultados.csv", index=False)
    texto = resumen(filas, args.repeticiones)
    (RAIZ / "resumen_evaluacion.md").write_text(texto, encoding="utf-8")
    print(texto)
    print("Guardado en resultados.csv y resumen_evaluacion.md")


if __name__ == "__main__":
    main()
