import re
import time
import sys
import pandas as pd
from agente import preguntar, INSTRUCCIONES

df = pd.read_csv("ventas.csv", parse_dates=["fecha"])


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


def alternativas(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


INSTRUCCIONES_B = "Responde en español."  # versión mínima, para comparar


def normalizar(texto: str) -> str:
    texto = texto.lower()
    return re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", texto)  # quita separadores de miles


def evaluar(instrucciones: str, etiqueta: str) -> list[dict]:
    filas = []
    for pregunta, esperados in PREGUNTAS:
        try:
            respuesta = preguntar(pregunta, instrucciones)
        except Exception as e:
            respuesta = f"ERROR: {e}"
        if respuesta.startswith("ERROR"):
            ok = None  # no evaluable
        else:
            norm = normalizar(respuesta)
            ok = all(any(str(a).lower() in norm for a in alternativas(e)) for e in esperados)
        filas.append({"version": etiqueta, "pregunta": pregunta, "esperado": esperados,
                      "ok": ok, "respuesta": respuesta, "inventa_moneda": ("€" in respuesta) or ("euro" in respuesta.lower())})
    validas = [f for f in filas if f["ok"] is not None]
    aciertos = sum(f["ok"] for f in validas)
    print(f"\n>>> {etiqueta}: {aciertos}/{len(validas)} válidas "
          f"({len(filas) - len(validas)} con error)\n")
    print(f">>> {etiqueta}: respuestas que inventan moneda: {sum(f['inventa_moneda'] for f in filas)}")
    return filas


if __name__ == "__main__":
    resultados = evaluar(INSTRUCCIONES, "A (completas)")
    if "--completa" in sys.argv:  # python evaluacion.py --completa
        resultados += evaluar(INSTRUCCIONES_B, "B (mínimas)")
    pd.DataFrame(resultados).to_csv("resultados.csv", index=False)
    print("Guardado en resultados.csv")

    