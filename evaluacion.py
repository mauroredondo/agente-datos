import re
import time
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
            ok = all(str(e).lower() in norm for e in esperados)
        filas.append({"version": etiqueta, "pregunta": pregunta, "esperado": esperados,
                      "ok": ok, "respuesta": respuesta})
    validas = [f for f in filas if f["ok"] is not None]
    aciertos = sum(f["ok"] for f in validas)
    print(f"\n>>> {etiqueta}: {aciertos}/{len(validas)} válidas "
          f"({len(filas) - len(validas)} con error)\n")
    return filas


if __name__ == "__main__":
    resultados = evaluar(INSTRUCCIONES, "A (completas)") + evaluar(INSTRUCCIONES_B, "B (mínimas)")
    pd.DataFrame(resultados).to_csv("resultados.csv", index=False)
    print("Guardado en resultados.csv")