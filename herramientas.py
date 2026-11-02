import pandas as pd
import matplotlib
matplotlib.use("Agg")  # dibuja en archivos, sin abrir ventanas
import matplotlib.pyplot as plt

df = pd.read_csv("ventas.csv", parse_dates=["fecha"])

OPERACIONES = {"suma": "sum", "media": "mean", "maximo": "max", "minimo": "min", "recuento": "count"}


def describir_datos() -> dict:
    """Devuelve el número de filas, las columnas con su tipo y 3 filas de ejemplo."""
    return {
        "filas": len(df),
        "columnas": {c: str(t) for c, t in df.dtypes.items()},
        "ejemplo": df.head(3).astype(str).to_dict(orient="records"),
    }


def calcular(columna: str, operacion: str, agrupar_por: str | None = None) -> dict:
    """Calcula suma, media, maximo, minimo o recuento de una columna, opcionalmente agrupando por otra."""
    if columna not in df.columns:
        return {"error": f"La columna '{columna}' no existe. Columnas: {list(df.columns)}"}
    if operacion not in OPERACIONES:
        return {"error": f"Operación no válida. Usa una de: {list(OPERACIONES)}"}
    if agrupar_por and agrupar_por not in df.columns:
        return {"error": f"La columna '{agrupar_por}' no existe. Columnas: {list(df.columns)}"}
    try:
        if agrupar_por:
            res = df.groupby(agrupar_por)[columna].agg(OPERACIONES[operacion])
            return {"resultado": {str(k): round(float(v), 2) for k, v in res.items()}}
        return {"resultado": round(float(df[columna].agg(OPERACIONES[operacion])), 2)}
    except Exception as e:
        return {"error": str(e)}


def grafico(tipo: str, x: str, y: str, archivo: str = "grafico.png") -> dict:
    """Dibuja un gráfico de 'barras' o 'lineas' con la suma de y agrupada por x y lo guarda como imagen."""
    if x not in df.columns or y not in df.columns:
        return {"error": f"Columnas válidas: {list(df.columns)}"}
    if tipo not in ("barras", "lineas"):
        return {"error": "El tipo debe ser 'barras' o 'lineas'"}
    try:
        datos = df.groupby(x)[y].sum()
        fig, ax = plt.subplots()
        datos.plot(kind="bar" if tipo == "barras" else "line", ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        fig.savefig(archivo)
        plt.close(fig)
        return {"archivo": archivo}
    except Exception as e:
        return {"error": str(e)}