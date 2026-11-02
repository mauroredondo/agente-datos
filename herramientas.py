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


def _error_columnas(*columnas):
    for c in columnas:
        if c is not None and c not in df.columns:
            return f"La columna '{c}' no existe. Columnas: {list(df.columns)}"
    return None


def _filtrar(filtro_columna, filtro_valor):
    """Devuelve (datos, error). Sin filtro, devuelve todos los datos."""
    if not filtro_columna:
        return df, None
    error = _error_columnas(filtro_columna)
    if error:
        return None, error
    if filtro_valor is None:
        return None, "Si indicas filtro_columna, indica también filtro_valor"
    datos = df[df[filtro_columna].astype(str).str.lower() == str(filtro_valor).lower()]
    if datos.empty:
        validos = sorted(df[filtro_columna].astype(str).unique())[:20]
        return None, f"No hay filas con {filtro_columna} = {filtro_valor}. Valores posibles: {validos}"
    return datos, None


def calcular(columna: str, operacion: str, agrupar_por: str | None = None,
             filtro_columna: str | None = None, filtro_valor: str | None = None) -> dict:
    """Calcula suma, media, maximo, minimo o recuento de una columna. Opcional: agrupar por otra
    columna y/o filtrar las filas donde filtro_columna sea igual a filtro_valor
    (por ejemplo filtro_columna='ciudad', filtro_valor='Madrid')."""
    if operacion not in OPERACIONES:
        return {"error": f"Operación no válida. Usa una de: {list(OPERACIONES)}"}
    error = _error_columnas(columna, agrupar_por)
    if error:
        return {"error": error}
    datos, error = _filtrar(filtro_columna, filtro_valor)
    if error:
        return {"error": error}
    try:
        if agrupar_por:
            res = datos.groupby(agrupar_por)[columna].agg(OPERACIONES[operacion])
            return {"resultado": {str(k): round(float(v), 2) for k, v in res.items()}}
        return {"resultado": round(float(datos[columna].agg(OPERACIONES[operacion])), 2)}
    except Exception as e:
        return {"error": str(e)}


def calcular_por_mes(columna: str, operacion: str,
                     filtro_columna: str | None = None, filtro_valor: str | None = None) -> dict:
    """Calcula suma, media, maximo, minimo o recuento de una columna agrupado por mes (formato AAAA-MM).
    Admite el mismo filtro opcional que calcular."""
    if operacion not in OPERACIONES:
        return {"error": f"Operación no válida. Usa una de: {list(OPERACIONES)}"}
    error = _error_columnas(columna)
    if error:
        return {"error": error}
    datos, error = _filtrar(filtro_columna, filtro_valor)
    if error:
        return {"error": error}
    try:
        res = datos.groupby(datos["fecha"].dt.to_period("M"))[columna].agg(OPERACIONES[operacion])
        return {"resultado": {str(k): round(float(v), 2) for k, v in res.items()}}
    except Exception as e:
        return {"error": str(e)}


def ranking(columna: str, operacion: str, agrupar_por: str, n: int = 3, orden: str = "mayor") -> dict:
    """Devuelve los n grupos de agrupar_por con mayor (orden='mayor') o menor (orden='menor')
    valor de la operación (suma, media, maximo, minimo o recuento) sobre la columna."""
    if operacion not in OPERACIONES:
        return {"error": f"Operación no válida. Usa una de: {list(OPERACIONES)}"}
    if orden not in ("mayor", "menor"):
        return {"error": "orden debe ser 'mayor' o 'menor'"}
    error = _error_columnas(columna, agrupar_por)
    if error:
        return {"error": error}
    try:
        res = df.groupby(agrupar_por)[columna].agg(OPERACIONES[operacion])
        res = res.sort_values(ascending=(orden == "menor")).head(int(n))
        return {"resultado": {str(k): round(float(v), 2) for k, v in res.items()}}
    except Exception as e:
        return {"error": str(e)}


def porcentaje_del_total(columna: str, agrupar_por: str) -> dict:
    """Calcula qué porcentaje del total (suma) de una columna representa cada grupo de agrupar_por."""
    error = _error_columnas(columna, agrupar_por)
    if error:
        return {"error": error}
    try:
        res = df.groupby(agrupar_por)[columna].sum()
        pct = (res / res.sum() * 100).round(1)
        return {"porcentajes": {str(k): float(v) for k, v in pct.items()}}
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