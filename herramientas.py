import pandas as pd
import matplotlib
matplotlib.use("Agg")  # dibuja en archivos, sin abrir ventanas
import matplotlib.pyplot as plt

df = pd.read_csv("ventas.csv", parse_dates=["fecha"])

OPERACIONES = {"suma": "sum", "media": "mean", "mediana": "median", "maximo": "max",
               "minimo": "min", "recuento": "count", "desviacion": "std", "varianza": "var"}


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


def ranking(columna: str, operacion: str, agrupar_por: str, n: int = 3, orden: str = "mayor",
            filtro_columna: str | None = None, filtro_valor: str | None = None) -> dict:
    """Devuelve los n grupos de agrupar_por con mayor (orden='mayor') o menor (orden='menor')
    valor de la operación (suma, media, maximo, minimo o recuento) sobre la columna.
    Admite el mismo filtro opcional que calcular."""
    if operacion not in OPERACIONES:
        return {"error": f"Operación no válida. Usa una de: {list(OPERACIONES)}"}
    if orden not in ("mayor", "menor"):
        return {"error": "orden debe ser 'mayor' o 'menor'"}
    error = _error_columnas(columna, agrupar_por)
    if error:
        return {"error": error}
    datos, error = _filtrar(filtro_columna, filtro_valor)
    if error:
        return {"error": error}
    try:
        res = datos.groupby(agrupar_por)[columna].agg(OPERACIONES[operacion])
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


def _r(x):
    """Redondea a 2 decimales y convierte los NaN en None."""
    return None if pd.isna(x) else round(float(x), 2)


def resumen_estadistico(columna: str, agrupar_por: str | None = None,
                        filtro_columna: str | None = None, filtro_valor: str | None = None) -> dict:
    """Devuelve estadísticos descriptivos de una columna numérica: recuento, media, desviación típica
    (muestral), mínimo, Q1, mediana, Q3 y máximo. Opcional: por grupos (agrupar_por) y/o con filtro."""
    error = _error_columnas(columna, agrupar_por)
    if error:
        return {"error": error}
    datos, error = _filtrar(filtro_columna, filtro_valor)
    if error:
        return {"error": error}
    if not pd.api.types.is_numeric_dtype(datos[columna]):
        return {"error": f"La columna '{columna}' no es numérica"}

    def resumen(serie):
        d = serie.describe()
        return {"recuento": int(d["count"]), "media": _r(d["mean"]), "desviacion": _r(d["std"]),
                "minimo": _r(d["min"]), "q1": _r(d["25%"]), "mediana": _r(d["50%"]),
                "q3": _r(d["75%"]), "maximo": _r(d["max"])}

    try:
        if agrupar_por:
            return {"resultado": {str(k): resumen(g[columna]) for k, g in datos.groupby(agrupar_por)}}
        return {"resultado": resumen(datos[columna])}
    except Exception as e:
        return {"error": str(e)}


def correlacion(columna_a: str, columna_b: str, metodo: str = "pearson") -> dict:
    """Calcula el coeficiente de correlación entre dos columnas numéricas (metodo: 'pearson' o 'spearman')."""
    if metodo not in ("pearson", "spearman"):
        return {"error": "metodo debe ser 'pearson' o 'spearman'"}
    error = _error_columnas(columna_a, columna_b)
    if error:
        return {"error": error}
    if not (pd.api.types.is_numeric_dtype(df[columna_a]) and pd.api.types.is_numeric_dtype(df[columna_b])):
        return {"error": "Las dos columnas deben ser numéricas"}
    try:
        return {"correlacion": _r(df[columna_a].corr(df[columna_b], method=metodo)),
                "metodo": metodo, "n": int(df[[columna_a, columna_b]].dropna().shape[0])}
    except Exception as e:
        return {"error": str(e)}