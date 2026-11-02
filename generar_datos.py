import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semilla fija: siempre sale lo mismo
n = 300

productos = {
    "Camiseta": ("Ropa", 15), "Pantalón": ("Ropa", 35),
    "Zapatillas": ("Calzado", 60), "Botas": ("Calzado", 80),
    "Mochila": ("Accesorios", 40), "Gorra": ("Accesorios", 12),
}
ciudades = ["Barcelona", "Madrid", "Valencia", "Sevilla"]

prod = rng.choice(list(productos), n)
df = pd.DataFrame({
    "fecha": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "producto": prod,
    "categoria": [productos[p][0] for p in prod],
    "ciudad": rng.choice(ciudades, n),
    "unidades": rng.integers(1, 10, n),
})
df["precio_unitario"] = [productos[p][1] for p in prod]
df["ingresos"] = df["unidades"] * df["precio_unitario"]
df = df.sort_values("fecha")

df.to_csv("ventas.csv", index=False)
print(df.head())