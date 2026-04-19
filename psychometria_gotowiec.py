import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "data_wyniki.csv"
OUT_PATH = "psychometria_gotowiec_glm.csv"
P_THRESHOLD = 0.75

df = pd.read_csv(DATA_PATH)
df["poziom_bodzca"] = pd.to_numeric(df["poziom_bodzca"], errors="coerce")

sig = df[(df["bodziec_obecny"] == 1) & df["poziom_bodzca"].notna()].copy()
sig["y"] = pd.to_numeric(sig["odpowiedz"], errors="coerce").fillna(0).astype(int)

rows = []

for (sesja, warunek), g in sig.groupby(["id_sesji", "warunek"], dropna=False):
    x = g["poziom_bodzca"].values.astype(float)
    y = g["y"].values.astype(int)

    X = sm.add_constant(x)  # [1, x]
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit()

    b0, b1 = res.params  # logit(p) = b0 + b1*x

    # próg dla p=0.75: p = 1/(1+exp(-(b0+b1*x))) -> x = (logit(p)-b0)/b1
    logit_p = np.log(P_THRESHOLD / (1 - P_THRESHOLD))
    thr75 = (logit_p - b0) / b1 if b1 != 0 else np.nan

    rows.append({
        "id_sesji": sesja,
        "warunek": warunek,
        "b0": b0,
        "b1": b1,
        "threshold_75": thr75,
        "n_trials_signal": len(g),
        "n_levels": int(len(np.unique(x))),
        "deviance": float(res.deviance)
    })

out = pd.DataFrame(rows).sort_values(["id_sesji", "warunek"])
out.to_csv(OUT_PATH, index=False)
print(out.to_string(index=False))
print(f"\nZapisano: {OUT_PATH}")