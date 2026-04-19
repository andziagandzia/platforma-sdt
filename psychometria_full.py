import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_PATH = "data_wyniki.csv"
OUT_PATH = "psychometria_full_wzory.csv"
P_THRESHOLD = 0.75                    # próg 75%

# --- model: pełna psychometryczna logistyka z gamma i lapse ---
def psychometric_logistic(x, alpha, beta, gamma, lapse):
    # zabezpieczenie beta>0
    beta = max(beta, 1e-6)
    s = 1.0 / (1.0 + np.exp(-(x - alpha) / beta))
    return gamma + (1.0 - gamma - lapse) * s

def neg_log_likelihood(params, x, y):
    alpha, log_beta, gamma, lapse = params
    beta = np.exp(log_beta)  # beta>0
    p = psychometric_logistic(x, alpha, beta, gamma, lapse)

    # obetnij p, żeby nie było log(0)
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)

    # Bernoulli NLL
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def threshold_from_params(alpha, beta, gamma, lapse, p=0.75):
    """
    Szukamy x takiego, że p(x)=p.
    Dla pełnego modelu:
    p = gamma + (1-gamma-lapse) * sigmoid((x-alpha)/beta)
    -> sigmoid = (p-gamma)/(1-gamma-lapse)
    -> x = alpha + beta * ln(sigmoid/(1-sigmoid))
    """
    denom = (1.0 - gamma - lapse)
    if denom <= 0:
        return np.nan

    s = (p - gamma) / denom
    if s <= 0 or s >= 1:
        return np.nan

    return alpha + beta * np.log(s / (1.0 - s))

df = pd.read_csv(DATA_PATH)
df["poziom_bodzca"] = pd.to_numeric(df["poziom_bodzca"], errors="coerce")

# bierzemy tylko próby z bodźcem (signal-present) i z poziomem
sig = df[(df["bodziec_obecny"] == 1) & df["poziom_bodzca"].notna()].copy()

# tu modelujemy P(TAK | bodziec, x)
# odpowiedz: 1 = TAK, 0 = NIE
sig["y"] = pd.to_numeric(sig["odpowiedz"], errors="coerce").fillna(0).astype(int)

rows = []

for (sesja, warunek), g in sig.groupby(["id_sesji", "warunek"], dropna=False):
    x = g["poziom_bodzca"].values.astype(float)
    y = g["y"].values.astype(int)

    # jeśli masz mało punktów/poziomów, fit może być niestabilny
    if len(np.unique(x)) < 3:
        print(f"[UWAGA] {sesja}|{warunek}: <3 poziomy bodźca -> fit może być słaby.")

    # --- startowe wartości ---
    alpha0 = np.median(x)
    beta0 = max(np.std(x), 1e-2)
    gamma0 = 0.0        # start: 0
    lapse0 = 0.02       # start: mały lapse
    x0 = np.array([alpha0, np.log(beta0), gamma0, lapse0], dtype=float)

    # --- ograniczenia ---
    # gamma w [0, 0.4], lapse w [0, 0.2] (bezpieczne)
    # (w praktyce lapse zwykle małe, ale dajemy zakres)
    bounds = [
        (min(x) - 1.0, max(x) + 1.0),     # alpha
        (np.log(1e-4), np.log(10.0)),     # log_beta
        (0.0, 0.4),                       # gamma
        (0.0, 0.2)                        # lapse
    ]

    res = minimize(
        neg_log_likelihood,
        x0=x0,
        args=(x, y),
        method="L-BFGS-B",
        bounds=bounds
    )

    if not res.success:
        print(f"[UWAGA] Fit nieudany {sesja}|{warunek}: {res.message}")

    alpha, log_beta, gamma, lapse = res.x
    beta = np.exp(log_beta)

    thr75 = threshold_from_params(alpha, beta, gamma, lapse, p=P_THRESHOLD)

    rows.append({
        "id_sesji": sesja,
        "warunek": warunek,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lapse": lapse,
        "threshold_75": thr75,
        "n_trials_signal": len(g),
        "n_levels": int(len(np.unique(x))),
        "fit_success": bool(res.success),
        "nll": float(res.fun)
    })

out = pd.DataFrame(rows).sort_values(["id_sesji", "warunek"])
out.to_csv(OUT_PATH, index=False)
print(out.to_string(index=False))
print(f"\nZapisano: {OUT_PATH}")