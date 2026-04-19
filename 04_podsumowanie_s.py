import pandas as pd
from scipy.stats import norm

def rate_correction(k, n):
    # korekta na skrajne wartości 0 i 1 (żeby nie było inf)
    return (k + 0.5) / (n + 1.0)

def sdt_from_counts(tp, fn, fp, tn):
    n_signal = tp + fn
    n_noise = fp + tn
    H = tp / n_signal if n_signal else 0.0
    F = fp / n_noise if n_noise else 0.0

    Hc = rate_correction(tp, n_signal) if n_signal else 0.5
    Fc = rate_correction(fp, n_noise) if n_noise else 0.5

    d_prime = norm.ppf(Hc) - norm.ppf(Fc)
    C = -0.5 * (norm.ppf(Hc) + norm.ppf(Fc))
    return H, F, d_prime, C

df = pd.read_csv("data_wyniki.csv")


gr = df.groupby(["id_sesji", "warunek"], dropna=False)

rows = []
for (sesja, warunek), g in gr:
    tp = (g["klasa_wyniku"] == "TP").sum()
    fn = (g["klasa_wyniku"] == "FN").sum()
    fp = (g["klasa_wyniku"] == "FP").sum()
    tn = (g["klasa_wyniku"] == "TN").sum()

    H, F, dprime, C = sdt_from_counts(tp, fn, fp, tn)

    rt = pd.to_numeric(g["czas_reakcji_ms"], errors="coerce")
    rt_mean = rt.mean()

    rows.append({
        "id_sesji": sesja,
        "warunek": warunek,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "hit_rate": round(H, 3),
        "false_alarm_rate": round(F, 3),
        "d_prime": round(dprime, 3),
        "criterion_C": round(C, 3),
        "rt_mean_ms": round(rt_mean, 1) if pd.notna(rt_mean) else None,
        "n_trials": len(g)
    })

out = pd.DataFrame(rows).sort_values(["id_sesji", "warunek"])
print(out.to_string(index=False))
out.to_csv("podsumowanie_sesji.csv", index=False)
print("\nZapisano: podsumowanie_sesji.csv")