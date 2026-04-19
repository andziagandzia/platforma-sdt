import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# USTAWIENIA
# =========================
DATA_PATH = "data_wyniki.csv"  # jeśli trzymasz w data/: "data/data_wyniki.csv"
OUT_DIR = "psychometria_sdt_wykresy"
OUT_TABLE = "psychometria_sdt_dprime.csv"

DPRIME_THRESHOLD = 1.0         # próg SDT: d′ = 1
SAVE_PLOTS = True

# =========================
# Funkcje pomocnicze
# =========================
def rate_correction(k, n):
    """
    Korekta na skrajne wartości (żeby nie było z(0) i z(1)).
    Zamiast H=0 lub H=1 robimy: (k+0.5)/(n+1)
    """
    return (k + 0.5) / (n + 1.0)

def dprime_from_counts(tp, fn, fp, tn):
    """
    Liczy d′ i C z macierzy pomyłek (TP,FN,FP,TN) z korektą.
    """
    n_signal = tp + fn
    n_noise = fp + tn

    # H = TP/(TP+FN), F = FP/(FP+TN)
    H = tp / n_signal if n_signal else 0.0
    F = fp / n_noise if n_noise else 0.0

    # korekty (żeby norm.ppf nie zwrócił inf)
    Hc = rate_correction(tp, n_signal) if n_signal else 0.5
    Fc = rate_correction(fp, n_noise) if n_noise else 0.5

    zH = norm.ppf(Hc)
    zF = norm.ppf(Fc)

    d_prime = zH - zF
    C = -0.5 * (zH + zF)

    return H, F, d_prime, C

def interpolate_threshold(x, y, target=1.0):
    """
    Prosta interpolacja liniowa: szukamy, gdzie y przecina target.
    x: rosnące poziomy bodźca
    y: d′(x)
    Zwraca x*, jeśli przecięcie istnieje, inaczej None.
    """
    if len(x) < 2:
        return None

    # szukamy odcinka, na którym target leży pomiędzy y[i] a y[i+1]
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 <= target <= y2) or (y2 <= target <= y1):
            x1, x2 = x[i], x[i + 1]
            # jeśli y1==y2, nie da się policzyć nachylenia
            if y2 == y1:
                return x1
            # interpolacja liniowa
            return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    return None

# =========================
# Wczytaj dane
# =========================
df = pd.read_csv(DATA_PATH)

# upewnij się, że poziom bodźca jest liczbą
df["poziom_bodzca"] = pd.to_numeric(df["poziom_bodzca"], errors="coerce")

# =========================
# Główne liczenie: d′(x) w każdej sesji
# =========================
if SAVE_PLOTS:
    os.makedirs(OUT_DIR, exist_ok=True)

rows = []

# grupujemy po sesji i warunku
for (sesja, warunek), gses in df.groupby(["id_sesji", "warunek"], dropna=False):

    # 1) Liczymy FP i TN z prób bez bodźca w tej sesji (noise trials)
    g_noise = gses[gses["bodziec_obecny"] == 0]
    fp = (g_noise["klasa_wyniku"] == "FP").sum()
    tn = (g_noise["klasa_wyniku"] == "TN").sum()

    # jeżeli w sesji nie ma prób bez bodźca, to nie policzysz F/d′ poprawnie
    if (fp + tn) == 0:
        print(f"[UWAGA] {sesja}|{warunek}: brak prób bez bodźca -> pomijam.")
        continue

    # policz F (jeden dla sesji)
    F_raw = fp / (fp + tn)
    Fc = rate_correction(fp, fp + tn)  # skorygowane F do z-score
    zF = norm.ppf(Fc)

    # 2) Dla każdego poziomu bodźca liczymy TP i FN (signal trials na danym poziomie)
    g_sig = gses[gses["bodziec_obecny"] == 1].dropna(subset=["poziom_bodzca"])

    # jeśli nie ma prób z bodźcem, nie ma psychometrii
    if len(g_sig) == 0:
        print(f"[UWAGA] {sesja}|{warunek}: brak prób z bodźcem -> pomijam.")
        continue

    level_groups = g_sig.groupby("poziom_bodzca")

    # zbieramy punkty d′(x) do wykresu
    xs = []
    ys = []

    for level, gl in level_groups:
        tp_level = (gl["klasa_wyniku"] == "TP").sum()
        fn_level = (gl["klasa_wyniku"] == "FN").sum()

        n_signal = tp_level + fn_level
        if n_signal == 0:
            continue

        # H dla tego poziomu
        H_raw = tp_level / n_signal
        Hc = rate_correction(tp_level, n_signal)
        zH = norm.ppf(Hc)

        # d′(x) = z(Hc(x)) - z(Fc)
        dprime_x = zH - zF

        # kryterium też można policzyć per poziom, ale tu głównie d′(x)
        C_x = -0.5 * (zH + zF)

        rows.append({
            "id_sesji": sesja,
            "warunek": warunek,
            "poziom_bodzca": float(level),

            "TP_level": int(tp_level),
            "FN_level": int(fn_level),
            "FP_session": int(fp),
            "TN_session": int(tn),

            "H_level": float(H_raw),
            "F_session": float(F_raw),
            "d_prime_level": float(dprime_x),
            "criterion_C_level": float(C_x),

            "n_signal_level": int(n_signal),
            "n_noise_session": int(fp + tn)
        })

        xs.append(float(level))
        ys.append(float(dprime_x))

    # 3) Wykres d′(x) i próg d′=1 (interpolacja)
    if len(xs) >= 2:
        order = np.argsort(xs)
        xs_sorted = np.array(xs)[order]
        ys_sorted = np.array(ys)[order]

        thr_x = interpolate_threshold(xs_sorted, ys_sorted, target=DPRIME_THRESHOLD)

        plt.figure()
        plt.plot(xs_sorted, ys_sorted, marker="o", linestyle="-")
        plt.axhline(DPRIME_THRESHOLD)

        if thr_x is not None:
            plt.axvline(thr_x)
            plt.title(f"d′(x) | {sesja} | {warunek} | próg d′={DPRIME_THRESHOLD}: {thr_x:.4f}")
        else:
            plt.title(f"d′(x) | {sesja} | {warunek} | próg d′={DPRIME_THRESHOLD}: brak przecięcia")

        plt.xlabel("Poziom bodźca")
        plt.ylabel("d′ (czułość SDT)")
        plt.ylim(bottom=min(-0.2, ys_sorted.min() - 0.2))

        if SAVE_PLOTS:
            safe = f"{sesja}_{warunek}".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(OUT_DIR, f"dprime_{safe}.png"), dpi=200)

        plt.show()

        # zapis progu do osobnej tabeli wierszowej (żeby potem łatwo scalać)
        rows.append({
            "id_sesji": sesja,
            "warunek": warunek,
            "poziom_bodzca": np.nan,

            "TP_level": np.nan,
            "FN_level": np.nan,
            "FP_session": int(fp),
            "TN_session": int(tn),

            "H_level": np.nan,
            "F_session": float(F_raw),
            "d_prime_level": np.nan,
            "criterion_C_level": np.nan,

            "n_signal_level": np.nan,
            "n_noise_session": int(fp + tn),

            "threshold_dprime1": float(thr_x) if thr_x is not None else np.nan
        })

    else:
        print(f"[UWAGA] {sesja}|{warunek}: za mało poziomów bodźca do wyznaczenia progu d′=1.")

# =========================
# Zapis wyników
# =========================
out = pd.DataFrame(rows)

# jeśli nie ma kolumny threshold_dprime1 (bo nie dodano), dodaj ją
if "threshold_dprime1" not in out.columns:
    out["threshold_dprime1"] = np.nan

out.to_csv(OUT_TABLE, index=False)
print(f"\nZapisano: {OUT_TABLE}")
print(f"Wykresy (jeśli SAVE_PLOTS=True): {OUT_DIR}/")