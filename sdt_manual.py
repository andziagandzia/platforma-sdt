import math
import pandas as pd

def inverse_normal_cdf(p):
    """
    Aproksymacja odwrotnej dystrybuanty rozkładu normalnego (z-score), 
    często klasyfikowana jako aproksymacja Hastingsa (Abramowitz & Stegun).
    
    Wzór:
    Z(p) ≈ t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
    Gdzie:
    t = sqrt(-2 * ln(p)) dla p <= 0.5
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Prawdopodobieństwo musi być w przedziale (0, 1).")
        
    c0 = 2.515517; c1 = 0.802853; c2 = 0.010328
    d1 = 1.432788; d2 = 0.189269; d3 = 0.001308
    
    is_upper_half = p > 0.5
    if is_upper_half:
        p = 1.0 - p
        
    t = math.sqrt(-2.0 * math.log(p))
    
    numerator = c0 + c1 * t + c2 * (t**2)
    denominator = 1.0 + d1 * t + d2 * (t**2) + d3 * (t**3)
    
    z = t - (numerator / denominator)
    
    if is_upper_half:
        return z
    else:
        return -z

def calculate_rates(TP, FN, FP, TN):
    """
    Oblicza wartości odsetka trafień (HR) i fałszywych alarmów (FAR).
    Wzór (z zalecaną klasyczną korektą proporcjonalną na puste wartości):
    HR = (TP + 0.5) / (TP + FN + 1.0)
    FAR = (FP + 0.5) / (FP + TN + 1.0)
    """
    signal_trials = TP + FN
    noise_trials = FP + TN
    
    # Stosujemy korektę "log-linear" Macmillana i Creelmana (2005)
    # Zabezpiecza przed wynikiem dokładnie 1.0 (z-score dążyłby do nieskończoności)
    HR = (TP + 0.5) / (signal_trials + 1.0)
    FAR = (FP + 0.5) / (noise_trials + 1.0)
    
    return HR, FAR

def calculate_dprime_and_c(HR, FAR):
    """
    Oblicza czułość d' (d-prime) i kryterium (c) na podstawie funkcji z(p).
    
    Wzory analityczne SDT:
    d' = Z(HR) - Z(FAR)
    c  = -0.5 * (Z(HR) + Z(FAR))
    """
    z_HR = inverse_normal_cdf(HR)
    z_FAR = inverse_normal_cdf(FAR)
    
    d_prime = z_HR - z_FAR
    c = -0.5 * (z_HR + z_FAR)
    
    return d_prime, c

def compute_sdt_metrics(df):
    """
    Funkcja przyjmuje DataFrame z wynikami, liczy macierz pomyłek 
    oraz podstawowe metryki SDT wykorzystując manualne równania.
    Wymagane kolumny: 'klasa_wyniku' wprost lub 'bodziec_obecny' i 'odpowiedz'.
    """
    if 'klasa_wyniku' in df.columns:
        TP = (df['klasa_wyniku'] == 'TP').sum()
        FN = (df['klasa_wyniku'] == 'FN').sum()
        FP = (df['klasa_wyniku'] == 'FP').sum()
        TN = (df['klasa_wyniku'] == 'TN').sum()
    else:
        # manualna kategoryzacja
        TP = ((df['bodziec_obecny'] == 1) & (df['odpowiedz'] == 1)).sum()
        FN = ((df['bodziec_obecny'] == 1) & (df['odpowiedz'] == 0)).sum()
        FP = ((df['bodziec_obecny'] == 0) & (df['odpowiedz'] == 1)).sum()
        TN = ((df['bodziec_obecny'] == 0) & (df['odpowiedz'] == 0)).sum()
        
    HR, FAR = calculate_rates(TP, FN, FP, TN)
    d_prime, c = calculate_dprime_and_c(HR, FAR)
    
    return {
        'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
        'HR': HR, 'FAR': FAR,
        'd_prime': d_prime, 'criterion_c': c
    }

def fit_psychometric_curve_manual(levels, responses, epochs=1000, lr=0.1):
    """
    Dopasowanie krzywej psychometrycznej (funkcji logistycznej) z użyciem metody spadku gradientu (Gradient Descent).
    Tutaj funkcja logistyczna symuluje psychometryczną miarę p(x) = 1 / (1 + exp(-(beta_0 + beta_1 * x))).
    Jest to wzór analityczny rozwiązany w pełni matematycznie.
    Zwraca parametry b0, b1 oraz obliczony próg progu detekcji na 75%.
    """
    b0 = 0.0
    b1 = 0.0
    n = len(levels)
    
    if n == 0:
        return None, None, None
        
    for _ in range(epochs):
        grad_b0 = 0.0
        grad_b1 = 0.0
        
        for x_i, y_i in zip(levels, responses):
            # p_i = 1 / (1 + e^{-(b0 + b1*x_i)})
            p_i = 1.0 / (1.0 + math.exp(-(b0 + b1 * x_i)))
            
            # gradient Log-Likelihood względem parametrów (podstawowy model binominalny)
            error = y_i - p_i
            grad_b0 += error
            grad_b1 += error * x_i
            
        # Aktualizacja
        b0 += lr * (grad_b0 / n)
        b1 += lr * (grad_b1 / n)
        
    # Próg dla 75% detekcji: p = 0.75 => x = (ln(0.75/0.25) - b0) / b1
    logit_75 = math.log(0.75 / 0.25)
    threshold_75 = (logit_75 - b0) / b1 if b1 != 0 else float('nan')
    
    return b0, b1, threshold_75

import numpy as np
from scipy.stats import norm

def calculate_empirical_roc(df):
    """
    Oblicza punkty (FAR, HR) dla empirycznej krzywej ROC
    na podstawie ocen pewności (1-5) i odpowiedzi T/N (1/0).
    """
    if 'pewnosc' not in df.columns or df['pewnosc'].isnull().all():
        return [], []
        
    df = df.dropna(subset=['pewnosc', 'bodziec_obecny', 'odpowiedz']).copy()
    if len(df) == 0:
        return [], []
        
    def map_rating(row):
        r = float(row['pewnosc'])
        o = int(row['odpowiedz'])
        if o == 0:
            return 6 - r
        else:
            return r + 5
            
    df['roc_rating'] = df.apply(map_rating, axis=1)
    
    far_points = [0.0]
    hr_points = [0.0]
    
    for c in sorted(df['roc_rating'].unique(), reverse=True):
        TP = ((df['roc_rating'] >= c) & (df['bodziec_obecny'] == 1)).sum()
        FN = ((df['roc_rating'] < c) & (df['bodziec_obecny'] == 1)).sum()
        FP = ((df['roc_rating'] >= c) & (df['bodziec_obecny'] == 0)).sum()
        TN = ((df['roc_rating'] < c) & (df['bodziec_obecny'] == 0)).sum()
        
        sig_trials = TP + FN
        noise_trials = FP + TN
        
        hr = TP / sig_trials if sig_trials > 0 else 0
        far = FP / noise_trials if noise_trials > 0 else 0
        
        far_points.append(far)
        hr_points.append(hr)
        
    far_points.append(1.0)
    hr_points.append(1.0)
    return far_points, hr_points

def calculate_theoretical_roc(d_prime):
    """
    Generuje punkty teoretycznej krzywej ROC dla danego d'.
    """
    c_vals = np.linspace(3, -3, 100)
    far_points = norm.cdf(-c_vals)
    hr_points = norm.cdf(d_prime - c_vals)
    return far_points.tolist(), hr_points.tolist()

def calculate_auc(far_points, hr_points):
    """
    Oblicza pole pod krzywą ROC metodą trapezów.
    """
    if len(far_points) < 2:
        return 0.0
    return np.trapz(hr_points, far_points)

def calculate_z_roc(far_points, hr_points):
    """
    Przekształca wskaźniki (FAR, HR) z przestrzeni krzywej prawdopodobieństw ROC na przestrzeń wektorów Z-Score
    Z-ROC model Unequal-Variance SDT
    """
    z_hr = []
    z_far = []

    for f, h in zip(far_points, hr_points):
        # Trzeba pominąć wierzchołkowe 0.0 i 1.0 (które by dawały -inf i inf)
        if f <= 0 or h <= 0 or f >= 1 or h >= 1:
            continue
            
        z_f = norm.ppf(f)
        z_h = norm.ppf(h)
        
        z_far.append(z_f)
        z_hr.append(z_h)
        
    return z_far, z_hr

def fit_z_roc(z_far, z_hr):
    """
    Wykonuje estymację Liniową Najmniejszych Kwadratów, aby dopasować prostą do punktów z-ROC.
    Wylicza i zwraca:
    nachylenie `s` -> wskazuje wariancję gęstości sygnału (np. s<1 to poszerzenie wariancji względem szumu).
    przecięcie prostej `d_e`
    wrażliwość nierówno-wariancyjną `d_a`.
    """
    if len(z_far) < 2:
        return 1.0, 0.0, 0.0
        
    # Dopasowanie matematyczne z(hr) = s*z(far) + d_e
    slope, intercept = np.polyfit(z_far, z_hr, 1)
    
    s = float(slope)
    d_e = float(intercept)
    
    # Przeliczona miara nierówno-wariancyjna progu d_a
    d_a = float(d_e * np.sqrt(2 / (1 + s**2)))
    
    return s, d_e, d_a
