import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

def calculate_sdt(df):
    """
    Oblicza podstawowe parametry Teorii Detekcji Sygnału (SDT):
    - HR (Hit Rate) - odsetek poprawnych detekcji
    - FAR (False Alarm Rate) - odsetek fałszywych alarmów
    - d' (d-prime) - wrażliwość percepcyjna
    - c (criterion) - kryterium decyzyjne
    """
    TP = (df['klasa_wyniku'] == 'TP').sum()
    FN = (df['klasa_wyniku'] == 'FN').sum()
    FP = (df['klasa_wyniku'] == 'FP').sum()
    TN = (df['klasa_wyniku'] == 'TN').sum()
    
    # Korekta log-linear aby uniknąć wartości 0 i 1 dla prawdopodobieństw (które uniemożliwiają policzenie z-score)
    HR = (TP + 0.5) / (TP + FN + 1)
    FAR = (FP + 0.5) / (FP + TN + 1)
    
    # Obliczanie d' i kryterium decyzyjnego
    d_prime = norm.ppf(HR) - norm.ppf(FAR)
    c = -(norm.ppf(HR) + norm.ppf(FAR)) / 2.0
    
    return pd.Series({
        'HR': HR,
        'FAR': FAR,
        'd_prime': d_prime,
        'criterion_c': c,
        'n_TP': TP, 'n_FN': FN, 'n_FP': FP, 'n_TN': TN
    })

def main():
    plik_danych = 'data_wyniki.csv'
    
    if not os.path.exists(plik_danych):
        print(f"Błąd: Plik {plik_danych} nie istnieje w bieżącym katalogu.")
        return
        
    print(f"Wczytywanie danych z {plik_danych}...")
    df = pd.read_csv(plik_danych)
    
    # Konwersja czasu reakcji na wartości numeryczne, puste zamienią się na NaN
    df['czas_reakcji_ms'] = pd.to_numeric(df['czas_reakcji_ms'], errors='coerce')
    
    print(f"Liczba załadowanych prób: {len(df)}")
    
    # -------------------------------------------------------------
    # 1. OGÓLNA ANALIZA PODSTAWOWA
    # -------------------------------------------------------------
    acc = df['czy_poprawna'].mean() * 100
    print(f"\n--- PODSTAWOWE WYNIKI ---")
    print(f"Ogólna sprawdzalność (Accuracy): {acc:.2f}%")
    
    sdt_overall = calculate_sdt(df)
    print("\n--- OGÓLNE WSKAŹNIKI SDT (Signal Detection Theory) ---")
    print(f"Hit Rate (HR):   {sdt_overall['HR']:.3f} (Trafienia: {int(sdt_overall['n_TP'])}/{int(sdt_overall['n_TP']+sdt_overall['n_FN'])})")
    print(f"False Alarms Rate (FAR): {sdt_overall['FAR']:.3f} (Fałszywe Alarmy: {int(sdt_overall['n_FP'])}/{int(sdt_overall['n_FP']+sdt_overall['n_TN'])})")
    print(f"d' (Sensitivity):  {sdt_overall['d_prime']:.3f}")
    print(f"c (Criterion):     {sdt_overall['criterion_c']:.3f}")

    # -------------------------------------------------------------
    # 2. ANALIZA SDT WG WARUNKÓW EKSPERYMENTALNYCH
    # -------------------------------------------------------------
    if 'warunek' in df.columns:
        print("\n--- SDT WG WARUNKÓW (np. rano, wieczorem, po pracy) ---")
        sdt_cond = df.groupby('warunek').apply(calculate_sdt)
        print(sdt_cond[['HR', 'FAR', 'd_prime', 'criterion_c']].round(3))

    # -------------------------------------------------------------
    # 3. KRZYWA PSYCHOMETRYCZNA (Hit Rate vs Poziom Bodźca)
    # -------------------------------------------------------------
    print("\n--- SKUTECZNOŚĆ DETEKCJI WG POZIOMU BODŹCA ---")
    # interesują nas tylko próby, w których bodziec rzeczywiście był (bodziec_obecny == 1)
    df_signal = df[df['bodziec_obecny'] == 1]
    hit_rates = df_signal.groupby('poziom_bodzca').apply(
        lambda x: (x['klasa_wyniku'] == 'TP').sum() / len(x) if len(x) > 0 else np.nan
    ).reset_index(name='HR')
    
    print(hit_rates.round(3))

    # -------------------------------------------------------------
    # 4. STATYSTYKI CZASÓW REAKCJI
    # -------------------------------------------------------------
    print("\n--- ŚREDNI CZAS REAKCJI (RT) DLA KLAS ODPOWIEDZI ---")
    rt_stats = df.dropna(subset=['czas_reakcji_ms']).groupby('klasa_wyniku')['czas_reakcji_ms'].agg(['mean', 'median', 'std', 'count'])
    print(rt_stats.round(1))

    # =============================================================
    # GENEROWANIE WYKRESÓW POGLĄDOWYCH
    # =============================================================
    sns.set_theme(style="whitegrid")
    
    # Wykres 1: Wrażliwość (d') wg warunku (jeśli istnieje)
    if 'warunek' in df.columns:
        plt.figure(figsize=(8, 5))
        sdt_cond_plot = sdt_cond.reset_index()
        sns.barplot(data=sdt_cond_plot, x='warunek', y='d_prime', palette='viridis', hue='warunek', legend=False)
        plt.title("Wrażliwość percepcyjna (d') w zależności od warunku")
        plt.ylabel("d' (większe = lepsza detekcja)")
        plt.xlabel("Warunek testu")
        plt.tight_layout()
        plt.savefig('analiza_dprime_warunek.png', dpi=150)
        plt.close()
        
    # Wykres 2: Krzywa psychometryczna
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=hit_rates, x='poziom_bodzca', y='HR', marker='o', markersize=8, color='crimson')
    plt.title("Krzywa psychometryczna - detekcja zależna od sygnału")
    plt.xlabel("Poziom bodźca (intensywność)")
    plt.ylabel("Prawdopodobieństwo Hit (HR)")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('analiza_krzywa_psychometryczna.png', dpi=150)
    plt.close()
    
    # Wykres 3: Czasy reakcji wg klasy SDT
    plt.figure(figsize=(8, 5))
    # Wykluczamy NaN dla RT
    df_rt = df.dropna(subset=['czas_reakcji_ms']).copy()
    sns.boxplot(data=df_rt, x='klasa_wyniku', y='czas_reakcji_ms', order=['TP', 'TN', 'FP', 'FN'], palette='Set2')
    plt.title("Czas reakcji wg rodzaju odpowiedzi testowej")
    plt.xlabel("Klasa SDT (TP=Hit, TN=CR, FP=FA, FN=Miss)")
    plt.ylabel("Czas reakcji (ms)")
    plt.tight_layout()
    plt.savefig('analiza_czas_reakcji.png', dpi=150)
    plt.close()
    
    # Wykres 4: Krzywa ROC (SDT)
    plt.figure(figsize=(6, 6))
    far_range = np.linspace(0.001, 0.999, 100)
    d_prime = sdt_overall['d_prime']
    HR_emp = sdt_overall['HR']
    FAR_emp = sdt_overall['FAR']
    
    # Przekształcenie z-score modelu ROC: HR = Φ(Φ⁻¹(FAR) + d')
    hr_model = norm.cdf(norm.ppf(far_range) + d_prime)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Linia szansy (d'=0)")
    plt.plot(far_range, hr_model, color='purple', linewidth=2, label=f"Teoretyczna ROC (d'={d_prime:.2f})")
    plt.plot(FAR_emp, HR_emp, 'o', color='darkorange', markersize=10, markeredgecolor='black', label="Punkt empiryczny", zorder=5)
    
    plt.title(f"Krzywa ROC (Całkowita wrażliwość d' = {d_prime:.2f})")
    plt.xlabel("False Alarm Rate (FAR)")
    plt.ylabel("Hit Rate (HR)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('analiza_krzywa_roc.png', dpi=150)
    plt.close()
    
    print("\n[ZAKOŃCZONO] Analiza przebiegła pomyślnie. Wygenerowano wykresy i zapisano w bieżącym folderze jako:")
    print(" - analiza_krzywa_psychometryczna.png")
    print(" - analiza_czas_reakcji.png")
    print(" - analiza_krzywa_roc.png")
    if 'warunek' in df.columns:
        print(" - analiza_dprime_warunek.png")

if __name__ == "__main__":
    main()
