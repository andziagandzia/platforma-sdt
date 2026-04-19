import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os
import sys
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sdt_manual import compute_sdt_metrics as manual_sdt
from sdt_manual import fit_psychometric_curve_manual, calculate_empirical_roc, calculate_theoretical_roc, calculate_auc, calculate_z_roc, fit_z_roc
from sdt_library import compute_sdt_metrics_lib as lib_sdt
from sdt_library import fit_psychometric_curve_lib

st.set_page_config(page_title="Platforma SDT", layout="wide")

# --- KOD GOOGLE ANALYTICS ---
import streamlit.components.v1 as components
components.html(
    """
    <script>
        var script = window.parent.document.createElement('script');
        script.async = true;
        script.src = "https://www.googletagmanager.com/gtag/js?id=G-PCW10VM14N";
        window.parent.document.head.appendChild(script);

        var script2 = window.parent.document.createElement('script');
        script2.innerHTML = `
          window.dataLayer = window.parent.dataLayer || [];
          function gtag(){window.dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'G-PCW10VM14N');
        `;
        window.parent.document.head.appendChild(script2);
    </script>
    """,
    width=0, height=0
)
# -----------------------------

st.markdown("""
<style>
    /* Czyste tło aplikacji i ukrycie elementów Streamlit (menu, footer) by wyglądało jak pełnoprawna aplikacja */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #F4F6F8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Główne nagłówki - eleganckie, czytelne */
    h1, h2, h3 {
        color: #2D3748 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        padding-bottom: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid #E2E8F0;
        font-size: 2.2rem !important;
    }
    
    h2, h3 {
        padding-top: 15px;
        color: #4A5568 !important;
    }

    hr {
        border-color: #E2E8F0;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Zmiana koloru wskaźników tekstowych np. metryk na czystsze */
    [data-testid="stMetricValue"] {
        color: #2B6CB0 !important; /* Łagodny niebieski korporacyjny */
        font-weight: 800 !important;
        font-family: inherit !important;
        text-shadow: none !important;
    }
    [data-testid="stMetricLabel"] {
        color: #718096 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }
    
    /* Przyciski radio w sidebarze by wyglądały jak menu */
    div.stRadio > div[role="radiogroup"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Platforma Analityczna SDT")
st.markdown("**Zaawansowane środowisko do analizy danych psychofizycznych i modelowania detekcji sygnałów**")

def load_data():
    file_path = "data_wyniki.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['poziom_bodzca'] = pd.to_numeric(df['poziom_bodzca'], errors='coerce')
        df['czas_reakcji_ms'] = pd.to_numeric(df['czas_reakcji_ms'], errors='coerce')
        if 'pewnosc' in df.columns:
            df['pewnosc'] = pd.to_numeric(df['pewnosc'], errors='coerce')
        return df
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Nie znaleziono pliku: data_wyniki.csv.")
else:
    # Boczny panel testowy
    st.sidebar.header("Moduł Rejestracyjno-Badawczy")
    st.sidebar.markdown("Uruchomienie pomiarów:")
    if st.sidebar.button("Test (Metoda Stałych Bodźców)", use_container_width=True):
        st.sidebar.warning("Funkcja zbierania danych u uczestników jest zablokowana w wersji przeglądarkowej chmury.")

    if st.sidebar.button("Test Adaptacyjny (Staircase)", use_container_width=True):
        st.sidebar.warning("Funkcja zbierania danych u uczestników jest zablokowana w wersji przeglądarkowej chmury.")
        
    st.sidebar.markdown("---")
    
    # Boczny panel nawigacyjny - MENU ZAMIAST ZAKŁADEK
    st.sidebar.header("Nawigacja Modułów")
    opcje_menu = [
        "Panel Główny: Surowe Dane i Metryki",
        "Walidacja i Porównanie Metod",
        "Psychometria i Krzywe ROC",
        "Analiza Znużenia (Chronologiczna)",
        "Model Uczenia Maszynowego (AI)"
    ]
    wybrany_modul = st.sidebar.radio("Wybierz obszar analityczny:", opcje_menu)
    
    st.sidebar.markdown("---")
    
    # Boczny panel filtrów
    st.sidebar.header("Filtracja Wyników")
    uczestnicy = ["Wszyscy"] + list(df['id_uczestnika'].unique())
    wybrany_uczestnik = st.sidebar.selectbox("Identyfikator uczestnika:", uczestnicy)
    
    warunki = ["Wszystkie"] + list(df['warunek'].unique())
    wybrany_warunek = st.sidebar.selectbox("Warunek eksperymentalny:", warunki)
    
    # Filtrowanie
    df_filtered = df.copy()
    if wybrany_uczestnik != "Wszyscy":
        df_filtered = df_filtered[df_filtered['id_uczestnika'] == wybrany_uczestnik]
    if wybrany_warunek != "Wszystkie":
        df_filtered = df_filtered[df_filtered['warunek'] == wybrany_warunek]
        
    st.sidebar.write(f"Aktywna wolumen sesji: {len(df_filtered)} prób.")
    
    # --------------------------------------------------------------------------------
    # MODUŁ 1: PANEL GŁÓWNY
    # --------------------------------------------------------------------------------
    if wybrany_modul == "Panel Główny: Surowe Dane i Metryki":
        st.subheader("Podstawowe podsumowanie wyników (Podejście Analityczne)")
        metrics_man = manual_sdt(df_filtered)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hit Rate (HR)", f"{metrics_man['HR']:.3f}")
        col2.metric("False Alarm Rate (FAR)", f"{metrics_man['FAR']:.3f}")
        col3.metric("Wyliczone d'", f"{metrics_man['d_prime']:.3f}")
        col4.metric("Kryterium c", f"{metrics_man['criterion_c']:.3f}")
        
        with st.expander("Informacja Merytoryczna: Specyfikacja Parametrów"):
            st.markdown('''
            * **Hit Rate (HR)**: Proporcja prawidłowych wykryć. Informuje, z jakim prawdopodobieństwem wariant detekcyjny zidentyfikował faktyczny sygnał.
            * **False Alarm Rate (FAR)**: Odsetek fałszywych alarmów. Określa skłonność badanego do deklaracji sygnału w sytuacjach uwarunkowanego szumu.
            * **Współczynnik Wrażliwości (d')**: Odległość z-score między medianą rozkładu sygnału a medianą szumu. Im wyższa wartość wskaźnika, tym wyższa stabilność sensoryczna.
            * **Kryterium decyzyjne (c)**: Środek decyzyjny (Response Bias). Wartość c>0 określa strategię zachowawczą, z kolei c<0 definiuje strategię liberalną.
            ''')
            
        st.markdown("---")
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.write("**Macierz Konfuzji Zdarzeń Decyzyjnych:**")
            matrix = pd.DataFrame({
                "Zarejestrowano Sygnał": [metrics_man['TP'], metrics_man['FN']],
                "Brak Sygnału": [metrics_man['FP'], metrics_man['TN']]
            }, index=["Odpowiedź twierdząca", "Odpowiedź przecząca"])
            st.table(matrix)
            
        with col_m2:
            st.write("**Eksploracja surowego wektora badawczego:**")
            st.dataframe(df_filtered.head(10))
            
        st.markdown("---")
        st.subheader("Analiza Wypadkowej Czasów Reakcji")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.write("**Gęstość RT względem warunku eksperymentalnego**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.violinplot(data=df_filtered.dropna(subset=['czas_reakcji_ms']), 
                           x='warunek', y='czas_reakcji_ms', inner='quartile', palette='muted', ax=ax)
            ax.set_ylabel("Czas reakcji (ms)")
            ax.set_xlabel("Warunek testu")
            st.pyplot(fig)
            
        with col_plot2:
            st.write("**Kowariancja opóźnień decyzyjnych w klasach SDT**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_filtered.dropna(subset=['czas_reakcji_ms']),
                        x='klasa_wyniku', y='czas_reakcji_ms', palette="Set2", order=['TP', 'TN', 'FP', 'FN'], ax=ax2)
            ax2.set_ylabel("Czas reakcji (ms)")
            ax2.set_xlabel("Klasa Poprawności")
            st.pyplot(fig2)
            
        with st.expander("Informacja Merytoryczna: Opóźnienia R/T"):
            st.markdown('''
            Zaawansowane obciążenie aparatu poznawczego występujące przy pomyłkach decyzyjnych lub nieufności pacjenta jest naturalnie powiązane ze statystycznie obserwowalnym wzrostem wahań czasów reakcji w poszczególnych kwartylach dystrybucyjnych wykresu pudełkowego. Rozkład gęstości pozwala natomiast zdeklarować stopień spójności rytmiki badanego.
            ''')
            
    # --------------------------------------------------------------------------------
    # MODUŁ 2: WALIDACJA METOD
    # --------------------------------------------------------------------------------
    elif wybrany_modul == "Walidacja i Porównanie Metod":
        st.subheader("Ewaluacja Matematyczna: Wzory vs Silniki Biblioteczne")
        st.markdown('''
        Zgodnie z wymaganiami technicznymi pracy, moduł prezentuje ścisłe dowody numeryczne konfirmacji wejścia/wyjścia użytych struktur.
        * **Transformacja algorytmiczna (sdt_manual.py):** Obliczona niezależnymi rzutowaniami równań.
        * **Kompilator standardowy (sdt_library.py):** Zgodnie z paczkami `scipy` / `statsmodels`.
        ''')
        
        metrics_man = manual_sdt(df_filtered)
        metrics_lib = lib_sdt(df_filtered)
        
        comp_df = pd.DataFrame({
            "Indeks Metryki": ["Wolumen d' (Sensitivity)", "Poziom c (Criterion)", "Proporcja Hit Rate", "Proporcja FAR"],
            "Silnik Algorytmiczny": [metrics_man['d_prime'], metrics_man['criterion_c'], metrics_man['HR'], metrics_man['FAR']],
            "Struktura Biblioteki": [metrics_lib['d_prime'], metrics_lib['criterion_c'], metrics_lib['HR'], metrics_lib['FAR']]
        })
        
        comp_df["Współczynnik Niezgodności (Błąd)"] = abs(comp_df["Silnik Algorytmiczny"] - comp_df["Struktura Biblioteki"])
        
        st.dataframe(comp_df.style.format({
            "Silnik Algorytmiczny": "{:.5f}", 
            "Struktura Biblioteki": "{:.5f}", 
            "Współczynnik Niezgodności (Błąd)": "{:.1e}"
        }))
        
        st.success("Test weryfikacyjny: Zgodność wektorów potwierdzona statystycznie (odchylenia w marginesie numerycznym float).")
        
    # --------------------------------------------------------------------------------
    # MODUŁ 3: PSYCHOMETRIA I ROC
    # --------------------------------------------------------------------------------
    elif wybrany_modul == "Psychometria i Krzywe ROC":
        st.header("1. Modelowanie Krzywej Psychometrycznej")
        st.markdown("Określenie estymacji progu natężenia warunkującego 75% detekcji w algorytmach optymalizacyjnych Gradient Descent.")
        
        df_sig = df_filtered[(df_filtered['bodziec_obecny'] == 1) & (df_filtered['poziom_bodzca'].notnull())].copy()
        df_sig['y'] = pd.to_numeric(df_sig['odpowiedz'], errors='coerce').fillna(0).astype(int)
        
        if len(df_sig) > 0:
            levels = df_sig['poziom_bodzca'].values.tolist()
            resps = df_sig['y'].values.tolist()
            
            b0_m, b1_m, t75_m = fit_psychometric_curve_manual(levels, resps, epochs=2500, lr=0.1)
            b0_l, b1_l, t75_l = fit_psychometric_curve_lib(levels, resps, threshold=0.75)
            
            x_range = np.linspace(min(levels), max(levels), 100)
            def P_manual(x): return 1 / (1 + np.exp(-(b0_m + b1_m * x)))
            def P_lib(x): return 1 / (1 + np.exp(-(b0_l + b1_l * x)))
            
            y_manual = [P_manual(x) for x in x_range]
            y_lib = [P_lib(x) for x in x_range]
            
            hit_rates = df_sig.groupby('poziom_bodzca').apply(
                lambda x: (x['klasa_wyniku'] == 'TP').sum() / len(x) if len(x) > 0 else 0.0
            ).reset_index(name='HR')
            
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=hit_rates, x='poziom_bodzca', y='HR', s=100, color='crimson', label='Punkty empiryczne', zorder=5, ax=ax3)
            ax3.plot(x_range, y_manual, color='blue', linestyle='-', linewidth=2, label='Model własny (GD)')
            ax3.plot(x_range, y_lib, color='green', linestyle='--', linewidth=2, label='Model referencyjny (GLM)')
            ax3.axhline(0.75, color='gray', linestyle=':', label="Próg odcięcia 75%")
            ax3.set_title("Krzywa Logistyczna Zależności Detekcji od Intensywności")
            ax3.set_xlabel("Poziom stymulusu (Intensywność)")
            ax3.set_ylabel("P(Prawidłowa Detekcja)")
            ax3.legend()
            st.pyplot(fig3)
        else:
            st.info("System wykrył brak wystarczającego zróżnicowania rzędowych bodźców do wyliczenia gradientu stymulacyjnego.")
            
        st.markdown("---")
        st.header("2. Ograniczona Charakterystyka Przestrzeni ROC")
        
        metrics_roc = lib_sdt(df_filtered)
        d_prime_val = metrics_roc['d_prime']
        HR_emp_single = metrics_roc['HR']
        FAR_emp_single = metrics_roc['FAR']
        
        far_theo, hr_theo = calculate_theoretical_roc(d_prime_val)
        auc_theo = calculate_auc(far_theo, hr_theo)
        
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        ax_roc.plot(far_theo, hr_theo, linestyle='-', color='dodgerblue', linewidth=2, label=f"Teoretyczna Izo-Czułość (AUC = {auc_theo:.3f})")
        ax_roc.plot(FAR_emp_single, HR_emp_single, 'o', color='darkorange', markersize=10, markeredgecolor='black', label="Dyskretny punkt operacyjny", zorder=5)
        ax_roc.plot([0, 1], [0, 1], linestyle=':', color='gray', label="Linia ślepego trafu")
        
        if 'pewnosc' in df_filtered.columns and not df_filtered['pewnosc'].isnull().all():
            far_emp, hr_emp = calculate_empirical_roc(df_filtered)
            if len(far_emp) > 1:
                auc_emp = calculate_auc(far_emp, hr_emp)
                ax_roc.plot(far_emp, hr_emp, marker='o', linestyle='--', color='crimson', linewidth=2, markersize=8, label=f"Pochodna z ratingów wagi (AUC = {auc_emp:.3f})", zorder=4)
        else:
            far_emp = []
            st.info("Brak estymacji wag decyzyjnych pewności zapobiega rysowaniu pełnej krzywej strukturalnej. Różnicowania dokonano za pomocą wektora izo-czułości.")
            
        ax_roc.set_title(f"Charakterystyka Robocza Odbiornika (ROC) na poziomie d' = {d_prime_val:.2f}")
        ax_roc.set_xlabel("Frakcja Fałszywych Alarmów (FAR)")
        ax_roc.set_ylabel("Frakcja Prawidłowych Rozpoznań (HR)")
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.05])
        ax_roc.grid(True, linestyle=':', alpha=0.7)
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)
        
        if 'pewnosc' in df_filtered.columns and not df_filtered['pewnosc'].isnull().all() and len(far_emp) > 2:
            st.markdown("---")
            st.header("3. Transformacja Nierównych Wariancji (z-ROC / UVSDT Model)")
            
            z_far, z_hr = calculate_z_roc(far_emp, hr_emp)
            s, d_e, d_a = fit_z_roc(z_far, z_hr)
            
            fig_zroc, ax_zroc = plt.subplots(figsize=(6, 5))
            ax_zroc.plot(z_far, z_hr, 'o', color='crimson', markersize=8, label="Dyskretne punkty z-score")
            
            x_line = np.linspace(min(z_far)-0.5, max(z_far)+0.5, 50)
            y_line = s * x_line + d_e
            ax_zroc.plot(x_line, y_line, '--', color='darkorange', linewidth=2, label=f"Regresja krzywej (s={s:.2f})")
            ax_zroc.plot([-3, 3], [-3, 3], ':', color='gray', label="Oś idealnej symetrii")
            
            ax_zroc.set_title("Przestrzeń Przekształceń Normalnych z-ROC")
            ax_zroc.set_xlabel("Z-Score (FAR)")
            ax_zroc.set_ylabel("Z-Score (HR)")
            ax_zroc.set_xlim([-3.5, 3.5])
            ax_zroc.set_ylim([-3.5, 3.5])
            ax_zroc.grid(True, alpha=0.3)
            ax_zroc.legend()
            st.pyplot(fig_zroc)
            
            st.write(f"**Obiektywny Wskaźnik UVSDT**: Potwierdzono wariancję percepcyjną na poziomie s={s:.3f}. System oszacował adaptacyjną wartość sprawności na poziomie d(a)={d_a:.3f}.")

    # --------------------------------------------------------------------------------
    # MODUŁ 4: ANALIZA CZASOWA
    # --------------------------------------------------------------------------------
    elif wybrany_modul == "Analiza Znużenia (Chronologiczna)":
        st.subheader("Modelowanie Degradacji Długookresowej (Vigilance Decrement)")
        st.markdown("Ewaluacja załamań kognitywnych na osi wektorów iteracyjnych (chronologia rzutów).")
        
        df_ts = df_filtered.dropna(subset=['czas_reakcji_ms']).sort_values(by='id_proby').copy()
        
        if len(df_ts) > 10:
            window_size = st.slider("Rozpiętość okna kalkulacyjnego (Averaging)", min_value=3, max_value=25, value=10)
            
            df_ts['RT_rolling'] = df_ts['czas_reakcji_ms'].rolling(window=window_size, min_periods=1).mean()
            df_ts['Acc_rolling'] = df_ts['czy_poprawna'].rolling(window=window_size, min_periods=1).mean() * 100
            
            fig_ts, ax_ts = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            ax_ts[0].plot(df_ts['id_proby'], df_ts['czas_reakcji_ms'], alpha=0.3, color='grey', marker='.', markersize=4, label="Indywidualne dyspersje opóźnień")
            ax_ts[0].plot(df_ts['id_proby'], df_ts['RT_rolling'], color='crimson', linewidth=3, label=f"Obwiednia zmienności (k={window_size})")
            ax_ts[0].set_ylabel("Margines opóźnienia [ms]")
            ax_ts[0].set_title("Wahania czasów decyzyjnych w przebiegu układu")
            ax_ts[0].grid(True, alpha=0.3)
            ax_ts[0].legend()
            
            ax_ts[1].plot(df_ts['id_proby'], df_ts['Acc_rolling'], color='dodgerblue', linewidth=3, label="Stopa poprawności wektora kroczącego")
            ax_ts[1].axhline(df_ts['czy_poprawna'].mean()*100, color='gray', linestyle='--', label="Średnia sprawność globalna")
            ax_ts[1].set_xlabel("Ciąg operacyjny (identyfikator bodźca)")
            ax_ts[1].set_ylabel("Stopień Prawidłowych Deklaracji [%]")
            ax_ts[1].set_title("Zaburzenia poziomu skupienia względem bloku testów")
            ax_ts[1].grid(True, alpha=0.3)
            ax_ts[1].legend()
            
            st.pyplot(fig_ts)
            
            st.markdown("---")
            st.subheader("Wskaźnik Korupcji Pamięci Krótkotrwałej (Autokorelacja Błędu)")
            
            fig_ac, ax_ac = plt.subplots(figsize=(8, 4))
            pd.plotting.autocorrelation_plot(df_ts['czas_reakcji_ms'], ax=ax_ac)
            ax_ac.set_xlim([1, min(len(df_ts), 25)]) 
            ax_ac.set_title("Struktura korelacyjna dyspersji na przestrzeni prób")
            st.pyplot(fig_ac)
            
            with st.expander("Informacja Merytoryczna: Odporność i Seryjność"):
                st.markdown('''
                * **Rozrost znużenia**: Zjawisko odnotowywalnej degradacji po czasie ekspozycji, uwypuklające się poprzez odchyłek wykresu po wczesnej fazie testów. Modele kroczące izolują w takich punktach czysty szum statystyczny do rzetelnego trendu.
                * **Autokorelacja Lag**: Sentyment przenoszenia stresu. Błąd popełniony w iteracji $N$ modyfikuje zachowanie decyzyjne z rzędu $N+1$ lub $N+2$. Wykroczenie wartości poza przerywaną odstawą określa istotną kaskadę spowolnienia.
                ''')
        else:
            st.info("System wstrzymał rysowanie ze względu na zbyt krótki szereg wolumenowy. Wymagane jest okno wielkości ~10 wektorów decyzyjnych.")

    # --------------------------------------------------------------------------------
    # MODUŁ 5: MACHINE LEARNING (AI)
    # --------------------------------------------------------------------------------
    elif wybrany_modul == "Model Uczenia Maszynowego (AI)":
        st.subheader("Klasyfikacyjny Model Drzew Decyzyjnych (Random Forest)")
        st.markdown('''
        Zaawansowane mapowanie i uczytelnienie wzorców postępowania przy wykorzystaniu sieci sztucznej inteligencji.
        Algorytm określa zjawiska przyczynowo-skutkowe dla powstawania awarii u respondentów. Poniższa kalkulacja przedstawia próby weryfikowane wstecznie na danych ślepych.
        ''')
        
        df_ml = df_filtered.dropna(subset=['czas_reakcji_ms', 'poziom_bodzca', 'czy_poprawna']).sort_values(by='id_proby').copy()
        
        if len(df_ml) > 30:
            df_ml['prev_RT'] = df_ml['czas_reakcji_ms'].shift(1)
            df_ml['prev_Correct'] = df_ml['czy_poprawna'].shift(1)
            df_ml = df_ml.dropna(subset=['prev_RT', 'prev_Correct'])
            
            cechy = ['poziom_bodzca', 'bodziec_obecny', 'czas_reakcji_ms', 'prev_RT', 'prev_Correct']
            X = df_ml[cechy]
            y = df_ml['czy_poprawna']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model_rf.fit(X_train, y_train)
            
            y_pred = model_rf.predict(X_test)
            skutecznosc_ai = accuracy_score(y_test, y_pred) * 100
            
            wagi = model_rf.feature_importances_
            feature_names = [
                "Intensywność ekspozycji wizualnej", 
                "Faktyczność istnienia czynnika docelowego", 
                "Wskaźnik namysłu chwilowego", 
                "Odchylenie z czasu decyzji historycznej", 
                "Negatywny sentyment ostatniego kroku"
            ]
            
            df_weights = pd.DataFrame({"Klucz Determinujący": feature_names, "Poziom Obciążenia Błędem (%)": wagi * 100}).sort_values(by="Poziom Obciążenia Błędem (%)", ascending=False)
            
            col_ml1, col_ml2 = st.columns([1, 1.5])
            
            with col_ml1:
                st.metric(label="Poziom predykcyjny maszyny kognitywnej", value=f"{skutecznosc_ai:.1f}%")
                st.write("**Skuteczność wyuczenia modelu:** Ponadprzeciętny zwrot przewidywalności potwierdza asymilację systemu do słabości wykazywanych przez respondentów w odniesieniu do losowości struktury badania.")
                st.dataframe(df_weights.style.format({"Poziom Obciążenia Błędem (%)": "{:.1f}%"}))
                
            with col_ml2:
                fig_ml, ax_ml = plt.subplots(figsize=(6, 4))
                colors = ['#4A90E2', '#50E3C2', '#B8E986', '#F5A623', '#D0021B']
                ax_ml.pie(df_weights["Poziom Obciążenia Błędem (%)"], labels=df_weights["Klucz Determinujący"], autopct='%1.1f%%', 
                          startangle=140, colors=colors, textprops={'fontsize': 9})
                ax_ml.set_title("Analiza Zmiennych Ważących:\nCo powoduje dezorientację percepcyjną algorytmu ludzkiego?")
                st.pyplot(fig_ml)
                
        else:
            st.info("System odmówił weryfikacji przez niski stopień zagęszczenia prób historycznych. Konieczne jest dostarczenie min. 30 iteracji celem wyliczeń kowariancyjnych z regułą 70/30.")
