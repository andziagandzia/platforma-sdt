import os
import csv
import random
import numpy as np
from datetime import datetime

# Ścieżka docelowa pliku z bazą z prób
PLIK_WYNIK = "data_wyniki.csv"

# Parametry na nowe symulowane sesje
SIM_SESSIONS = [
    {"id_sesji": "sesja_10_sim", "warunek": "skupiony (d=2.0)", "d_prime": 2.0},
    {"id_sesji": "sesja_11_sim", "warunek": "sredni (d=1.0)",   "d_prime": 1.0},
    {"id_sesji": "sesja_12_sim", "warunek": "zmeczony (d=0.2)", "d_prime": 0.2}
]

N_PROJEKT = 100 # Ilość prób na pojedynczą sesję
POZIOMY = [0.005, 0.01, 0.03, 0.05, 0.07, 0.09]

def klasyfikuj(bodziec_obecny: int, odpowiedz: int) -> str:
    if bodziec_obecny == 1 and odpowiedz == 1: return "TP"
    if bodziec_obecny == 1 and odpowiedz == 0: return "FN"
    if bodziec_obecny == 0 and odpowiedz == 1: return "FP"
    return "TN"

def symuluj_odpowiedz_i_pewnosc(bodziec_obecny, d_prime):
    """
    Używa ustrukturyzowanej teorii sygnałów z równolicznym szumem (SDT Equal-Variance).
    x ~ N(mean, 1), gdzie mean to 0 dla hałasu lub d' dla sygnału.
    """
    mean = d_prime if bodziec_obecny == 1 else 0.0
    x = np.random.normal(loc=mean, scale=1.0)
    
    # Optymalne kryterium z reguły maksimum precyzyjności
    c_dec = d_prime / 2.0
    
    # Wybór binarny - odpowiedź na zewnątrz (Tak=1, Nie=0)
    odpowiedz = 1 if x >= c_dec else 0
    
    # Ocena pewności od 1 do 5 skalowana absolutnym dystansem od progu (jak bardzo sygnał przewyższył/zatopił się)
    dist = abs(x - c_dec)
    
    # Zmodyfikowane progi żeby równomierniej wyłapywać odpowiedzi "Średnie"
    if dist > 1.5: pewnosc = 5       # Całkowicie pewnie
    elif dist > 0.9: pewnosc = 4     # Dość pewnie
    elif dist > 0.5: pewnosc = 3     # Średnio
    elif dist > 0.15: pewnosc = 2    # Słabo
    else: pewnosc = 1                # Wcale (zgadywanie rzutem na taśmę)
    
    # Symulacja średniego czasu reakcji
    # Im bliżej kryterium tym trudniejsza decyzja, czyli wolniejsza
    bazowy_czas = 300
    rt_ms = int(bazowy_czas + (400 * np.exp(-dist))) + random.randint(-50, 100)
    
    return odpowiedz, pewnosc, rt_ms


def generuj_sesje():
    print(f"Otwieranie pliku {PLIK_WYNIK} do nadpisania (tryb append)...")
    
    naglowki = [
        "timestamp", "id_uczestnika", "id_sesji", "id_proby", "typ_bodzca", "warunek",
        "bodziec_obecny", "poziom_bodzca", "odpowiedz", "czas_reakcji_ms",
        "czy_poprawna", "klasa_wyniku", "pewnosc"
    ]
    
    # Sprawdzam czy mamy pusty plik
    plik_istnieje = os.path.exists(PLIK_WYNIK)
    if plik_istnieje and os.path.getsize(PLIK_WYNIK) == 0:
        plik_istnieje = False
        
    with open(PLIK_WYNIK, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=naglowki)
        if not plik_istnieje:
            writer.writeheader()
            
        for sesja in SIM_SESSIONS:
            print(f"Generowanie sesji: {sesja['id_sesji']} (warunek = {sesja['warunek']})")
            
            for nr in range(1, N_PROJEKT + 1):
                bodziec_obecny = 1 if nr <= (N_PROJEKT // 2) else 0
                poziom = random.choice(POZIOMY) if bodziec_obecny else 0.0
                
                odp, pewn, rt = symuluj_odpowiedz_i_pewnosc(bodziec_obecny, sesja['d_prime'])
                
                czy_popr = 1 if odp == bodziec_obecny else 0
                klasa = klasyfikuj(bodziec_obecny, odp)
                
                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "id_uczestnika": "symulacja_02",    # Odseparujmy ich od starych
                    "id_sesji": sesja["id_sesji"],
                    "id_proby": nr,
                    "typ_bodzca": "wzrok_symulowany",
                    "warunek": sesja["warunek"],
                    "bodziec_obecny": bodziec_obecny,
                    "poziom_bodzca": poziom,
                    "odpowiedz": odp,
                    "czas_reakcji_ms": rt,
                    "czy_poprawna": czy_popr,
                    "klasa_wyniku": klasa,
                    "pewnosc": pewn
                })
                
    print(f"\\nUkończono pisanie do skryptu! Zapisano wpisy ({len(SIM_SESSIONS)*N_PROJEKT} prób).")


if __name__ == "__main__":
    generuj_sesje()
