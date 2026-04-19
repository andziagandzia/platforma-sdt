import csv
import random
from datetime import datetime
from psychopy import visual, core, event, gui


exp_info = {
    'ID_Uczestnika': 'ja_01',
    'ID_Sesji': 'sesja_staircase_01',
    'Rodzaj_Bodzca': ['wzrok', 'sluch'],  # Lista rozwijana na wzrok / słuch
    'Warunek_Otoczenia': 'po poludniu (staircase)',
    'Liczba_Prob_Eksperymentu': 100
}

dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title="Panel Startowy SDT (Metoda Schodkowa)")
if dlg.OK == False:
    core.quit()  # Użytkownik wcisnął "Cancel"

ID_UCZESTNIKA = exp_info['ID_Uczestnika']
ID_SESJI = exp_info['ID_Sesji']
TYP_BODZCA = exp_info['Rodzaj_Bodzca']
WARUNEK = exp_info['Warunek_Otoczenia']
N_PROB = int(exp_info['Liczba_Prob_Eksperymentu'])
CZAS_FIX = 0.5
CZAS_BODZCA = 0.12
OKNO_ODP = 1.5
PLIK_WYNIK = "data_wyniki.csv"


aktualny_poziom = 0.10     # Zaczynamy od dość łatwego poziomu
mnoznik_w_gore = 1.25      # O ile rośnie bodziec po błędzie
mnoznik_w_dol = 0.80       # O ile maleje po 2 sukcesach
min_poziom = 0.001
max_poziom = 0.5

sukcesy_z_rzedu = 0

def klasyfikuj(bodziec_obecny: int, odpowiedz: int) -> str:
    if bodziec_obecny == 1 and odpowiedz == 1: return "TP"
    if bodziec_obecny == 1 and odpowiedz == 0: return "FN"
    if bodziec_obecny == 0 and odpowiedz == 1: return "FP"
    return "TN"


win = visual.Window(fullscr=True, units="height")
fix = visual.TextStim(win, text="+", height=0.08)
stim = visual.Rect(win, width=0.06, height=0.06)

instr = visual.TextStim(
    win,
    text="Test Adaptacyjny - Metoda Schodkowa\n\n"
         "T = TAK (bodziec był)\n"
         "N = NIE (bodźca nie było)\n\n"
         "SPACJA = start\nESC = wyjście",
    height=0.05
)

prompt = visual.TextStim(
    win,
    text="T = TAK     |     N = NIE",
    height=0.06,
    pos=(0, -0.28)
)

# Układamy harmonogram prób Sygnał/Szum i tasujemy - sam Poziom jest jednak wyznaczany w pętli
proby_obecnosc = [1] * (N_PROB // 2) + [0] * (N_PROB - N_PROB // 2)
random.shuffle(proby_obecnosc)


naglowki = [
    "timestamp", "id_uczestnika", "id_sesji", "id_proby", "typ_bodzca", "warunek",
    "bodziec_obecny", "poziom_bodzca", "odpowiedz", "czas_reakcji_ms",
    "czy_poprawna", "klasa_wyniku", "pewnosc"
]

plik_istnieje = False
try:
    with open(PLIK_WYNIK, "r", encoding="utf-8") as f:
        pass
    plik_istnieje = True
except FileNotFoundError:
    pass

instr.draw()
win.flip()
key = event.waitKeys(keyList=["space", "escape"])[0]
if key == "escape":
    win.close()
    core.quit()

with open(PLIK_WYNIK, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=naglowki)
    if not plik_istnieje:
        writer.writeheader()

    for idx, bodziec_obecny in enumerate(proby_obecnosc, start=1):
        
        # Jasność dla tej konkretnej próby
        renderowany_poziom = aktualny_poziom if bodziec_obecny else 0.0

        # 1) fixacja
        fix.draw()
        win.flip()
        core.wait(CZAS_FIX)

        # 2) bodziec / brak
        if bodziec_obecny:
            stim.fillColor = [renderowany_poziom, renderowany_poziom, renderowany_poziom]
            stim.lineColor = [renderowany_poziom, renderowany_poziom, renderowany_poziom]
            stim.draw()
        win.flip()
        core.wait(CZAS_BODZCA)

        # 3) okno odpowiedzi 
        event.clearEvents()
        clock = core.Clock()

        prompt.draw()
        win.flip()

        keys = event.waitKeys(
            maxWait=OKNO_ODP,
            keyList=["t", "n", "escape"],
            timeStamped=clock
        )

        pewnosc = ""
        odpowiedz = 0 # default
        rt_ms = ""
        
        if keys is None:
            # brak odpowiedzi to NIE
            odpowiedz = 0
            key_pressed = None
        else:
            key_pressed, rt_s = keys[0]
            if key_pressed == "escape":
                break
            odpowiedz = 1 if key_pressed == "t" else 0
            rt_ms = int(rt_s * 1000)
            
            # Zapytanie o pewność z klawiatury
            prompt_pewnosc = visual.TextStim(
                win,
                text="Na ile pewna/y jesteś swojej odpowiedzi?\n\n1 - Wcale\n2 - Słabo\n3 - Średnio\n4 - Dość pewnie\n5 - Całkowicie pewnie\n\n(Wybierz 1-5)",
                height=0.05,
                pos=(0, 0)
            )
            prompt_pewnosc.draw()
            win.flip()
            event.clearEvents()
            keys_p = event.waitKeys(keyList=["1", "2", "3", "4", "5", "escape"], maxWait=OKNO_ODP * 2)
            if keys_p is not None:
                if keys_p[0] == "escape":
                    break
                pewnosc = int(keys_p[0])

        czy_poprawna = 1 if odpowiedz == bodziec_obecny else 0
        klasa = klasyfikuj(bodziec_obecny, odpowiedz)


        if bodziec_obecny == 1:
            if czy_poprawna == 1:  # Trafienie (TP)
                sukcesy_z_rzedu += 1
                if sukcesy_z_rzedu == 2:
                    # 2-Down -> utrudniamy
                    aktualny_poziom = max(min_poziom, aktualny_poziom * mnoznik_w_dol)
                    sukcesy_z_rzedu = 0
            else: # Pomyłka (FN)
                # 1-Up -> ułatwiamy
                aktualny_poziom = min(max_poziom, aktualny_poziom * mnoznik_w_gore)
                sukcesy_z_rzedu = 0
        # Hałas (Szum - 0) nie zmienia trudności bodźców na schodkach - to filar SDT.
        
        # Zapisz do loga
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "id_uczestnika": ID_UCZESTNIKA,
            "id_sesji": ID_SESJI,
            "id_proby": idx,
            "typ_bodzca": TYP_BODZCA,
            "warunek": WARUNEK,
            "bodziec_obecny": bodziec_obecny,
            "poziom_bodzca": round(renderowany_poziom, 4),
            "odpowiedz": odpowiedz,
            "czas_reakcji_ms": rt_ms,
            "czy_poprawna": czy_poprawna,
            "klasa_wyniku": klasa,
            "pewnosc": pewnosc
        })

win.close()
core.quit()
