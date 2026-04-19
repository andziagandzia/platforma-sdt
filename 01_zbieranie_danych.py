import csv
import random
from datetime import datetime
from psychopy import visual, core, event, gui


exp_info = {
    'ID_Uczestnika': 'ja_01',
    'ID_Sesji': 'sesja_11',
    'Rodzaj_Bodzca': ['wzrok', 'sluch'],
    'Warunek_Otoczenia': 'po poludniu',
    'Liczba_Prob_Eksperymentu': 100
}

dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title="Panel Startowy SDT (Metoda Stałych Bodźców)")
if dlg.OK == False:
    core.quit()

ID_UCZESTNIKA = exp_info['ID_Uczestnika']
ID_SESJI = exp_info['ID_Sesji']
TYP_BODZCA = exp_info['Rodzaj_Bodzca']
WARUNEK = exp_info['Warunek_Otoczenia']

N_PROB = int(exp_info['Liczba_Prob_Eksperymentu'])
POZIOMY_BODZCA = [0.005, 0.01, 0.03, 0.05, 0.07, 0.09]
CZAS_FIX = 0.5
CZAS_BODZCA = 0.12
OKNO_ODP = 1.5

PLIK_WYNIK = "data_wyniki.csv"

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
    text="Test detekcji bodźca (YES/NO)\n\n"
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


proby = []
for i in range(N_PROB):
    bodziec_obecny = 1 if i < N_PROB // 2 else 0
    poziom = random.choice(POZIOMY_BODZCA) if bodziec_obecny else 0.0
    proby.append((bodziec_obecny, poziom))
random.shuffle(proby)


naglowki = [
    "timestamp", "id_uczestnika", "id_sesji", "id_proby", "typ_bodzca", "warunek",
    "bodziec_obecny", "poziom_bodzca", "odpowiedz", "czas_reakcji_ms",
    "czy_poprawna", "klasa_wyniku", "pewnosc"
]

plik_istnieje = False
try:
    open(PLIK_WYNIK, "r", encoding="utf-8").close()
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

    for idx, (bodziec_obecny, poziom) in enumerate(proby, start=1):

        # 1) fixacja
        fix.draw()
        win.flip()
        core.wait(CZAS_FIX)

        # 2) bodziec / brak
        if bodziec_obecny:
            stim.fillColor = [poziom, poziom, poziom]
            stim.lineColor = [poziom, poziom, poziom]
            stim.draw()
        win.flip()
        core.wait(CZAS_BODZCA)

        # 3) okno odpowiedzi (minimalne)
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
        if keys is None:
            # brak odpowiedzitraktuj jako NIE
            odpowiedz = 0
            rt_ms = ""
            key_pressed = None
        else:
            key_pressed, rt_s = keys[0]
            if key_pressed == "escape":
                break
            odpowiedz = 1 if key_pressed == "t" else 0
            rt_ms = int(rt_s * 1000)
            
            # Zapytanie o pewność
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

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "id_uczestnika": ID_UCZESTNIKA,
            "id_sesji": ID_SESJI,
            "id_proby": idx,
            "typ_bodzca": TYP_BODZCA,
            "warunek": WARUNEK,
            "bodziec_obecny": bodziec_obecny,
            "poziom_bodzca": poziom,
            "odpowiedz": odpowiedz,
            "czas_reakcji_ms": rt_ms,
            "czy_poprawna": czy_poprawna,
            "klasa_wyniku": klasa,
            "pewnosc": pewnosc
        })

win.close()
core.quit()