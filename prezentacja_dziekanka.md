# Konspekt Prezentacji Inżynierskiej: Część I (Założenia Pracy)

Poniżej znajduje się gotowy tekst i struktura prezentacji. Została ona podzielona na "slajdy", abyś mógł/mogła łatwo przekopiować zawartość bezpośrednio do pól tekstowych w Canvie. Miejsca w nawiasach kwadratowych `[...]` należy wypełnić własnymi danymi (np. nazwisko promotora).

---

## Slajd 1: Strona Tytułowa
* **Tytuł:** Projekt i implementacja platformy analitycznej do modelowania danych psychofizycznych w oparciu o Teorię Detekcji Sygnałów (SDT)
* **Autor:** [Twoje Imię i Nazwisko]
* **Promotor:** [Tytuł i Imię Nazwisko Promotora]
* **Temat prezentacji:** Założenia i cele pracy inżynierskiej – Część I

---

## Slajd 2: Problem badawczy i technologiczny
**(Jaki problem techniczny lub organizacyjny starasz się rozwiązać?)**

Współczesne badania neuroinformatyczne i kognitywne cierpią na rozdrobnienie narzędziowe operacji. 
* Zbieranie danych od pacjenta odbywa się w jednym oprogramowaniu.
* Analityka matematyczna odbywa się w innym (często drogim lub zamkniętym) środowisku.
* Wiele programów używa metryk jako tzw. „czarnych skrzynek”, ukrywając przed badaczem dokładny proces wyliczania wskaźników psychofizycznych.
* **Rozwiązanie problemu:** Stworzenie zunifikowanego, nowoczesnego i w pełni transparentnego środowiska "End-to-End", w którym od razu po wykonaniu testu badawczego, z poziomu przejrzystej i zautomatyzowanej aplikacji przeprowadzana jest potężna analityka matematyczna.

---

## Slajd 3: Finalny produkt inżynierski
**(Co będzie finalnym produktem Twojej pracy?)**

Pełnoprawna **Platforma Analityczna SDT (Interaktywny Dashboard)** z graficznym panelem obsługi, w skład której wchodzą:
1. **Moduł Rejestracyjno-Badawczy:** Zintegrowane środowisko pobudzające do przeprowadzania eksperymentów wzrokowych na ludziach (w tym zaawansowana metoda adaptacyjna „Staircase”).
2. **Moduł Statystyczny i Machine Learning:** Posiadający algorytm wyliczający na żywo:
   * Wykrywalność (Hit Rate, FAR), wrażliwość decyzyjną (d') i kryterium (c).
   * Dopasowanie modeli wektorowych do Krzywych Psychometrycznych i Charakterystyk Odbiornika ROC.
   * Szeregi Czasowe badające degradację pamięci czy zmęczenia ubadanego pacjenta.
   * Model predykcyjny oparty o Drzewa Losowe (AI), przewidujący błędy pacjenta i wskazujący kluczowe powody takich zachowań w matrycy oceny uczącego.

---

## Slajd 4: Grupa docelowa
**(Dla kogo ta praca jest użyteczna?)**

Rozwiązanie ma systematyzujący na polu nauki charakter interdyscyplinarny. Głównymi odbiorcami są:
* **Inżynierowie Danych i Programiści MDI** – zajmujący się skalowaniem procesów obróbki testów psychologicznych.
* **Badacze psychofizyki, neurobiolodzy i otolaryngolodzy** – potrzebujący darmowego systemu testów decyzyjnych po to by m.in badać cząsteczki widzenia (czy ślepotę informacyjną).
* **Środowisko uniwersyteckie (studenci)** – jako platforma edukacyjna demonstrująca transparentność klasyków z wzorami statystycznymi SDT od "kuchni kodu".

---

## Slajd 5: Teza i cel weryfikacyjny pracy
**(Czy chcesz coś udowodnić lub sprawdzić w ramach pracy?)**

**Główne cele badawcze to inżynierskie udowodnienie zbieżności dwóch paradygmatów, że:**
1. Ręcznie zaimplementowane, transparentne równania algorytmiczne statystyki (rozkładane od podstaw dla analizy dystrybuanty wewnątrz kodu Python) **dają w zasadzie identyczny, niezaburzony rezultat w liczeniu psychofizyki** co ciężkie biblioteki numeryczne udostępniane globalnie.
2. Współczesne modele uczenia maszynowego są w stanie bezbłędnie zmapować nawyki oraz zmęczenie umysłowe (długie okna opóźnienia, czas R/T, wcześniejsze pudła) badanych i przewidzieć ich pomyłkę z wysoką sprawdzalnością jeszcze zanim uczestnik wdroży tę pomyłkę po bodźcu w eksperymencie.

---

## Slajd 6: Technologie i stos użytych narzędzi
**(Jakie konkretne narzędzia, technologie lub materiały zostaną wykorzystane?)**

Zastosowano stabilny, zorientowany wektorowo Data-Science stos dla języka Python:
* **Tworzenie interfejsu (Dashboard UI):** *Streamlit* – dla korporacyjnego widoku raportów i GUI.
* **Środowisko wykonawcze eksperymentów:** Silnik wbudowany *PsychoPy* używany w neuroscience.
* **Transformacja danych i macierzy:** Moduły *Pandas* i *NumPy*.
* **Wizualizacja modeli matematycznych:** Biblioteki wektorowe *Matplotlib* / *Seaborn* rozrysowujące ROC i krzywe estymacyjne Odbiornika.
* **Obliczenia logistyczne:** Półsurowe algorytmy Gradient Descent własnej implementacji w konfrontacji ze sprawdzonym statystkami modułów *SciPy* oraz *Statsmodels.GLM*.
* **Sieci kognitywnej analityki AI:** Moduł operacyjny *Scikit-Learn* (dla modelu RandomForest Classifier wspierającego predykcję).

---

## Slajd 7: Procedura pozyskiwania Danych
**(Jakie dane będziesz badać i skąd je pobierzesz?)**

Platforma udowodni swoją sprawność w oparciu o żywy strumień danych.
* **Skąd pobiorę:** Utworzony eksperyment komputerowy wygeneruje je empirycznie – zapraszając prawdziwego uczestnika do panelu przed ekran ("moduł PsychoPy"), który udokumentuje plik po podkomisji bazy w ujednoliconym trybie csv.
* **Badane zbiory testów do analizy to np:** 
    * Czasy absolutnych opóźnień reakcji użytkownika tzw. R/T.
    * Klasyfikacja trafień (Ominięcie znaku, strzał trafiony, FA).
    * Bity obiektywnego statusu występowania sygnału przed pacjentem (prawda/fałsz).
    * Ocena subiektywnej pewności u badanego wpisywana do systemu numerycznie (skala 1-4).

---

## Slajd 8: Ograniczenia i ramy usterkowe rozwiązania
**(Jakie są ograniczenia Twojego rozwiązania?)**

* **Prawo Wymaganej Skali (Dane):** Posiłkowa struktura wektoryzacji modeli predykcyjnych SDT wymaga min. 30 ugruntowanych przejść rycina-ocena by w ogóle AI nie uległa overfittingowi statystycznemu, uniemożliwiając ocenę na bardzo krótkich pilotach próbowania sprzętu. Moduł zapobiegnie temu wysyłając odpowiednie info z ostrzeżeniem dla użytkownika jeśli badzo mało testował i uciął badanie.
* **Ekstrema otoczeń badanych:** W zrobotyzowanym module pobierania prób nie jesteśmy w pełni określić (na dystans bez kamery) czy np. wybuch FA to naturalny błąd, czy faktycznie użytkownik odwrócił na minutę wzrok poza ekran na swojego smartfona i klikał ślepo klawisze psując całe serie chronologii. Metodyka SDT w samej swej zasłonie ma być niewrażliwa na obiektywne ślepe trafienia (eliminacja guessing strategy poprzez kryterium 'c'), jednak fizyczny brak opaski neurofali (EEG/EyeTracker) wymusza szacowanie odchyleń tylko w drodze poślizgu czasowego zadeklarowanych R/T.
