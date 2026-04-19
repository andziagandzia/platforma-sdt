@echo off
echo ==== Uruchamianie Srodowiska Wirtualnego ====
call .venv\Scripts\activate.bat
echo ==== Startowanie aplikacji Dashboard ====
streamlit run 06_dashboard.py
pause
