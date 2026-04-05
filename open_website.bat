@echo off
cd /d C:\Users\kanar\OneDrive\Desktop\ai_news_agent
start "" http://127.0.0.1:8000
call venv\Scripts\python.exe news_agent.py
pause
