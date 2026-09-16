@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set LANGCHOICE=1
echo 1^) English
echo 2^) Русский
set /p LANGCHOICE=^> 
if not "%LANGCHOICE%"=="2" set LANGCHOICE=1

:menu
if "%LANGCHOICE%"=="2" (
  echo.
  echo 🧠 LeanGPT — установщик
  echo 1^) 📦 Установить зависимости
  echo 2^) ⚙️  Настроить пути и токены
  echo 3^) 📚 Скачать корпус Lean4 с GitHub
  echo 4^) 🔤 Обучить токенизатор
  echo 5^) 🏋️ Претрейн модели с нуля
  echo 6^) 🚀 Запустить самообучение
  echo 7^) 📤 Собрать снапшот доказательств
  echo 8^) 🌐 Опубликовать код в GitHub
  echo 9^) 🤖 Запушить доказательства ботом
  echo 0^) 🚪 Выход
  set /p CHOICE=Выбери пункт: 
) else (
  echo.
  echo 🧠 LeanGPT — setup
  echo 1^) 📦 Install dependencies
  echo 2^) ⚙️  Configure paths and tokens
  echo 3^) 📚 Collect Lean4 corpus from GitHub
  echo 4^) 🔤 Train tokenizer
  echo 5^) 🏋️ Pretrain the model from scratch
  echo 6^) 🚀 Run self-play
  echo 7^) 📤 Build a proof snapshot
  echo 8^) 🌐 Publish sources to GitHub
  echo 9^) 🤖 Push proofs via bot
  echo 0^) 🚪 Exit
  set /p CHOICE=Choose an option: 
)

if "%CHOICE%"=="1" goto install_deps
if "%CHOICE%"=="2" goto configure
if "%CHOICE%"=="3" goto collect_corpus
if "%CHOICE%"=="4" goto train_tokenizer
if "%CHOICE%"=="5" goto pretrain
if "%CHOICE%"=="6" goto run_selfplay
if "%CHOICE%"=="7" goto export_proofs
if "%CHOICE%"=="8" goto publish_sources
if "%CHOICE%"=="9" goto bot_push
if "%CHOICE%"=="0" goto bye
goto menu

:check_python
where python >nul 2>nul
if errorlevel 1 (
  if "%LANGCHOICE%"=="2" (echo ❌ Python не найден. Установи Python 3.11+.) else (echo ❌ python not found. Install Python 3.11+.)
  exit /b 1
)
exit /b 0

:install_deps
call :check_python
if errorlevel 1 goto menu
python -m pip install -r requirements.txt
echo ✅
goto menu

:configure
if "%LANGCHOICE%"=="2" (
  set /p REPL_BIN_IN=Путь к бинарнику Lean REPL: 
  set /p PROJECT_IN=Путь к твоему lake-проекту: 
  set /p TOKEN_IN=GitHub token (можно пусто): 
  set /p SOURCES_IN=Ссылка на репозиторий Исходники (можно пусто): 
  set /p PROOFS_IN=Ссылка на репозиторий Доказательства (можно пусто): 
) else (
  set /p REPL_BIN_IN=Path to your Lean REPL binary: 
  set /p PROJECT_IN=Path to your lake project: 
  set /p TOKEN_IN=GitHub token (can be empty): 
  set /p SOURCES_IN=Sources repo URL (can be empty): 
  set /p PROOFS_IN=Proofs repo URL (can be empty): 
)
(
  echo repl_bin: %REPL_BIN_IN%
  echo lean_project_dir: %PROJECT_IN%
  echo env_import: null
  echo checkpoint_path: data/checkpoints/lean_gpt_latest.pt
  echo tokenizer_dir: data/tokenizer
  echo results_jsonl: data/results.jsonl
  echo github_token: %TOKEN_IN%
  echo proofs_repo: %PROOFS_IN%
  echo sources_repo: %SOURCES_IN%
) > config.local.yaml
echo ✅
goto menu

:collect_corpus
call :check_python
if errorlevel 1 goto menu
python scripts\collect_corpus.py
echo ✅
goto menu

:train_tokenizer
call :check_python
if errorlevel 1 goto menu
python -m leangpt.train_tokenizer
echo ✅
goto menu

:pretrain
call :check_python
if errorlevel 1 goto menu
python -m leangpt.train
echo ✅
goto menu

:run_selfplay
call :check_python
if errorlevel 1 goto menu
python scripts\run_pipeline.py
echo ✅
goto menu

:export_proofs
call :check_python
if errorlevel 1 goto menu
python scripts\export_proofs.py
echo ✅
goto menu

:publish_sources
where git >nul 2>nul
if errorlevel 1 (echo git not found & goto menu)
bash scripts\publish.sh
echo ✅
goto menu

:bot_push
where git >nul 2>nul
if errorlevel 1 (echo git not found & goto menu)
bash scripts\bot_push_proofs.sh
echo ✅
goto menu

:bye
if "%LANGCHOICE%"=="2" (echo 👋 Пока!) else (echo 👋 Bye!)
exit /b 0
