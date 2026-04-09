@echo off
chcp 65001 >nul
REM ============================================================
REM  AstroPipeline — Windows 빌드 스크립트
REM  결과물: dist\AstroPipeline.exe  (단일 실행 파일)
REM
REM  사전 준비:
REM    1. Python 3.10+ 설치 (python.org) — "Add to PATH" 체크
REM    2. requirements.txt 설치:
REM         pip install -r requirements.txt
REM    3. 이 배치 파일을 프로젝트 루트에서 실행
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set APP_NAME=AstroPipeline
set DIST_DIR=dist
set BUILD_DIR=build

echo ======================================================
echo   AstroPipeline Windows Build
echo ======================================================

REM ── 1. Python 확인 ───────────────────────────────────────────
echo.
echo [1/5] Python 확인...
python -c "import sys; print(f'Python {sys.version}')" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python이 PATH에 없습니다.
    echo python.org 에서 Python 3.10+ 를 설치하고 "Add to PATH"를 체크하세요.
    pause
    exit /b 1
)
python -V

REM ── 2. 의존성 확인 ────────────────────────────────────────────
echo.
echo [2/5] 빌드 의존성 확인...

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo   PyInstaller 설치 중...
    python -m pip install pyinstaller pyinstaller-hooks-contrib
) else (
    echo   PyInstaller OK
)

python -c "import _pyinstaller_hooks_contrib" >nul 2>&1
if errorlevel 1 (
    echo   pyinstaller-hooks-contrib 설치 중...
    python -m pip install pyinstaller-hooks-contrib
)

REM ── 3. 아이콘 안내 (선택) ────────────────────────────────────
echo.
echo [3/5] 아이콘 확인...
if exist "gui\icons\app_icon.ico" (
    echo   app_icon.ico 발견 - 아이콘이 적용됩니다.
) else (
    echo   app_icon.ico 없음 - 기본 아이콘 사용
)

REM ── 4. 이전 빌드 정리 ─────────────────────────────────────────
echo.
echo [4/5] 이전 빌드 정리 중... (파일이 사용 중이면 실패할 수 있습니다)
taskkill /f /im %APP_NAME%.exe >nul 2>&1
timeout /t 1 /nobreak >nul

if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%" 2>nul
if exist "%DIST_DIR%"  rmdir /s /q "%DIST_DIR%"  2>nul

if exist "%DIST_DIR%\%APP_NAME%.exe" (
    echo ERROR: 기존 %APP_NAME%.exe를 삭제할 수 없습니다. 프로그램을 종료한 뒤 다시 시도하세요.
    pause
    exit /b 1
)
echo   완료

REM ── 5. 빌드 ──────────────────────────────────────────────────
echo.
echo [5/5] PyInstaller 빌드 시작...
echo   (scipy/astropy 수집에 수 분 소요될 수 있습니다)
echo.

if not exist "astro_pipeline.spec" (
    echo ERROR: astro_pipeline.spec 파일이 없습니다.
    echo 처음 빌드하는 경우 'python -m PyInstaller --onefile main_script.py'를 먼저 실행하세요.
    pause
    exit /b 1
)

python -m PyInstaller --clean astro_pipeline.spec
if errorlevel 1 (
    echo.
    echo ERROR: 빌드 실패
    echo 위의 오류 메시지를 확인하세요.
    pause
    exit /b 1
)

REM ── 결과 확인 ─────────────────────────────────────────────────
set EXE_PATH=%DIST_DIR%\%APP_NAME%.exe
if exist "%EXE_PATH%" (
    echo.
    echo ======================================================
    echo   빌드 성공!
    echo   경로: %EXE_PATH%
    echo ======================================================
    echo.
    echo 실행 방법: %EXE_PATH% 를 더블클릭
    echo.
    echo 참고: 첫 실행 시 %%TEMP%% 에 파일을 압축 해제하므로
    echo       5~15초 소요됩니다. (이후 실행은 빠름)
    echo.
    echo 배포 시: %EXE_PATH% 단일 파일만 전달하면 됩니다.
) else (
    echo.
    echo ERROR: %EXE_PATH% 가 생성되지 않았습니다.
    pause
    exit /b 1
)

pause
