@echo off
echo.
echo =================================
echo 🚀 ImageCapSeg Local Launcher  
echo =================================
echo.

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found. Please run this script from the ImageCapSeg directory.
    pause
    exit /b 1
)

REM Check if conda environment exists
conda info --envs | findstr "visionml" >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda environment 'visionml' not found.
    echo 💡 Please create the environment first:
    echo    conda create -n visionml python=3.10
    echo    conda activate visionml  
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Found visionml environment
echo.

REM Create necessary directories
if not exist "models" mkdir models
if not exist "uploads" mkdir uploads
if not exist "temp_images" mkdir temp_images
echo ✅ Directories ready
echo.

echo Choose how to run ImageCapSeg:
echo.
echo 1) Run ImageCapSeg Now! (Streamlit)
echo 2) Test Application First
echo 3) Install/Update Dependencies  
echo 4) Quick Run (Skip Environment Check)
echo 5) Exit
echo.

set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto test
if "%choice%"=="3" goto install
if "%choice%"=="4" goto quickrun
if "%choice%"=="5" goto exit

echo Invalid choice. Please try again.
pause
exit /b 1

:streamlit
echo.
echo 🚀 Starting ImageCapSeg with Streamlit...
echo 💡 The app will open in your browser automatically
echo 🛑 Press Ctrl+C in this window to stop the app
echo.
call conda activate visionml && streamlit run app.py
goto end

:test
echo.
echo 🧪 Testing ImageCapSeg application...
call conda activate visionml && python test_app.py
goto end

:install
echo.
echo 📦 Installing dependencies...
call conda activate visionml && pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Installation failed
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully
goto end

:quickrun
echo.
echo ⚡ Quick launching ImageCapSeg...
echo 💡 If you get errors, use option 1 instead
echo.
streamlit run app.py
goto end

:exit
echo.
echo 👋 Goodbye!
exit /b 0

:end
echo.
echo 💡 You can run this script again anytime to launch ImageCapSeg
pause