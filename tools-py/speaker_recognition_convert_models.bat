@echo off
setlocal
echo =======================================================
echo   Speaker Recognition ESP32-P4 - Model Converter
echo =======================================================
echo.
echo Converting pure PyTorch models in models/py/ to ESPM...
echo.

cd /d "%~dp0"

echo [1/2] Exporting ECAPA-TDNN512 and TDNN-Tiny INT8...
python speaker_recognition_models_converter\speaker_recognition_convert_models.py --model all --precision int8
if errorlevel 1 (
    echo.
    echo [ERROR] Model conversion INT8 failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Exporting ECAPA-TDNN512 and TDNN-Tiny FP16...
python speaker_recognition_models_converter\speaker_recognition_convert_models.py --model all --precision fp16
if errorlevel 1 (
    echo.
    echo [ERROR] Model conversion FP16 failed.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo Done! Models are saved in:
echo   - models/espm/ecapa_tdnn512_int8.espm
echo   - models/espm/tdnn_tiny_int8.espm
echo   - models/espm/ecapa_tdnn512_fp16.espm
echo   - models/espm/tdnn_tiny_fp16.espm
echo =======================================================
pause
exit /b 0
