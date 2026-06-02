@echo off
chcp 65001 >nul
cd /d D:\222
echo ================================================
echo  每日自动学习 %date% %time%
echo ================================================

echo.
echo [1/2] 涨停股增量学习...
python train_model.py --daily
if %errorlevel% neq 0 echo [WARN] 涨停股学习异常

echo.
echo [2/2] 全A股最新数据学习...
python train_model.py --learn-all
if %errorlevel% neq 0 echo [WARN] 全量学习异常

echo.
echo ================================================
echo  学习完成 %date% %time%
echo ================================================
