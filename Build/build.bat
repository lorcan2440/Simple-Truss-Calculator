cd Build

:: ensure pyinstaller is updated in order to use the (currently experimental) --splash flag
pip install --upgrade pyinstaller

:: build with basic optimisations
python -O -m PyInstaller                                                                    ^
    ../Source/Truss_Calculator.py                                                           ^
    --onefile                                                                               ^
    --splash ../Media/TrussBridge.jpg                                                       ^
    --icon ../Media/TrussIcon.ico                                                           ^  
    --upx-dir C:\Users\lnick\Downloads\upx-3.96-win64\upx-3.96-win64                        ^
    --noconsole

echo " --- If the build completed succesfully, it can be found in the `dist` folder. --- "