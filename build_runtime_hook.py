"""PyInstaller runtime hook.

실행 시 sys._MEIPASS(압축 해제 임시 디렉터리)를 sys.path 앞에 추가합니다.
이렇게 해야 'gui', 'pipeline' 패키지를 frozen 상태에서도 올바르게 import합니다.
"""
import sys
from pathlib import Path

if hasattr(sys, "_MEIPASS"):
    meipass = sys._MEIPASS
    if meipass not in sys.path:
        sys.path.insert(0, meipass)
