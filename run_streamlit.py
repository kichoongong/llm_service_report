#!/usr/bin/env python3
"""
Streamlit 자동차 에이전트 실행 스크립트
"""

import subprocess
import sys
import os


def main():
    """Streamlit 앱을 실행합니다."""
    print("🚀 Streamlit 자동차 에이전트를 시작합니다...")

    # 현재 디렉토리를 작업 디렉토리로 설정
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Streamlit 앱 실행
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_car_agent.py",
                "--server.port",
                "8501",
                "--server.address",
                "0.0.0.0",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Streamlit 앱을 종료합니다.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
