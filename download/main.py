import os
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(1, os.getcwd())
    print(os.getcwd())

from manager_1d import DownloadManager
from manager_1d import UIManager
import manager_1d

if __name__ == "__main__":
    for idx, path in enumerate(sys.path):
        print(f"{idx}: {path}")

    # 데이터 저장 경로
    os.makedirs(manager_1d.CANDLE_1D_PATH, exist_ok=True)

    # 다운로드할 종목들
    tickers = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]

    # 다운로드 관리 및 UI
    download_manager = DownloadManager(tickers)
    ui_manager = UIManager(download_manager.ticker_status)

    # 데이터 다운로드 및 UI 업데이트
    download_manager.update_all_tickers()  # 데이터를 다운로드하고 상태 업데이트
    ui_manager.run()  # GUI 팝업 실행
