# ui_manager.py
import tkinter as tk
from tkinter import messagebox

# download_manager.py
import yfinance as yf
from datetime import datetime

import utils

CANDLE_1D_PATH = utils.file_path("saves/candle_1d")


class DownloadManager:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data_dir = CANDLE_1D_PATH
        self.ticker_status = {ticker: {"status": "pending", "error": None} for ticker in tickers}

    def download_data(self, ticker):
        try:
            print(f"🔄 {ticker} 데이터 업데이트 시작...")
            data = yf.download(ticker, period="10y", interval="1d")
            file_name = f"{self.data_dir}/{ticker}_daily_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            data.to_csv(file_name)

            self.ticker_status[ticker]["status"] = "success"
            self.ticker_status[ticker]["error"] = None
            print(f"✅ {ticker} 데이터 다운로드 성공! -> {file_name}")

        except Exception as e:
            self.ticker_status[ticker]["status"] = "failed"
            self.ticker_status[ticker]["error"] = str(e)
            print(f"❌ {ticker} 데이터 다운로드 실패: {e}")

    def update_all_tickers(self):
        for ticker in self.tickers:
            self.download_data(ticker)

    def retry_failed_updates(self):
        for ticker, status in self.ticker_status.items():
            if status["status"] == "failed":
                print(f"🔄 {ticker} 재시도...")
                self.download_data(ticker)



class UIManager:
    def __init__(self, ticker_status):
        self.ticker_status = ticker_status
        self.root = tk.Tk()
        self.root.title("업데이트 현황")
        self.root.geometry("300x300")
        self.create_ui()

    def create_ui(self):
        self.status_text = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_text, font=("Helvetica", 10))
        self.status_label.pack(pady=20)

        self.update_button = tk.Button(self.root, text="업데이트 시작", command=self.start_update)
        self.update_button.pack(pady=10)

        self.quit_button = tk.Button(self.root, text="종료", command=self.root.quit)
        self.quit_button.pack(pady=10)

    def start_update(self):
        self.status_text.set("업데이트 진행 중...")
        self.root.update()  # UI 갱신
        self.update_status()

    def update_status(self):
        status_message = "\n".join([f"{ticker}: {status['status']}" for ticker, status in self.ticker_status.items()])
        self.status_text.set(status_message)
        self.root.after(1000, self.update_status)  # 1초마다 업데이트 현황 갱신

    def run(self):
        self.root.mainloop()
