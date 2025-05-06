import yfinance as yf
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from log import logger
from config import Config

class StockDownloader:
    @staticmethod
    def download_single(stock):
        try:
            file_path = Config.DATA_DIR / f"{stock}.parquet"  # 使用parquet格式节省空间

            if file_path.exists():
                logger.info(f"已存在，跳过: {stock}")
                return True, stock

            full_stock = f"{stock}.HK"

            df = yf.Ticker(full_stock).history(
                start=Config.START_DATE,
                end=Config.END_DATE,
                # interval=Config.INTERVAL,
                auto_adjust=Config.AUTO_ADJUST
            )

            if df.empty:
                logger.error(f"下载数据为空: {full_stock}")
                return False, stock

            df.to_parquet(file_path)
            logger.info(f"下载成功: {full_stock}")
            return True, stock
        except Exception as e:
            logger.error(f"下载失败 {full_stock}: {str(e)}")
            return False, stock

    @classmethod
    def download_all(cls, max_workers=5, retry=2):
        failed_stocks = []

        for attempt in range(retry + 1):
            if attempt > 0:
                logger.warning(f"开始第{attempt}次重试，失败股票数量:{len(failed_stocks)}")
                if not failed_stocks:
                    break
                stocks_to_retry = failed_stocks.copy()
                failed_stocks = []
            else:
                stocks_to_retry = Config.STOCKS

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(cls.download_single, stock): stock
                    for stock in stocks_to_retry
                }

                for future in tqdm(
                        as_completed(futures),
                        total=len(stocks_to_retry),
                        desc=f"下载进度(尝试{attempt + 1} / {retry + 1}"
                ):
                    success, stock = future.result()
                    if not success:
                        failed_stocks.append(stock)
            if not failed_stocks:
                break
        return failed_stocks