from pathlib import Path
from stock_list import HK_TECH_STOCKS

class Config:
    STOCKS = HK_TECH_STOCKS

    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data/stocks"
    LOG_DIR = BASE_DIR / "data/logs"

    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    AUTO_ADJUST = True

Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)