import yfinance as yf
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
import pandas as pd
import time
import logging

DB_HOST = "xxxx"
DB_NAME = "xxxx"
DB_USER = "xxxx"
DB_PASSWORD = "xxxx"

HSI_COMPONENTS = [
    "0001.HK", "0002.HK", "0003.HK", "0005.HK", "0006.HK",
    "0011.HK", "0012.HK", "0016.HK", "0017.HK", "0027.HK",
    "0066.HK", "0101.HK", "0175.HK", "0241.HK", "0285.HK",
    "0288.HK", "0291.HK", "0316.HK", "0322.HK", "0386.HK",
    "0388.HK", "0669.HK", "0688.HK", "0700.HK", "0762.HK",
    "0823.HK", "0836.HK", "0857.HK", "0868.HK", "0881.HK",
    "0883.HK", "0939.HK", "0941.HK", "0960.HK", "0968.HK",
    "0981.HK", "0992.HK", "1024.HK", "1038.HK", "1044.HK",
    "1088.HK", "1093.HK", "1099.HK", "1109.HK", "1113.HK",
    "1177.HK", "1209.HK", "1211.HK", "1299.HK", "1378.HK",
    "1398.HK", "1810.HK", "1876.HK", "1928.HK", "1929.HK",
    "1997.HK", "2015.HK", "2020.HK", "2057.HK", "2269.HK",
    "2313.HK", "2318.HK", "2319.HK", "2331.HK", "2359.HK",
    "2382.HK", "2388.HK", "2628.HK", "2688.HK", "2899.HK",
    "3690.HK", "3692.HK", "3968.HK", "3988.HK", "6618.HK",
    "6690.HK", "6862.HK", "9618.HK", "9633.HK", "9888.HK",
    "9901.HK", "9961.HK", "9988.HK", "9999.HK",
]

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365 * 1)  # 假设获取过去1年的数据

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logging.info("成功连接到数据库")
        return conn
    except Exception as e:
        logging.error(f"连接数据库失败: {e}")
        return None


def create_tables(conn):
    cur = None
    try:
        cur = conn.cursor()

        create_stock_table_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL,
            trade_date DATE NOT NULL,
            open_price DECIMAL(10, 4),
            high_price DECIMAL(10, 4),
            low_price DECIMAL(10, 4),
            close_price DECIMAL(10, 4),
            volume BIGINT,
            dividends DECIMAL(10, 4) DEFAULT 0,
            stock_splits DECIMAL(10, 4) DEFAULT 0,
            UNIQUE (stock_code, trade_date)
        );
        """
        cur.execute(create_stock_table_query)
        logging.info("表 'stock_data' 创建或已存在。")

        conn.commit()
        cur.close()
    except Exception as e:
        logging.error(f"创建表格失败: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.closr()

def insert_stock_data(conn, stock_code, data):
    if data.empty:
        logging.info(f"股票 {stock_code} 没有历史数据可插入。")
        return

    cur = conn.cursor()
    insert_query = sql.SQL("""
        INSERT INTO stock_data (
            stock_code, trade_date, open_price, high_price, low_price,
            close_price, volume, dividends, stock_splits
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (stock_code, trade_date) DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume,
            dividends = EXCLUDED.dividends,
            stock_splits = EXCLUDED.stock_splits;
    """)

    data.index = data.index.date

    rows_to_insert = []
    for index, row in data.iterrows():
        rows_to_insert.append((
            stock_code,
            index,
            None if pd.isna(row['Open']) else round(float(row['Open']), 4),
            None if pd.isna(row['High']) else round(float(row['High']), 4),
            None if pd.isna(row['Low']) else round(float(row['Low']), 4),
            None if pd.isna(row['Close']) else round(float(row['Close']), 4),
            None if pd.isna(row['Volume']) else int(row['Volume']),
            None if pd.isna(row['Dividends']) else round(float(row['Dividends']), 4),
            None if pd.isna(row['Stock Splits']) else round(float(row['Stock Splits']), 4)
        ))

    try:
        cur.executemany(insert_query, rows_to_insert)
        conn.commit()
        logging.info(f"成功为 {stock_code} 插入/更新 {len(rows_to_insert)} 条股票历史数据。")
    except Exception as e:
        logging.error(f"插入股票 {stock_code} 数据失败: {e}")
        conn.rollback()
    finally:
        cur.close()

def get_stock_datas():
    conn = connect_db()
    if not conn:
        return

    create_tables(conn)

    for stock_code in HSI_COMPONENTS:
        logging.info(f"--- 正在处理股票: {stock_code} ---")
        ticker = yf.Ticker(stock_code)

        try:
            stock_data = ticker.history(start=START_DATE, end=END_DATE)
            if not stock_data.empty:
                insert_stock_data(conn, stock_code, stock_data)
            else:
                logging.info(f"未找到股票 {stock_code} 的历史数据。")
        except Exception as e:
            logging.error(f"获取股票 {stock_code} 历史数据失败: {e}")

        time.sleep(3)

    if conn:
        conn.close()
        logging.info("所有数据获取并存储完毕，数据库连接已关闭。")


if __name__ == "__main__":
    get_stock_datas()