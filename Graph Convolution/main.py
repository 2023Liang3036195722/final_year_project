from data_download import StockDownloader
from log import logger
import time
from config import Config

if __name__ == "__main__":
    start_time = time.time()

    logger.info("=" * 50)
    logger.info("开始下载股票数据")
    logger.info(f"股票数量: {len(Config.STOCKS)}")
    logger.info(f"时间范围: {Config.START_DATE} 至 {Config.END_DATE}")
    logger.info("=" * 50)

    failed = StockDownloader.download_all(max_workers=5)

    duration = time.time() - start_time
    logger.info(f"全部完成，耗时: {duration:.2f}秒")
    if failed:
        logger.error(f"最终失败股票: {failed}")
    else:
        logger.info("所有股票下载成功！")