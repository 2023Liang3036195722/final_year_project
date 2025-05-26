import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

def batch_check_parquets(directory):
    files = list(Path(directory).glob("*.parquet"))
    results = []

    for file in tqdm(files, desc="检查文件中"):
        try:
            pf = pq.ParquetFile(file)
            info = {
                "文件名": file.name,
                "行数": pf.metadata.num_rows,
                "列数": len(pf.metadata.schema.names),
                "大小(MB)": round(file.stat().st_size / (1024 * 1024), 2),
                "状态": "✓ 正常"
            }
            df_sample = pd.read_parquet(file, columns=['Open','High', 'Low', 'Close', 'Volume'])
            if 'Open' not in df_sample.columns:
                info["状态"] = "⚠ 缺少Open列"
            if 'High' not in df_sample.columns:
                info["状态"] = "⚠ 缺少High列"
            if 'Low' not in df_sample.columns:
                info["状态"] = "⚠ 缺少Low列"
            if 'Close' not in df_sample.columns:
                info["状态"] = "⚠ 缺少Close列"
            if 'Volume' not in df_sample.columns:
                info["状态"] = "⚠ 缺少Volume列"

        except Exception as e:
            info = {
                "文件名": file.name,
                "行数": None,
                "列数": None,
                "大小(MB)": round(file.stat().st_size / (1024 * 1024), 2),
                "状态": f"× 损坏: {str(e)}"
            }

        results.append(info)

    return pd.DataFrame(results)

df_report = batch_check_parquets("data/stocks")
print(df_report)

df_report.to_csv("parquet_quality_report.csv", index=False, encoding='utf-8-sig')