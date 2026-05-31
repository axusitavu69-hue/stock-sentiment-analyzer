"""
一次性批量下载所有A股K线数据，保存到本地Parquet文件。
运行一次即可，训练脚本直接读本地文件，不再调API。
"""
import os, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

OUTPUT_FILE = "kline_cache/all_stocks.parquet"
os.makedirs("kline_cache", exist_ok=True)


def download_all():
    """使用akshare批量下载+baostock兜底"""
    import akshare as ak
    import baostock as bs

    print("获取全A股列表...")
    try:
        stock_df = ak.stock_info_a_code_name()
        all_codes = stock_df['code'].astype(str).str.zfill(6).tolist()
    except:
        print("akshare获取列表失败，生成代码范围...")
        all_codes = []
        for pre in ['000', '001', '002', '003', '300', '301', '600', '601', '603', '605', '688']:
            for i in range(1, 999):
                all_codes.append(f'{pre}{i:03d}')
    print(f"共 {len(all_codes)} 只股票")

    end_dt = datetime.now().strftime('%Y-%m-%d')
    start_dt = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    all_data = []
    success = 0
    failed = 0

    # Phase 1: Try AKShare batch (faster)
    print("\n阶段1: AKShare批量下载...")
    try:
        # AKShare has a bulk API for all stocks
        for i, code in enumerate(all_codes):
            try:
                df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                        start_date=(datetime.now()-timedelta(days=400)).strftime('%Y%m%d'),
                                        end_date=datetime.now().strftime('%Y%m%d'),
                                        adjust='qfq')
                if df is not None and not df.empty and len(df) >= 40:
                    df['code'] = code
                    col_map = {'日期':'date','开盘':'open','最高':'high','最低':'low',
                               '收盘':'close','成交量':'volume','成交额':'amount',
                               '涨跌幅':'pct_chg','换手率':'turnover'}
                    df.rename(columns={k:v for k,v in col_map.items() if k in df.columns}, inplace=True)
                    keep_cols = ['date','code','open','high','low','close','volume','amount','pct_chg','turnover']
                    df = df[[c for c in keep_cols if c in df.columns]]
                    all_data.append(df)
                    success += 1
            except:
                pass
            if i % 100 == 0:
                print(f"  AKShare进度: {i}/{len(all_codes)}, 成功{success}")
    except Exception as e:
        print(f"  AKShare阶段结束: {e}")

    # Phase 2: Baostock for remaining
    remaining = [c for c in all_codes if c not in {d['code'].iloc[0] for d in all_data}]
    if remaining:
        print(f"\n阶段2: Baostock兜底 ({len(remaining)}只)...")
        bs.login()
        for i, code in enumerate(remaining[:2000]):
            try:
                bs_code = f'sh.{code}' if code.startswith('6') else f'sz.{code}'
                rs = bs.query_history_k_data_plus(
                    bs_code, 'date,open,high,low,close,volume,amount,turn,pctChg',
                    start_date=start_dt, end_date=end_dt, frequency='d', adjustflag='2')
                if rs.error_code == '0':
                    data = []
                    while rs.next():
                        data.append(rs.get_row_data())
                    if data and len(data) >= 40:
                        df = pd.DataFrame(data, columns=['date','open','high','low','close','volume','amount','turn','pctChg'])
                        for c in ['open','high','low','close','volume','amount','turn','pctChg']:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                        df['code'] = code
                        df = df.dropna(subset=['close'])
                        df.rename(columns={'turn':'turnover','pctChg':'pct_chg'}, inplace=True)
                        all_data.append(df)
                        success += 1
            except:
                failed += 1
            if i % 100 == 0:
                print(f"  Baostock进度: {i}/{len(remaining)}, 成功{success}, 失败{failed}")
        bs.logout()

    if all_data:
        merged = pd.concat(all_data, ignore_index=True)
        merged.to_parquet(OUTPUT_FILE, index=False)
        print(f"\n完成! {success}只股票, {len(merged)}条K线 -> {OUTPUT_FILE}")
    else:
        print("\n没有下载到任何数据!")


if __name__ == '__main__':
    download_all()
