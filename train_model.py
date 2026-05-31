"""
量化模型离线训练脚本
可在 PyCharm 中直接运行，不需要启动 Streamlit。
训练结果保存到 stock_reports/quant_tracker.json，网页端自动读取。

用法：
    python train_model.py              # 初始训练
    python train_model.py --incremental # 增量训练
    python train_model.py --full        # 先初始再增量
"""
import sys, os, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the analyzer class from the dashboard
# We need to extract just the StockAnalyzer core without Streamlit dependency
import akshare as ak
import baostock as bs
from lightgbm import LGBMClassifier

TRACKED_CONCEPTS = [
    '商业航天', 'PCB', '存储芯片', 'CPO', '共封装光学', '创新药',
    '算力租赁', '钠离子电池', '光纤', 'AI应用', '先进封装',
    '人形机器人', '机器人', '绿色电力', '数据中心', 'AIDC',
    '储能', 'AI芯片', '人工智能芯片', '半导体', '白酒',
    '房地产', '新能源汽车', '新能源车', '消费概念', '大消费'
]

QUANT_TRACKER_PATH = "stock_reports/quant_tracker.json"


def fetch_stock_history(code, days=180):
    """获取个股历史K线"""
    bs_code = f'sh.{code}' if code.startswith('6') else f'sz.{code}'

    # Try Baostock first
    try:
        bs.login()
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - pd.Timedelta(days=days + 10)).strftime('%Y-%m-%d')
        rs = bs.query_history_k_data_plus(
            bs_code, 'date,open,high,low,close,volume,amount,turn,pctChg',
            start_date=start, end_date=end, frequency='d', adjustflag='2')
        if rs.error_code == '0':
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            bs.logout()
            if data:
                df = pd.DataFrame(data, columns=['date','open','high','low','close','volume','amount','turn','pctChg'])
                for c in ['open','high','low','close','volume','amount','turn','pctChg']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna(subset=['close'])
                return df if len(df) >= 10 else pd.DataFrame()
        bs.logout()
    except Exception as e:
        print(f'  [WARN] baostock {code}: {e}')
        try: bs.logout()
        except: pass

    # Fallback: AKShare
    try:
        prefix = 'sh' + code if code.startswith('6') else 'sz' + code
        end2 = datetime.now().strftime('%Y%m%d')
        start2 = (datetime.now() - pd.Timedelta(days=days + 5)).strftime('%Y%m%d')
        df = ak.stock_zh_a_daily(symbol=prefix, start_date=start2, end_date=end2, adjust='qfq')
        if df is not None and not df.empty:
            col_map = {'date':'date','open':'open','high':'high','low':'low','close':'close',
                       'volume':'volume','日期':'date','开盘':'open','最高':'high','最低':'low',
                       '收盘':'close','成交量':'volume','amount':'amount','成交额':'amount',
                       'turnover':'turn','换手率':'turn'}
            df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
            return df
    except Exception as e2:
        print(f'  [WARN] akshare {code}: {e2}')
    return pd.DataFrame()


def extract_features(close, vol):
    """从价格和成交量提取训练特征"""
    features, labels = [], []
    for i in range(60, len(close) - 1):
        window = close[i-60:i]
        if len(window) < 20: continue
        ma5 = float(np.mean(window[-5:]))
        ma20 = float(np.mean(window[-20:]))
        ret_5d = float((close[i] - close[i-5]) / (close[i-5] + 1e-9) * 100)
        ret_20d = float((close[i] - close[i-20]) / (close[i-20] + 1e-9) * 100)
        vol_ratio = float(vol[i] / (np.mean(vol[max(0,i-20):i+1]) + 1e-9))
        trend = 1 if ma5 > ma20 else -1
        vola = float(np.std(window[-20:]) / (np.mean(window[-20:]) + 1e-9) * 100)
        features.append([ret_5d, ret_20d, vol_ratio, trend,
                         float((close[i] - ma20) / (ma20 + 1e-9) * 100), vola])
        next_ret = float((close[i+1] - close[i]) / (close[i] + 1e-9) * 100)
        labels.append(1 if next_ret > 0 else 0)
    return features, labels


def train_lightgbm(features, labels, tag=""):
    """训练LightGBM模型"""
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)
    split = int(len(X) * 0.8)

    model = LGBMClassifier(n_estimators=120, learning_rate=0.03, num_leaves=20,
                           max_depth=6, random_state=42, verbose=-1)
    model.fit(X[:split], y[:split])
    acc = float(model.score(X[split:], y[split:]))
    importance = model.feature_importances_

    print(f'\n  [{tag}] 训练完成:')
    print(f'    样本: {len(X)}条 (训练{len(X[:split])}/测试{len(X[split:])})')
    print(f'    准确率: {acc:.1%}')
    print(f'    特征重要性:')
    for name, imp in zip(['5日涨幅','20日涨幅','量比','趋势方向','距均线','波动率'],
                         [round(float(v), 4) for v in importance]):
        print(f'      {name}: {imp:.4f}')

    # Save to tracker
    os.makedirs('stock_reports', exist_ok=True)
    tracker = {}
    if os.path.exists(QUANT_TRACKER_PATH):
        with open(QUANT_TRACKER_PATH, 'r', encoding='utf-8') as f:
            tracker = json.load(f)

    prev_rounds = tracker.get('ml_model', {}).get('train_rounds', 0)
    prev_samples = tracker.get('ml_model', {}).get('samples', 0)
    total_samples = prev_samples + len(features) if prev_rounds > 0 else len(features)

    tracker['ml_model'] = {
        'trained_date': datetime.now().strftime('%Y%m%d'),
        'accuracy': round(acc, 3),
        'samples': total_samples,
        'train_rounds': prev_rounds + 1,
        'feature_names': ['5日涨幅', '20日涨幅', '量比', '趋势方向', '距均线', '波动率'],
        'feature_importance': [round(float(v), 4) for v in importance],
        'model_type': 'LightGBM'
    }
    with open(QUANT_TRACKER_PATH, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)

    return acc, importance


def initial_train():
    """初始训练：用今日涨停股180天数据"""
    print('=' * 50)
    print('初始训练：涨停股 180 天数据')
    print('=' * 50)

    today = datetime.now().strftime('%Y%m%d')
    print(f'日期: {today}')

    # Get limit-up pool
    print('获取涨停池...')
    limit_df = ak.stock_zt_pool_em(date=today)
    if limit_df.empty:
        print('今日无涨停数据，终止训练')
        return
    codes = limit_df['代码'].astype(str).str.strip().tolist()
    print(f'涨停股: {len(codes)} 只')

    all_features, all_labels = [], []
    trained_stocks = 0

    for i, code in enumerate(codes[:80]):
        print(f'  [{i+1}/{min(80, len(codes))}] {code}...', end=' ')
        df = fetch_stock_history(code, 180)
        if df.empty or len(df) < 65:
            print('SKIP (数据不足)')
            continue
        close = np.array(df['close'].values, dtype=float).flatten()
        vol = np.array(df.get('volume', np.ones(len(close))).values, dtype=float).flatten()
        if len(vol) != len(close): vol = np.ones(len(close))
        feats, lbls = extract_features(close, vol)
        if feats:
            all_features.extend(feats)
            all_labels.extend(lbls)
            trained_stocks += 1
            print(f'OK ({len(feats)}条)')
        else:
            print('SKIP (特征不足)')

    if len(all_features) < 100:
        print(f'\n训练数据不足! 仅 {len(all_features)} 条')
        return
    print(f'\n总特征: {len(all_features)}条, {trained_stocks}只股票')
    train_lightgbm(all_features, all_labels, tag='初始训练')
    print('\nModel saved to stock_reports/quant_tracker.json')


def incremental_train():
    """增量训练：用概念板块成分股100天数据"""
    print('=' * 50)
    print('增量训练：概念板块成分股 100 天数据')
    print('=' * 50)

    today = datetime.now().strftime('%Y%m%d')

    # Get today's limit-up codes (to exclude)
    limit_df = ak.stock_zt_pool_em(date=today)
    today_codes = set(limit_df['代码'].astype(str).str.strip().tolist()) if not limit_df.empty else set()

    # Get concept constituent stocks
    all_concept_codes = set()
    print('获取概念成分股...')

    # First get all concept names
    try:
        concept_names_df = ak.stock_board_concept_name_em()
        all_names = set()
        for c in concept_names_df.columns:
            if '板块' in str(c) or '名称' in str(c) or '概念' in str(c):
                all_names = set(concept_names_df[c].astype(str).tolist())
                break
    except:
        # Fallback: use fund flow concepts
        ff = ak.stock_fund_flow_concept(symbol='即时')
        all_names = set(ff['行业'].tolist()) if '行业' in ff.columns else set()

    print(f'  总概念板块: {len(all_names)} 个')

    matched = 0
    for tracked in TRACKED_CONCEPTS[:15]:
        # Fuzzy match
        found_names = []
        for real_name in all_names:
            if tracked in str(real_name) or str(real_name) in tracked:
                found_names.append(real_name)
        if not found_names:
            ts = tracked.replace('概念', '').replace('(', '').replace(')', '')
            for real_name in all_names:
                if len(ts) >= 2 and ts in str(real_name).replace('概念', ''):
                    found_names.append(real_name)
        if not found_names:
            continue
        matched += 1

        for name in found_names[:2]:
            try:
                cons = ak.stock_board_concept_cons_em(symbol=name)
                if cons is not None and not cons.empty:
                    code_col = '代码' if '代码' in cons.columns else cons.columns[0]
                    for c in cons[code_col].astype(str).str.zfill(6).str.strip().tolist()[:15]:
                        if c not in today_codes and c.isdigit() and len(c) == 6:
                            all_concept_codes.add(c)
            except Exception as e:
                print(f'  [WARN] concept {name}: {e}')

    print(f'  匹配概念: {matched} 个')
    print(f'  成分股: {len(all_concept_codes)} 只')

    if len(all_concept_codes) < 10:
        print('成分股不足10只，终止训练')
        return

    all_features, all_labels = [], []
    trained_stocks = 0

    codes_list = list(all_concept_codes)[:60]
    for i, code in enumerate(codes_list):
        print(f'  [{i+1}/{len(codes_list)}] {code}...', end=' ')
        df = fetch_stock_history(code, 100)
        if df.empty or len(df) < 65:
            print('SKIP')
            continue
        close = np.array(df['close'].values, dtype=float).flatten()
        vol = np.array(df.get('volume', np.ones(len(close))).values, dtype=float).flatten()
        if len(vol) != len(close): vol = np.ones(len(close))
        feats, lbls = extract_features(close, vol)
        if feats:
            all_features.extend(feats)
            all_labels.extend(lbls)
            trained_stocks += 1
            print(f'OK ({len(feats)}条)')
        else:
            print('SKIP')

    if len(all_features) < 100:
        print(f'\n训练数据不足! 仅 {len(all_features)} 条')
        return
    print(f'\n总特征: {len(all_features)}条, {trained_stocks}只股票')
    train_lightgbm(all_features, all_labels, tag='增量训练')
    print('\nModel saved to stock_reports/quant_tracker.json')


def main():
    parser = argparse.ArgumentParser(description='量化模型离线训练')
    parser.add_argument('--incremental', action='store_true', help='增量训练')
    parser.add_argument('--full', action='store_true', help='先初始训练再增量')
    args = parser.parse_args()

    if args.full:
        initial_train()
        print('\n' + '=' * 50 + '\n')
        incremental_train()
    elif args.incremental:
        incremental_train()
    else:
        initial_train()


if __name__ == '__main__':
    main()
