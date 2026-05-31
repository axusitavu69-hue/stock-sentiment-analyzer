"""
量化模型完整训练系统
- 概念板块成分股 300 天价格数据训练
- 每日自动增量学习
- 因子自动发现与淘汰
- 在 PyCharm 中运行: python train_model.py

用法:
    python train_model.py                 # 首次完整训练
    python train_model.py --daily         # 每日自动增量学习
    python train_model.py --retrain       # 重新全量训练
"""
import sys, os, json, time, argparse
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================

TRACKED_CONCEPTS = [
    '商业航天', 'PCB概念', '存储芯片', 'CPO', '共封装光学', '创新药',
    '算力租赁', '钠离子电池', '光纤概念', 'AI应用', '人工智能', '先进封装',
    '人形机器人', '机器人概念', '机器人', '绿色电力', '数据中心', 'AIDC',
    '储能', '人工智能芯片', 'AI芯片', '半导体', '白酒概念', '白酒',
    '房地产', '新能源汽车', '新能源车', '消费概念', '大消费'
]

QUANT_TRACKER = "stock_reports/quant_tracker.json"
TRAINING_DAYS = 300
FEATURE_NAMES = ['5日涨幅', '20日涨幅', '量比', '趋势方向', '距均线偏离', '波动率',
                 'RSI信号', 'MACD信号', '均线乖离', '量价背离']

os.makedirs("stock_reports", exist_ok=True)


# ===================== 数据获取 =====================

def fetch_kline(code, days=TRAINING_DAYS):
    """获取个股K线 — Baostock + AKShare双源"""
    import baostock as bs
    bs_code = f'sh.{code}' if code.startswith('6') else f'sz.{code}'
    import threading, queue

    result_queue = queue.Queue()
    def _fetch():
        try:
            bs.login()
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=days + 20)).strftime('%Y-%m-%d')
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
                    result_queue.put(df.dropna(subset=['close']))
                    return
            bs.logout()
            result_queue.put(None)
        except:
            try: bs.logout()
            except: pass
            result_queue.put(None)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    try:
        result = result_queue.get(timeout=15)
        if result is not None and len(result) >= 60:
            return result
    except queue.Empty:
        pass

    # AKShare fallback
    try:
        import akshare as ak
        prefix = 'sh' + code if code.startswith('6') else 'sz' + code
        end2 = datetime.now().strftime('%Y%m%d')
        start2 = (datetime.now() - timedelta(days=days + 20)).strftime('%Y%m%d')
        df = ak.stock_zh_a_daily(symbol=prefix, start_date=start2, end_date=end2, adjust='qfq')
        if df is not None and not df.empty and len(df) >= 60:
            for c_map in [{'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','成交额':'amount','换手率':'turn'},
                          {'date':'date','open':'open','high':'high','low':'low','close':'close','volume':'volume','amount':'amount','turn':'turn'}]:
                rename = {k:v for k,v in c_map.items() if k in df.columns}
                if rename:
                    return df.rename(columns=rename)
    except:
        pass
    return pd.DataFrame()


def get_concept_stocks():
    """获取所有追踪概念的成分股 — 用涨停池历史数据做行业映射，不调被拦API"""
    import akshare as ak
    all_stocks = defaultdict(set)  # concept_name -> set(codes)

    # Collect stocks from historical limit-up pools using 所属行业 field
    end = datetime.now()
    print('  从历史涨停池中收集行业→个股映射...')
    sample_days = 0
    for i in range(365, 0, -2):  # every 2 days, up to 1 year back
        dt = end - timedelta(days=i)
        d_str = dt.strftime('%Y%m%d')
        try:
            df = ak.stock_zt_pool_em(date=d_str)
            if df is None or df.empty or '所属行业' not in df.columns:
                continue
            sample_days += 1
            for _, row in df.iterrows():
                sector = str(row.get('所属行业', '')).strip()
                code = str(row.get('代码', '')).strip()
                if sector and code.isdigit() and len(code) == 6:
                    # Check if this sector matches any tracked concept
                    for tracked in TRACKED_CONCEPTS:
                        ts = tracked.replace('概念', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '')
                        rs = sector.replace('概念', '')
                        if ts in rs or rs in ts or (len(ts) >= 3 and ts[:3] in rs):
                            all_stocks[tracked].add(code)
                            break
        except Exception as e:
            continue
        if sample_days % 10 == 0:
            print(f'    已采样 {sample_days} 个交易日...')

    print(f'  采样了 {sample_days} 个交易日')

    # Report results
    result = {}
    for tracked in TRACKED_CONCEPTS:
        codes = all_stocks.get(tracked, set())
        if codes:
            result[tracked] = list(codes)[:50]  # max 50 per concept
            print(f'  [OK] {tracked}: {len(codes)}只 (取前50)')
        else:
            print(f'  [MISS] {tracked}: 未在历史涨停中找到匹配行业')

    return result


# ===================== 特征工程 =====================

def extract_features(df):
    """从K线数据提取多维特征"""
    close = np.array(df['close'].values, dtype=float).flatten()
    high = np.array(df.get('high', close).values, dtype=float).flatten()
    low = np.array(df.get('low', close).values, dtype=float).flatten()
    vol_raw = df.get('volume', np.ones(len(close)))
    vol = np.array(vol_raw.values if hasattr(vol_raw, 'values') else vol_raw, dtype=float).flatten()
    if len(vol) != len(close): vol = np.ones(len(close))

    features, labels = [], []
    for i in range(80, len(close) - 1):
        win60 = close[i-60:i]
        win20 = close[i-20:i]
        win10 = close[i-10:i]
        if len(win20) < 20: continue

        # Core features
        ma5 = float(np.mean(win20[-5:]))
        ma20 = float(np.mean(win20))
        ma60 = float(np.mean(win60))
        ret_5d = float((close[i] - close[i-5]) / (close[i-5] + 1e-9) * 100)
        ret_20d = float((close[i] - close[i-20]) / (close[i-20] + 1e-9) * 100)
        avg_vol20 = float(np.mean(vol[max(0,i-20):i+1]))
        vol_ratio = float(vol[i] / (avg_vol20 + 1e-9))
        trend = 1 if ma5 > ma20 else -1
        vola = float(np.std(win20) / (np.mean(win20) + 1e-9) * 100)

        # RSI-like signal
        gains = np.sum(np.diff(win10)[np.diff(win10) > 0]) if len(np.diff(win10)[np.diff(win10) > 0]) > 0 else 0
        losses = -np.sum(np.diff(win10)[np.diff(win10) < 0]) if len(np.diff(win10)[np.diff(win10) < 0]) > 0 else 1e-9
        rsi_signal = 1 if gains / (gains + losses) > 0.5 else -1

        # MACD-like signal
        ema12 = pd.Series(close[i-26:i]).ewm(span=12, adjust=False).mean().values[-1]
        ema26 = pd.Series(close[i-26:i]).ewm(span=26, adjust=False).mean().values[-1]
        macd_signal = 1 if ema12 > ema26 else -1

        # Bollinger-like 乖离
        bb_mid = ma20
        bb_std = np.std(win20)
        bb_position = float((close[i] - bb_mid) / (bb_std + 1e-9))

        # Volume-price divergence
        price_up = close[i] > close[i-5]
        vol_up = vol[i] > np.mean(vol[max(0,i-5):i])
        divergence = -1 if price_up and not vol_up else (1 if not price_up and vol_up else 0)

        features.append([ret_5d, ret_20d, vol_ratio, trend, bb_position, vola,
                         rsi_signal, macd_signal, bb_position, divergence])
        next_ret = float((close[i+1] - close[i]) / (close[i] + 1e-9) * 100)
        labels.append(1 if next_ret > 0 else 0)

    return features, labels


# ===================== 模型训练 =====================

def train_model(features, labels, tag=""):
    """训练LightGBM并保存"""
    if len(features) < 200:
        print(f'  [{tag}] 数据不足: {len(features)}条')
        return None, 0, None

    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)
    split = int(len(X) * 0.8)

    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators=150, learning_rate=0.02, num_leaves=24,
                           max_depth=8, random_state=42, verbose=-1)
    model.fit(X[:split], y[:split])
    acc = float(model.score(X[split:], y[split:]))
    importance = model.feature_importances_

    print(f'  [{tag}] 准确率: {acc:.1%} | 样本: {len(X)}条 | 特征数: {X.shape[1]}')
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f'    {name:12s} {imp:.4f} {bar}')

    return model, acc, importance


def save_model(acc, importance, samples, stocks, train_rounds, factor_info):
    """保存模型到 JSON"""
    tracker = {}
    if os.path.exists(QUANT_TRACKER):
        with open(QUANT_TRACKER, 'r', encoding='utf-8') as f:
            tracker = json.load(f)

    if 'factor_pool' not in tracker:
        tracker['factor_pool'] = {
            'active': {name: name for name in FEATURE_NAMES},
            'candidates': {},
            'retired': {},
            'significance': {name: 0.5 for name in FEATURE_NAMES}
        }
    if 'factor_history' not in tracker:
        tracker['factor_history'] = []

    # Auto factor evolution
    sig_map = tracker['factor_pool']['significance']
    for i, name in enumerate(FEATURE_NAMES):
        imp = float(importance[i])
        old_sig = sig_map.get(name, 0.5)
        sig_map[name] = round(old_sig * 0.7 + min(1.0, imp * 3) * 0.3, 3)

    # Retire weak factors
    retired_today = []
    for name, sig in list(sig_map.items()):
        if sig < 0.15 and name not in ['5日涨幅', '20日涨幅', '量比', '趋势方向']:  # Keep core
            tracker['factor_pool']['retired'][name] = {
                'retired_date': datetime.now().strftime('%Y%m%d'), 'significance': sig}
            del sig_map[name]
            retired_today.append(name)
            print(f'  [淘汰因子] {name} (显著性{sig:.3f})')

    tracker['factor_history'].append({
        'date': datetime.now().strftime('%Y%m%d'),
        'active_count': len(sig_map),
        'retired': retired_today,
        'retired_count': len(tracker['factor_pool']['retired'])
    })
    if len(tracker['factor_history']) > 60:
        tracker['factor_history'] = tracker['factor_history'][-60:]

    tracker['ml_model'] = {
        'trained_date': datetime.now().strftime('%Y%m%d'),
        'accuracy': round(acc, 3),
        'samples': samples,
        'stocks_used': stocks,
        'train_rounds': train_rounds,
        'feature_count': len(sig_map),
        'feature_names': list(sig_map.keys()),
        'feature_importance': [round(float(v), 4) for v in importance],
        'model_type': 'LightGBM'
    }

    with open(QUANT_TRACKER, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n  模型已保存: {QUANT_TRACKER}')
    return tracker


# ===================== 主流程 =====================

def full_train():
    """完整训练：所有概念成分股 300 天数据"""
    print('=' * 60)
    print(f'  量化模型完整训练 ({TRAINING_DAYS}天数据)')
    print(f'  开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # Step 1: Get concept stocks
    print('\n[1/4] 获取概念板块成分股...')
    concept_stocks = get_concept_stocks()
    total_concepts = len(concept_stocks)
    all_codes = set()
    for codes in concept_stocks.values():
        all_codes.update(codes)
    all_codes = sorted(all_codes)
    print(f'  总计: {total_concepts}个概念, {len(all_codes)}只不重复个股')

    # Step 2: Fetch K-line data
    print(f'\n[2/4] 获取{TRAINING_DAYS}天K线数据...')
    all_features, all_labels = [], []
    trained = 0
    failed = 0

    for i, code in enumerate(all_codes):
        if i % 20 == 0:
            print(f'  进度: {i}/{len(all_codes)} ({trained}只成功, {failed}只失败, {len(all_features)}条特征)')
        df = fetch_kline(code, TRAINING_DAYS)
        if df.empty or len(df) < 80:
            failed += 1; continue
        feats, lbls = extract_features(df)
        if feats:
            all_features.extend(feats)
            all_labels.extend(lbls)
            trained += 1
        else:
            failed += 1

    print(f'  完成: {trained}只成功, {failed}只失败, 总计{len(all_features)}条特征')

    if len(all_features) < 500:
        print(f'\n[ERROR] 训练数据严重不足 ({len(all_features)}条)，请检查网络或等交易日重试')
        return

    # Step 3: Train
    print(f'\n[3/4] 训练LightGBM模型...')
    model, acc, importance = train_model(all_features, all_labels, '完整训练')

    # Step 4: Save
    print(f'\n[4/4] 保存模型...')
    tracker = json.load(open(QUANT_TRACKER, 'r')) if os.path.exists(QUANT_TRACKER) else {}
    rounds = tracker.get('ml_model', {}).get('train_rounds', 0) + 1
    save_model(acc, importance, len(all_features), trained, rounds, {})
    print(f'\n{"=" * 60}')
    print(f'  训练完成! 准确率: {acc:.1%}, {len(all_features)}条样本')
    print(f'  模型文件: {QUANT_TRACKER}')
    print(f'{"=" * 60}')


def daily_learn():
    """每日增量学习"""
    print('=' * 60)
    print(f'  每日自动学习 ({datetime.now().strftime("%Y-%m-%d")})')
    print('=' * 60)

    today = datetime.now().strftime('%Y%m%d')

    # Get today's limit-up pool
    print('\n[1/3] 获取今日涨停股...')
    import akshare as ak
    limit_df = ak.stock_zt_pool_em(date=today)
    if limit_df.empty:
        print('  今日无涨停数据，跳过学习')
        return

    codes = limit_df['代码'].astype(str).str.strip().tolist()
    print(f'  涨停: {len(codes)}只')

    # Get 100 days of data for today's stocks
    print(f'\n[2/3] 获取100天K线...')
    all_features, all_labels = [], []
    trained = 0
    for code in codes[:60]:
        df = fetch_kline(code, 100)
        if not df.empty and len(df) >= 60:
            feats, lbls = extract_features(df)
            if feats: all_features.extend(feats); all_labels.extend(lbls); trained += 1

    print(f'  获取: {trained}只有效, {len(all_features)}条特征')

    if len(all_features) < 100:
        print('  数据不足，跳过今日学习')
        return

    # Train and merge
    print(f'\n[3/3] 增量训练...')
    model, acc, importance = train_model(all_features, all_labels, '每日学习')

    tracker = json.load(open(QUANT_TRACKER, 'r')) if os.path.exists(QUANT_TRACKER) else {}
    prev_samples = tracker.get('ml_model', {}).get('samples', 0)
    rounds = tracker.get('ml_model', {}).get('train_rounds', 0) + 1
    save_model(acc, importance, prev_samples + len(all_features), trained, rounds, {})
    print(f'\n  今日学习完成! 累计样本: {prev_samples + len(all_features)}条')


def auto_evolve_factors():
    """自动因子进化：检查是否需要增减因子"""
    tracker = json.load(open(QUANT_TRACKER, 'r')) if os.path.exists(QUANT_TRACKER) else {}
    fp = tracker.get('factor_pool', {})
    sig = fp.get('significance', {})

    print('\n因子池状态:')
    print(f'  活跃: {len(fp.get("active", {}))} | 候选: {len(fp.get("candidates", {}))} | 淘汰: {len(fp.get("retired", {}))}')

    # Check significance
    for name, s in sorted(sig.items(), key=lambda x: -x[1]):
        status = '✅' if s >= 0.4 else ('⚠️' if s >= 0.2 else '❌')
        print(f'  {status} {name}: {s:.3f}')
    print()


def main():
    parser = argparse.ArgumentParser(description='量化模型训练系统')
    parser.add_argument('--daily', action='store_true', help='每日增量学习')
    parser.add_argument('--retrain', action='store_true', help='重新全量训练')
    parser.add_argument('--evolve', action='store_true', help='仅因子进化检查')
    parser.add_argument('--status', action='store_true', help='查看模型状态')
    args = parser.parse_args()

    if args.status:
        tracker = json.load(open(QUANT_TRACKER, 'r')) if os.path.exists(QUANT_TRACKER) else {}
        ml = tracker.get('ml_model', {})
        print(f'模型状态:')
        print(f'  最后训练: {ml.get("trained_date", "从未")}')
        print(f'  准确率: {ml.get("accuracy", 0):.1%}' if ml.get('accuracy') else '  准确率: N/A')
        print(f'  样本量: {ml.get("samples", 0)}条')
        print(f'  训练轮次: {ml.get("train_rounds", 0)}')
        auto_evolve_factors()
        return

    if args.evolve:
        auto_evolve_factors()
        return

    if args.daily:
        daily_learn()
        auto_evolve_factors()
        return

    if args.retrain:
        # Reset tracker
        if os.path.exists(QUANT_TRACKER):
            os.rename(QUANT_TRACKER, QUANT_TRACKER + '.bak')
            print(f'旧模型备份: {QUANT_TRACKER}.bak')
        full_train()
        auto_evolve_factors()
        return

    # Default: full train
    full_train()
    auto_evolve_factors()


if __name__ == '__main__':
    main()
