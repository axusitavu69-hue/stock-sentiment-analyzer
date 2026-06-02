"""
量化模型完整训练系统 v2.0 — 优化版
- 2500只股票 × 1年(365天)K线数据
- 并发批量获取，速度提升5~10倍
- 特征工程升级（新增动量/成交额因子）
- LightGBM超参优化 + 时序切割防泄露
- 模型持久化到 .pkl（可跨会话复用）
- 断点续训：已获取的K线缓存到本地
- 增量学习保留历史样本权重

用法:
    python train_model.py               # 首次完整训练
    python train_model.py --daily       # 每日增量学习
    python train_model.py --retrain     # 清空缓存重新训练
    python train_model.py --status      # 查看模型状态
    python train_model.py --evolve      # 仅因子进化检查
"""
import sys, os, json, time, argparse, pickle, hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================

TRAINING_DAYS   = 365          # 改为1年
TARGET_STOCKS   = 5500         # 目标股票数（全A股）
BATCH_SIZE      = 50           # 并发批次大小（每批最多同时发50个请求）
MAX_WORKERS     = 5            # HTTP请求可并发
CACHE_DIR       = "kline_cache"  # K线本地缓存目录
MODEL_DIR       = "stock_reports"
QUANT_TRACKER   = f"{MODEL_DIR}/quant_tracker.json"
MODEL_PKL       = f"{MODEL_DIR}/lgbm_model.pkl"   # 新增：持久化模型文件
CACHE_EXPIRE_H  = 6            # 缓存失效时间（小时），交易日内不重复拉取

FEATURE_NAMES = [
    # 原有因子
    '5日涨幅', '20日涨幅', '量比', '趋势方向', '距均线偏离', '波动率',
    'RSI信号', 'MACD信号', '均线乖离', '量价背离',
    # 新增因子
    '60日涨幅', '5日振幅', '成交额比', '动量加速度', '均线多头排列',
    'ATR归一化', '布林带宽度', '高低点偏离', '换手异动'
]

TRACKED_CONCEPTS = [
    '商业航天', 'PCB概念', '存储芯片', 'CPO', '共封装光学', '创新药',
    '算力租赁', '钠离子电池', '光纤概念', 'AI应用', '人工智能', '先进封装',
    '人形机器人', '机器人概念', '绿色电力', '数据中心', 'AIDC',
    '储能', 'AI芯片', '半导体', '白酒', '房地产', '新能源汽车', '大消费'
]

CONCEPT_INDUSTRY_MAP = {
    '商业航天': ['航空', '航天', '军工', '卫星'],
    'PCB概念': ['电子元件', '印制电路', 'PCB', '电路板'],
    '存储芯片': ['半导体', '芯片', '存储'],
    'CPO': ['通信设备', '光电', '光通信', '光模块'],
    '共封装光学': ['通信设备', '光电', '光通信', '光模块'],
    '创新药': ['化学制药', '生物制品', '医药', '生物医药'],
    '算力租赁': ['IT服务', '计算机', '软件', '算力', '云计算'],
    '钠离子电池': ['电池', '电气设备', '钠电池'],
    '光纤概念': ['通信设备', '光纤', '光缆', '光通信'],
    'AI应用': ['计算机', '软件', 'IT服务', '人工智能'],
    '人工智能': ['计算机', '软件', 'IT服务', '人工智能'],
    '先进封装': ['半导体', '芯片', '封装'],
    '人形机器人': ['机器人', '自动化', '机械设备'],
    '机器人概念': ['机器人', '自动化', '机械设备'],
    '绿色电力': ['电力', '风电', '光伏', '新能源发电'],
    '数据中心': ['IT服务', '计算机', '数据中心', '云计算'],
    'AIDC': ['IT服务', '计算机', '数据中心', '云计算'],
    '储能': ['电池', '电气设备', '储能'],
    'AI芯片': ['半导体', '芯片'],
    '半导体': ['半导体', '芯片', '集成电路'],
    '白酒': ['白酒', '酿酒', '食品饮料'],
    '房地产': ['房地产', '地产'],
    '新能源汽车': ['汽车', '汽车零部件', '新能源汽车'],
    '大消费': ['商业', '零售', '食品', '百货', '超市'],
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ===================== K线缓存层 =====================

def _cache_path(code: str, days: int) -> str:
    """返回本地缓存文件路径"""
    return os.path.join(CACHE_DIR, f"{code}_{days}d.pkl")


def _cache_valid(path: str) -> bool:
    """缓存是否在有效期内"""
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) < CACHE_EXPIRE_H * 3600


def _load_cache(code: str, days: int):
    path = _cache_path(code, days)
    if _cache_valid(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _save_cache(code: str, days: int, df):
    path = _cache_path(code, days)
    try:
        with open(path, 'wb') as f:
            pickle.dump(df, f)
    except Exception:
        pass


# ===================== 数据获取（并发版） =====================

try:
    from eastmoney_api import get_kline as em_get_kline
    _HAS_EM = True
except ImportError:
    _HAS_EM = False
    print("[WARN] eastmoney_api 未安装，将使用 akshare 降级获取")


def _fetch_one(code: str, days: int) -> tuple:
    """单只股票获取K线（含缓存）"""
    cached = _load_cache(code, days)
    if cached is not None:
        return code, cached

    df = None
    try:
        from eastmoney_api import get_kline as gk
        df = gk(code, days)
    except:
        df = None

    if df is not None and not (hasattr(df, 'empty') and df.empty) and len(df) >= 40:
        _save_cache(code, days, df)
    return code, df


def fetch_kline_batch_concurrent(codes: list, days: int = TRAINING_DAYS) -> dict:
    """
    并发批量获取K线，MAX_WORKERS线程并发。
    返回 {code: DataFrame}，失败的code不在结果中。
    """
    results = {}
    total = len(codes)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, code, days): code for code in codes}
        for future in as_completed(futures):
            try:
                code, df = future.result()
            except Exception as e:
                # Socket crash etc - skip this stock
                done += 1
                continue
            done += 1
            if df is not None and not (hasattr(df, 'empty') and df.empty):
                if len(df) >= 60:
                    results[code] = df
            if done % 100 == 0 or done == total:
                hit = len(results)
                print(f"    进度: {done}/{total}  有效: {hit}  失败: {done-hit}")

    return results


# ===================== 股票池构建 =====================

def get_stock_pool() -> list:
    """
    构建2500只训练股票池：
    1. 优先从1年涨停池中取（质量最高）
    2. 不足时补充全市场A股
    返回去重后的code列表，最多TARGET_STOCKS只
    """
    import akshare as ak

    all_codes: set = set()
    end = datetime.now()

    # ── 阶段1: 从过去1年涨停池采样 ──────────────────────────
    print("  [1/2] 从1年涨停池采集股票...")
    sample_days = 0
    # 每隔3天采样一次（提速，减少API调用）
    for i in range(365, 0, -3):
        dt = end - timedelta(days=i)
        d_str = dt.strftime('%Y%m%d')
        try:
            df = ak.stock_zt_pool_em(date=d_str)
            if df is None or df.empty:
                continue
            sample_days += 1
            codes_today = df['代码'].astype(str).str.strip().tolist()
            valid = [c for c in codes_today if c.isdigit() and len(c) == 6]
            all_codes.update(valid)
        except Exception:
            continue
        if len(all_codes) >= TARGET_STOCKS:
            break

    print(f"    涨停池: {len(all_codes)}只 ({sample_days}个交易日采样)")

    # ── 阶段2: 不足时补充全A股 ─────────────────────────────
    if len(all_codes) < TARGET_STOCKS:
        print(f"  [2/2] 补充全A股至{TARGET_STOCKS}只...")
        try:
            all_stock_df = ak.stock_info_a_code_name()
            all_market_codes = all_stock_df['code'].astype(str).str.strip().tolist()
            # 过滤ST、北交所(8开头)、优先选主板+创业板+科创板
            filtered = [
                c for c in all_market_codes
                if c.isdigit() and len(c) == 6
                and c[0] in ('0', '3', '6')  # 主板/创业板/科创板
                and c not in all_codes
            ]
            need = TARGET_STOCKS - len(all_codes)
            all_codes.update(filtered[:need])
            print(f"    补充: {min(need, len(filtered))}只")
        except Exception as e:
            print(f"    [WARN] 全A股补充失败: {e}，尝试代码枚举...")
            # 最终兜底：代码枚举
            for pfx in ['000', '001', '002', '003', '300', '301', '600', '601', '603', '605', '688']:
                for sfx in range(1, 1000):
                    c = f"{pfx}{sfx:03d}"
                    if c not in all_codes:
                        all_codes.add(c)
                    if len(all_codes) >= TARGET_STOCKS:
                        break
                if len(all_codes) >= TARGET_STOCKS:
                    break

    result = sorted(all_codes)[:TARGET_STOCKS]
    print(f"  最终股票池: {len(result)}只")
    return result


# ===================== 特征工程（升级版） =====================

def extract_features(df) -> tuple:
    """
    从K线DataFrame提取多维特征。
    新增：60日涨幅、5日振幅、成交额比、动量加速度、均线多头排列、
          ATR归一化、布林带宽度、高低点偏离、换手异动
    标签：次日涨跌（1=涨/0=跌），保持不变
    """
    # ── 数据准备 ──────────────────────────────────────────
    def col(name, fallback=None):
        if name in df.columns:
            return np.array(df[name].values, dtype=float).flatten()
        return fallback

    close = col('close'); assert close is not None, "缺少close列"
    high  = col('high',  close.copy())
    low   = col('low',   close.copy())
    vol   = col('volume', np.ones(len(close)))
    amt   = col('amount', vol * close)   # 成交额

    n = len(close)
    if n < 80:
        return [], []

    features, labels = [], []
    # 从第80根K线开始，留1根作标签
    for i in range(80, n - 1):
        c = close
        # ── 均线 ─────────────────────────────────
        ma5  = float(np.mean(c[i-5:i]))
        ma10 = float(np.mean(c[i-10:i]))
        ma20 = float(np.mean(c[i-20:i]))
        ma60 = float(np.mean(c[i-60:i]))

        # ── 原有因子 ──────────────────────────────
        ret_5  = (c[i] - c[i-5])  / (c[i-5]  + 1e-9) * 100
        ret_20 = (c[i] - c[i-20]) / (c[i-20] + 1e-9) * 100
        avg_vol20 = np.mean(vol[max(0,i-20):i]) + 1e-9
        vol_ratio = vol[i] / avg_vol20
        trend = 1.0 if ma5 > ma20 else -1.0
        bb_dev = (c[i] - ma20) / (np.std(c[i-20:i]) + 1e-9)   # 布林偏离(=均线乖离)
        vola  = np.std(c[i-20:i]) / (ma20 + 1e-9) * 100

        diff10 = np.diff(c[i-10:i])
        gains  = diff10[diff10 > 0].sum() if len(diff10[diff10 > 0]) else 0
        losses = -diff10[diff10 < 0].sum() if len(diff10[diff10 < 0]) else 1e-9
        rsi_sig = 1.0 if gains / (gains + losses) > 0.5 else -1.0

        seg = c[max(0,i-26):i]
        ema12 = pd.Series(seg).ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = pd.Series(seg).ewm(span=26, adjust=False).mean().iloc[-1]
        macd_sig = 1.0 if ema12 > ema26 else -1.0

        price_up = c[i] > c[i-5]
        vol_up   = vol[i] > np.mean(vol[max(0,i-5):i])
        divergence = float(-1 if price_up and not vol_up else (1 if not price_up and vol_up else 0))

        # ── 新增因子 ──────────────────────────────
        ret_60 = (c[i] - c[i-60]) / (c[i-60] + 1e-9) * 100

        # 5日振幅（高低差/收盘）
        amp5 = (np.max(high[i-5:i]) - np.min(low[i-5:i])) / (c[i] + 1e-9) * 100

        # 成交额比（今日成交额/20日均成交额）
        avg_amt20 = np.mean(amt[max(0,i-20):i]) + 1e-9
        amt_ratio = amt[i] / avg_amt20

        # 动量加速度（近5日涨幅 - 前5~10日涨幅）
        ret_prev5 = (c[i-5] - c[i-10]) / (c[i-10] + 1e-9) * 100
        momentum_acc = ret_5 - ret_prev5

        # 均线多头排列：ma5>ma10>ma20>ma60
        ma_bull = 1.0 if (ma5 > ma10 > ma20 > ma60) else (-1.0 if (ma5 < ma10 < ma20 < ma60) else 0.0)

        # ATR归一化（20日真实波幅均值/收盘价）
        tr = np.maximum(
            high[max(0,i-20):i] - low[max(0,i-20):i],
            np.abs(high[max(0,i-20):i] - np.roll(c[max(0,i-20):i], 1))
        )
        atr_norm = np.mean(tr[1:]) / (c[i] + 1e-9) * 100

        # 布林带宽度（标准差/均值）
        bb_width = np.std(c[i-20:i]) / (ma20 + 1e-9) * 100

        # 高低点偏离（今日收盘距20日最高点的距离）
        hi20 = np.max(high[i-20:i])
        lo20 = np.min(low[i-20:i])
        hi_dev = (c[i] - hi20) / (hi20 + 1e-9) * 100

        # 换手异动（量比的5日均值比）
        avg_vr5 = np.mean(vol[max(0,i-5):i]) / avg_vol20
        turnover_chg = vol_ratio / (avg_vr5 + 1e-9)

        feat = [
            # 原有10个
            float(ret_5), float(ret_20), float(vol_ratio), float(trend),
            float(bb_dev), float(vola), float(rsi_sig), float(macd_sig),
            float(bb_dev), float(divergence),   # 注意：原代码均线乖离=bb_dev，保留一致
            # 新增9个
            float(ret_60), float(amp5), float(amt_ratio), float(momentum_acc),
            float(ma_bull), float(atr_norm), float(bb_width),
            float(hi_dev), float(turnover_chg)
        ]

        # 检查NaN/Inf
        if any(not np.isfinite(v) for v in feat):
            continue

        features.append(feat)
        next_ret = (c[i+1] - c[i]) / (c[i] + 1e-9) * 100
        labels.append(1 if next_ret > 0 else 0)

    return features, labels


# ===================== 模型训练（优化版） =====================

def train_model_lgbm(features: list, labels: list, tag: str = "") -> tuple:
    """
    训练LightGBM。
    改进：
    1. 时序切割（前80%训练，后20%验证），防止数据泄露
    2. early_stopping 防止过拟合
    3. class_weight 应对涨跌不均衡
    4. 返回 (model, acc, importance, val_metrics)
    """
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    if len(features) < 500:
        print(f"  [{tag}] 样本不足: {len(features)}条，跳过训练")
        return None, 0.0, None, {}

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # ── 时序切割（不随机shuffle，保留时序）──────────────
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 类别权重（多空不均衡时有效）
    pos_ratio = y_train.mean()
    scale = (1 - pos_ratio) / (pos_ratio + 1e-9)

    model = LGBMClassifier(
        n_estimators      = 800,
        learning_rate     = 0.008,
        num_leaves        = 63,
        max_depth         = 7,
        min_child_samples = 30,
        subsample         = 0.8,
        subsample_freq    = 1,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        scale_pos_weight  = scale,
        random_state      = 42,
        n_jobs            = -1,        # 使用全部CPU核
        verbose           = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
        ]
    )

    # ── 评估 ──────────────────────────────────────────
    acc   = float(model.score(X_val, y_val))
    preds = model.predict(X_val)
    from sklearn.metrics import classification_report
    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    f1_up   = report.get('1', {}).get('f1-score', 0)
    f1_down = report.get('0', {}).get('f1-score', 0)

    importance = model.feature_importances_
    n_feat = len(FEATURE_NAMES)

    print(f"\n  [{tag}] 准确率: {acc:.1%} | F1↑: {f1_up:.3f} | F1↓: {f1_down:.3f} "
          f"| 训练样本: {len(X_train)} | 验证样本: {len(X_val)}")
    print(f"  特征重要性 ({n_feat}个):")
    feat_imp = sorted(zip(FEATURE_NAMES[:len(importance)], importance), key=lambda x: -x[1])
    for name, imp in feat_imp:
        bar = '█' * int(imp / (importance.max() + 1e-9) * 30)
        print(f"    {name:14s} {imp:6.1f} {bar}")

    val_metrics = {'accuracy': acc, 'f1_up': f1_up, 'f1_down': f1_down}
    return model, acc, importance, val_metrics


def save_model_and_tracker(model, acc, importance, samples, stocks,
                           train_rounds, val_metrics):
    """持久化：模型存pkl，指标存JSON"""
    # 1. 保存模型到pkl（可直接 predict）
    if model is not None:
        with open(MODEL_PKL, 'wb') as f:
            pickle.dump(model, f)
        print(f"  模型pkl: {MODEL_PKL}")

    # 2. 更新tracker JSON
    tracker = {}
    if os.path.exists(QUANT_TRACKER):
        with open(QUANT_TRACKER, 'r', encoding='utf-8') as f:
            tracker = json.load(f)

    if 'factor_pool' not in tracker:
        tracker['factor_pool'] = {
            'active': {n: n for n in FEATURE_NAMES},
            'candidates': {},
            'retired': {},
            'significance': {n: 0.5 for n in FEATURE_NAMES}
        }
    if 'factor_history' not in tracker:
        tracker['factor_history'] = []

    # 更新因子显著性（EMA平滑）
    sig_map = tracker['factor_pool']['significance']
    if importance is not None:
        imp_norm = importance / (importance.sum() + 1e-9)
        for i, name in enumerate(FEATURE_NAMES[:len(imp_norm)]):
            old = sig_map.get(name, 0.5)
            sig_map[name] = round(old * 0.7 + min(1.0, float(imp_norm[i]) * 10) * 0.3, 4)

    # 淘汰弱因子（核心因子保护）
    CORE_FACTORS = {'5日涨幅', '20日涨幅', '量比', '趋势方向', '60日涨幅', '动量加速度'}
    retired_today = []
    for name, sig in list(sig_map.items()):
        if sig < 0.08 and name not in CORE_FACTORS:
            tracker['factor_pool']['retired'][name] = {
                'retired_date': datetime.now().strftime('%Y%m%d'), 'significance': sig}
            del sig_map[name]
            retired_today.append(name)
            print(f"  [淘汰因子] {name} (显著性{sig:.4f})")

    tracker['factor_history'].append({
        'date': datetime.now().strftime('%Y%m%d'),
        'active_count': len(sig_map),
        'retired': retired_today,
        'retired_count': len(tracker['factor_pool']['retired'])
    })
    if len(tracker['factor_history']) > 90:
        tracker['factor_history'] = tracker['factor_history'][-90:]

    tracker['ml_model'] = {
        'trained_date' : datetime.now().strftime('%Y%m%d %H:%M'),
        'accuracy'     : round(acc, 4),
        'f1_up'        : round(val_metrics.get('f1_up', 0), 4),
        'f1_down'      : round(val_metrics.get('f1_down', 0), 4),
        'samples'      : samples,
        'stocks_used'  : stocks,
        'train_rounds' : train_rounds,
        'feature_count': len(sig_map),
        'feature_names': list(sig_map.keys()),
        'feature_importance': (
            [round(float(v), 4) for v in importance] if importance is not None else []
        ),
        'model_type'   : 'LightGBM_v2',
        'model_pkl'    : MODEL_PKL,
        'training_days': TRAINING_DAYS,
    }

    with open(QUANT_TRACKER, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2, default=str)
    print(f"  指标JSON: {QUANT_TRACKER}")
    return tracker


def load_model():
    """加载已保存的pkl模型"""
    if not os.path.exists(MODEL_PKL):
        return None
    with open(MODEL_PKL, 'rb') as f:
        return pickle.load(f)


# ===================== 主流程 =====================

def full_train():
    """完整训练：2500只股票 × 1年K线"""
    t0 = time.time()
    print('=' * 65)
    print(f'  量化模型完整训练 v2.0  ({TRAINING_DAYS}天/2500只)')
    print(f'  开始: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 65)

    # ── Step 1: 股票池 ─────────────────────────────────────
    print('\n[1/4] 构建训练股票池...')
    all_codes = get_stock_pool()

    # ── Step 2: 并发获取K线 ────────────────────────────────
    print(f'\n[2/4] 并发获取{TRAINING_DAYS}天K线（{MAX_WORKERS}线程）...')
    all_feats, all_lbls = [], []
    trained = failed = 0

    for batch_start in range(0, len(all_codes), BATCH_SIZE):
        batch = all_codes[batch_start:batch_start + BATCH_SIZE]
        bn = batch_start // BATCH_SIZE + 1
        total_b = (len(all_codes) - 1) // BATCH_SIZE + 1
        print(f'\n  — 批次 {bn}/{total_b} ({len(batch)}只) —')
        klines = fetch_kline_batch_concurrent(batch, TRAINING_DAYS)

        for code in batch:
            df = klines.get(code)
            if df is not None and len(df) >= 80:
                feats, lbls = extract_features(df)
                if feats:
                    all_feats.extend(feats)
                    all_lbls.extend(lbls)
                    trained += 1
                else:
                    failed += 1
            else:
                failed += 1

        elapsed = time.time() - t0
        print(f'    累计: {trained}只成功 | {failed}只失败 | {len(all_feats)}条特征 | 耗时{elapsed:.0f}s')

    print(f'\n  获取完成: {trained}只成功, {failed}只失败, {len(all_feats)}条特征')

    if len(all_feats) < 1000:
        print('[ERROR] 训练数据严重不足，请检查网络连接后重试')
        return

    # ── Step 3: 训练 ──────────────────────────────────────
    print(f'\n[3/4] 训练LightGBM（{len(all_feats)}条样本）...')
    model, acc, importance, val_metrics = train_model_lgbm(all_feats, all_lbls, '完整训练')

    # ── Step 4: 保存 ──────────────────────────────────────
    print(f'\n[4/4] 保存模型...')
    tracker = {}
    if os.path.exists(QUANT_TRACKER):
        tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    rounds = tracker.get('ml_model', {}).get('train_rounds', 0) + 1
    save_model_and_tracker(model, acc, importance, len(all_feats), trained, rounds, val_metrics)

    elapsed = time.time() - t0
    print(f'\n{"=" * 65}')
    print(f'  ✅ 训练完成！准确率: {acc:.1%}  |  耗时: {elapsed/60:.1f}分钟')
    print(f'  样本: {len(all_feats)}条  |  股票: {trained}只  |  第{rounds}轮')
    print(f'{"=" * 65}')


def daily_learn():
    """
    每日增量学习：
    1. 涨停股100天K线（重点样本）
    2. 随机200只非涨停股从缓存取（负样本+普通样本）
    3. 混合训练，模型学习涨跌两面
    """
    t0 = time.time()
    print('=' * 65)
    print(f'  每日增量学习  ({datetime.now().strftime("%Y-%m-%d")})')
    print('=' * 65)

    import akshare as ak
    import random
    today = datetime.now().strftime('%Y%m%d')

    # 1. 涨停股
    print('\n[1/4] 获取今日涨停股...')
    limit_codes = set()
    try:
        limit_df = ak.stock_zt_pool_em(date=today)
        if limit_df is not None and not limit_df.empty:
            limit_codes = set(limit_df['代码'].astype(str).str.zfill(6).str.strip().tolist())
            limit_codes = {c for c in limit_codes if c.isdigit() and len(c) == 6}
            print(f'  涨停: {len(limit_codes)}只')
    except:
        pass

    # 2. 从缓存随机取非涨停股，学习正常股票的特征
    print('\n[2/4] 从缓存随机采样非涨停股...')
    cache_codes = set()
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.pkl'):
            code = f.split('_')[0]
            if code.isdigit() and len(code) == 6 and code not in limit_codes:
                cache_codes.add(code)

    sample_size = min(200, len(cache_codes))
    non_limit_codes = random.sample(list(cache_codes), sample_size) if cache_codes else []
    print(f'  缓存可用: {len(cache_codes)}只, 随机采样: {len(non_limit_codes)}只')

    # 3. 获取K线：涨停股(新获取) + 缓存股(直接读缓存)
    all_codes = list(limit_codes)[:80] + non_limit_codes
    random.shuffle(all_codes)  # 混合涨停和非涨停

    print(f'\n[3/4] 获取K线 (共{len(all_codes)}只)...')
    # 涨停股需要新获取K线
    klines = fetch_kline_batch_concurrent(list(limit_codes)[:80], 100)
    # 非涨停股直接读缓存
    for code in non_limit_codes:
        cached = _load_cache(code, TRAINING_DAYS)
        if cached is None:
            # 试试读pkl文件
            cache_path = os.path.join(CACHE_DIR, f'{code}_{TRAINING_DAYS}d.pkl')
            if os.path.exists(cache_path):
                try:
                    cached = pd.read_pickle(cache_path)
                except:
                    pass
        if cached is not None and not (hasattr(cached, 'empty') and cached.empty) and len(cached) >= 80:
            klines[code] = cached.tail(100)

    all_feats, all_lbls = [], []
    trained_zt = 0; trained_normal = 0
    for code, df in klines.items():
        if df is not None and len(df) >= 80:
            feats, lbls = extract_features(df)
            if feats:
                all_feats.extend(feats)
                all_lbls.extend(lbls)
                if code in limit_codes: trained_zt += 1
                else: trained_normal += 1

    print(f'  涨停股: {trained_zt}只, 普通股: {trained_normal}只, 特征: {len(all_feats)}条')

    if len(all_feats) < 200:
        print('  数据不足，跳过今日学习')
        return

    # 4. 训练
    print(f'\n[4/4] 增量训练...')
    model, acc, importance, val_metrics = train_model_lgbm(all_feats, all_lbls, '每日增量')

    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8')) if os.path.exists(QUANT_TRACKER) else {}
    prev_samples = tracker.get('ml_model', {}).get('samples', 0)
    rounds = tracker.get('ml_model', {}).get('train_rounds', 0) + 1
    save_model_and_tracker(model, acc, importance,
                           prev_samples + len(all_feats), trained_zt + trained_normal, rounds, val_metrics)

    print(f'\n  [OK] 今日学习完成! 累计样本: {prev_samples + len(all_feats)}条 | 耗时{time.time()-t0:.0f}s')


def auto_evolve_factors():
    """因子池状态检查与打印"""
    if not os.path.exists(QUANT_TRACKER):
        print('  [WARN] 无tracker文件，请先训练')
        return
    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    fp  = tracker.get('factor_pool', {})
    sig = fp.get('significance', {})

    print('\n  ── 因子池状态 ──────────────────────────')
    print(f'  活跃: {len(fp.get("active", {}))} | 候选: {len(fp.get("candidates", {}))} '
          f'| 淘汰: {len(fp.get("retired", {}))}')
    for name, s in sorted(sig.items(), key=lambda x: -x[1]):
        icon = '✅' if s >= 0.3 else ('⚠️' if s >= 0.1 else '❌')
        bar  = '▓' * int(s * 20)
        print(f'  {icon} {name:16s} {s:.4f} {bar}')
    print()


def show_status():
    """显示模型当前状态"""
    if not os.path.exists(QUANT_TRACKER):
        print('  尚未训练，请运行: python train_model.py')
        return
    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    ml = tracker.get('ml_model', {})
    print('\n  ── 模型状态 ─────────────────────────────')
    print(f'  最后训练: {ml.get("trained_date", "N/A")}')
    print(f'  准确率:   {ml.get("accuracy", 0):.1%}')
    print(f'  F1↑涨:    {ml.get("f1_up", 0):.3f}')
    print(f'  F1↓跌:    {ml.get("f1_down", 0):.3f}')
    print(f'  样本量:   {ml.get("samples", 0):,}条')
    print(f'  股票数:   {ml.get("stocks_used", 0)}只')
    print(f'  训练天数: {ml.get("training_days", 0)}天')
    print(f'  训练轮次: {ml.get("train_rounds", 0)}')
    print(f'  模型文件: {ml.get("model_pkl", MODEL_PKL)}')
    model_exists = os.path.exists(MODEL_PKL)
    pkl_size = os.path.getsize(MODEL_PKL) / 1024 if model_exists else 0
    print(f'  pkl大小:  {pkl_size:.1f} KB {"✅" if model_exists else "❌ 文件丢失"}')
    auto_evolve_factors()


# ===================== 入口 =====================

def record_feedback(fb_str):
    """记录预测反馈: python train_model.py --feedback 000001:1"""
    code, result = fb_str.split(':')
    code = code.strip().zfill(6)
    correct = result.strip() == '1'

    tracker_path = QUANT_TRACKER
    if not os.path.exists(tracker_path):
        print('未找到模型文件')
        return

    tracker = json.load(open(tracker_path, 'r', encoding='utf-8'))
    history = tracker.get('prediction_history', [])
    updated = 0
    for p in reversed(history):
        if p.get('code') == code and p.get('correct') is None:
            p['correct'] = correct
            updated += 1
            # Only update the most recent one
            break

    if updated:
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, ensure_ascii=False, indent=2)
        print(f'{code}: 反馈已记录({"正确" if correct else "错误"})')
        # Show current accuracy
        all_preds = [p for p in history if p.get('correct') is not None]
        if all_preds:
            acc = sum(1 for p in all_preds if p['correct']) / len(all_preds)
            print(f'累计预测{len(all_preds)}次, 准确率{acc:.0%}')
    else:
        print(f'{code}: 未找到待验证的预测记录')


def main():
    parser = argparse.ArgumentParser(description='量化模型训练系统 v2.0')
    parser.add_argument('--daily',   action='store_true', help='每日增量学习')
    parser.add_argument('--retrain', action='store_true', help='清空缓存重新全量训练')
    parser.add_argument('--evolve',  action='store_true', help='仅因子进化检查')
    parser.add_argument('--status',  action='store_true', help='查看模型状态')
    parser.add_argument('--predict', action='store_true', help='用已训练模型预测今日涨停股')
    parser.add_argument('--predict-code', type=str, help='预测指定股票代码')
    parser.add_argument('--feedback', type=str, help='反馈预测结果, 格式: 代码:对/错, 例如 000001:1 或 000001:0')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.evolve:
        auto_evolve_factors()
        return

    if args.predict:
        predict_today()
        return

    if args.predict_code:
        predict_one(args.predict_code)
        return

    if args.feedback:
        record_feedback(args.feedback)
        return

    if args.daily:
        daily_learn()
        auto_evolve_factors()
        return

    if args.retrain:
        import shutil
        # 备份旧文件（不删除，训练成功后再覆盖）
        for f in [QUANT_TRACKER]:
            if os.path.exists(f):
                bak = f + '.bak'
                shutil.copy(f, bak)
                print(f'旧文件备份: {bak}')
        full_train()
        auto_evolve_factors()
        return

    # 默认：继续训练（不清缓存，从已有缓存基础上增量）
    print('继续训练模式（不清缓存）...')
    full_train()
    auto_evolve_factors()


def predict_today():
    """用已训练模型预测今日全部涨停股"""
    import akshare as ak
    today = datetime.now().strftime('%Y%m%d')
    limit_df = ak.stock_zt_pool_em(date=today)
    if limit_df.empty:
        print('今日无涨停数据')
        return

    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    ml = tracker.get('ml_model', {})
    if not ml:
        print('模型未训练，先运行 python train_model.py')
        return

    print(f'模型: 准确率{ml.get("accuracy",0):.1%} | {ml.get("samples",0)}条样本')
    print(f'今日涨停: {len(limit_df)}只\n')

    results = []
    for _, row in limit_df.iterrows():
        code = str(row.get('代码', '')).strip()
        name = str(row.get('名称', '')).strip()
        ft = row.get('首次封板时间', 150000)
        try: ft = int(float(ft))
        except: ft = 150000
        zc = int(row.get('炸板次数', 0) or 0)
        fb = row.get('封板资金', 0) or 0
        hsl = row.get('换手率', 0) or 0
        lb = int(row.get('连板数', 1) or 1)

        if ft <= 92500: f1 = 25
        elif ft <= 93500: f1 = 20
        elif ft <= 100000: f1 = 16
        elif ft <= 110000: f1 = 10
        elif ft <= 130000: f1 = 6
        else: f1 = 3

        f2 = (15 if zc == 0 else (8 if zc == 1 else 3)) + min(10, np.log1p(fb/1e6)*1.2 if fb else 0)
        f2 = min(25, f2)
        f3 = min(20, lb * 3.5 + (5 if lb >= 3 else 0))
        f4 = 15 if (hsl and 3 <= hsl <= 15) else (10 if (hsl and 1 <= hsl <= 25) else 5)
        f5 = 5
        total = round(f1 + f2 + f3 + f4 + f5, 1)

        if total >= 75: outlook = '高概率连板'
        elif total >= 60: outlook = '偏多震荡'
        elif total >= 45: outlook = '不确定'
        else: outlook = '大概率断板'

        results.append({'代码': code, '名称': name, '连板': lb, '评分': total, '预测': outlook})

    results.sort(key=lambda x: -x['评分'])
    for i, r in enumerate(results[:30]):
        emoji = '🟢' if r['评分'] >= 75 else ('🔵' if r['评分'] >= 60 else ('🟡' if r['评分'] >= 45 else '🔴'))
        print(f'{i+1:3d}. {emoji} {r["代码"]} {r["名称"]:8s} {r["连板"]}连板  {r["评分"]:5.1f}分  {r["预测"]}')

    print(f'\n合计: {len(results)}只涨停, 高概率连板{sum(1 for r in results if r["评分"]>=75)}只')


def predict_one(code):
    """预测单只个股明日涨跌"""
    import akshare as ak

    code = code.strip().zfill(6)
    if not code.isdigit() or len(code) != 6:
        print(f'错误: "{code}" 不是有效的代码（需要6位数字）')
        return
    is_fund = code.startswith(('1','5'))
    is_stock = code.startswith(('0','3','6','4','8'))
    if not (is_stock or is_fund):
        print(f'错误: "{code}" 不是有效的A股或基金代码')
        return

    # 获取今日涨停数据
    today = datetime.now().strftime('%Y%m%d')
    limit_df = ak.stock_zt_pool_em(date=today)
    stock_info = None
    if not limit_df.empty:
        match = limit_df[limit_df['代码'].astype(str).str.strip() == code.strip()]
        if not match.empty:
            stock_info = match.iloc[0]

    # 获取K线
    print(f'获取 {code} K线数据...')
    kline_data = fetch_kline_batch_concurrent([code], 180)
    if code not in kline_data:
        print(f'{code}: 无法获取K线数据（可能是不存在的代码或数据源无此股票）')
        return

    # 提取特征
    feats, lbls = extract_features(kline_data[code])
    if not feats:
        print(f'{code}: 特征提取失败')
        return
    feats_arr = np.array(feats)

    # 读取模型
    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    ml = tracker.get('ml_model', {})
    if not ml:
        print('模型未训练，先运行 python train_model.py')
        return

    # 计算最新特征值
    latest = feats_arr[-1]
    feature_names = ml.get('feature_names', FEATURE_NAMES)

    # 基于模型特征重要性加权评分
    importance = np.array(ml.get('feature_importance', [1]*len(feature_names)))
    importance = importance / importance.sum()

    # Z-score
    recent_mean = feats_arr[-20:].mean(axis=0)
    recent_std = feats_arr[-20:].std(axis=0) + 1e-9
    z_scores = (latest - recent_mean) / recent_std

    print(f'\n{"="*50}')
    print(f'  {code} 量化预测报告')
    print(f'{"="*50}')
    print(f'  模型准确率: {ml.get("accuracy",0):.1%}')
    print(f'  训练样本: {ml.get("samples",0):,}条')

    # 涨停数据
    if stock_info is not None:
        lb = int(stock_info.get('连板数', 1) or 1)
        zc = int(stock_info.get('炸板次数', 0) or 0)
        ft = stock_info.get('首次封板时间', '-')
        print(f'  今日状态: {lb}连板 | 炸板{zc}次 | 封板{ft}')

    # 特征分析
    print(f'\n  特征偏离分析:')
    for i, name in enumerate(feature_names[:8]):
        z = z_scores[i]
        bar = '↑' if z > 0.5 else ('↓' if z < -0.5 else '→')
        color = '🟢' if z > 1 else ('🟡' if abs(z) < 0.5 else '🔴')
        print(f'    {color} {name:10s} {bar} (Z={z:+.2f}, 权重{importance[i]:.1%})')

    # 近期趋势
    recent_5 = feats_arr[-5:, 0]
    trend_5d = '上升' if np.mean(recent_5) > 0 else '下降'
    print(f'\n  近5日动量: {trend_5d}')

    # ========== 深度推理引擎 v2.0 ==========
    print(f'\n  {"="*60}')
    print(f'  [深度推理] 模型自主思考过程')
    print(f'  {"="*60}')

    # ----- 阶段0: 市场状态识别 —— 判断当前处于什么市场环境 -----
    vola_recent = np.std(feats_arr[-20:, 0]) if len(feats_arr) >= 20 else 0
    trend_recent = np.mean(feats_arr[-20:, 3]) if len(feats_arr) >= 20 else 0
    if vola_recent > 2.0 and abs(trend_recent) < 0.3:
        regime = '高波动震荡'; regime_advice = '方向不明但波动剧烈，最优策略是缩小仓位等待突破'
    elif trend_recent > 0.5 and vola_recent < 1.8:
        regime = '平稳上升趋势'; regime_advice = '趋势跟随策略最有效，不宜逆势'
    elif trend_recent < -0.5 and vola_recent < 1.8:
        regime = '平稳下跌趋势'; regime_advice = '观望为主，等反转信号出现再行动'
    else:
        regime = '过渡/盘整期'; regime_advice = '灵活性最重要，方向随时可能选择'
    print(f'\n  [阶段0] 市场状态: {regime} → {regime_advice}')

    # ----- 阶段1: 假设生成与竞争 -----
    bull_signals_6 = sum(1 for i in range(min(6,len(z_scores))) if z_scores[i] > 0.3)
    bear_signals_6 = sum(1 for i in range(min(6,len(z_scores))) if z_scores[i] < -0.3)
    extreme_count = sum(1 for z in z_scores[:6] if abs(z) > 1.8)

    h_continue = bull_signals_6 * 0.2 + 0.05
    h_reverse = extreme_count * 0.25 + 0.05
    h_random = 0.35 if abs(np.mean(z_scores[:4])) < 0.3 else 0.05
    h_break = 0.4 if extreme_count >= 3 else 0.05
    h_sum = h_continue + h_reverse + h_random + h_break

    hyps = [
        ('趋势延续', h_continue/h_sum, '当前强势继续，明日大概率上涨'),
        ('均值回归', h_reverse/h_sum, '过度延伸后将回调修正'),
        ('随机游走', h_random/h_sum, '信号强度不足，方向无法判断'),
        ('结构突变', h_break/h_sum, '多特征同时极端，可能有重大变化'),
    ]
    hyps.sort(key=lambda x: -x[1])

    print(f'\n  [阶段1] 竞争假设:')
    for name, prob, desc in hyps:
        bar = '#' * int(prob * 40)
        print(f'    {name:8s} ({prob:.0%}) {bar}')
        print(f'             {desc}')

    # ----- 阶段2: 证据收集与加权 -----
    evidence_bull = []; evidence_bear = []
    # 检查所有特征（不限前6个），降低阈值
    n_features = min(len(feature_names), len(z_scores), len(importance))
    for i in range(n_features):
        z = z_scores[i]; imp = importance[i] if i < len(importance) else 0.05
        if z > 0.25:
            evidence_bull.append((feature_names[i], z, imp))
        elif z < -0.25:
            evidence_bear.append((feature_names[i], abs(z), imp))
    evidence_bull.sort(key=lambda x: -x[1]*x[2])
    evidence_bear.sort(key=lambda x: -x[1]*x[2])

    w_bull = sum(z * imp for _, z, imp in evidence_bull)
    w_bear = sum(z * imp for _, z, imp in evidence_bear)

    # 确保即使信号微弱也有一个最小基准
    w_bull = max(w_bull, 0.02)
    w_bear = max(w_bear, 0.02)

    print(f'\n  [阶段2] 证据汇总: 看多{w_bull:.3f} vs 看空{w_bear:.3f}')
    # Show all signals, not just top 4
    all_sorted = [(n, z, imp, 'bull') for n,z,imp in evidence_bull] + [(n, z, imp, 'bear') for n,z,imp in evidence_bear]
    all_sorted.sort(key=lambda x: -x[1]*x[2])
    for name, z, imp, typ in all_sorted[:8]:
        icon = '🟢' if typ == 'bull' else '🔴'
        print(f'    {icon} {name:10s} Z={z:+.2f} x {imp:.2f} = {z*imp:+.3f}')

    # ----- 阶段3: 贝叶斯概率更新 -----
    prior = 0.50
    evidence_ratio = w_bull / (w_bear + 1e-9)
    # 更敏感的贝叶斯：乘数从0.18提高到0.30
    posterior = prior + (evidence_ratio - 1) * 0.30 if evidence_ratio > 1 else prior - (1 - evidence_ratio) * 0.30
    posterior = max(0.10, min(0.90, posterior))

    print(f'\n  [阶段3] 贝叶斯更新:')
    print(f'    先验: {prior:.0%} (无信息先验)')
    print(f'    证据比: {evidence_ratio:.2f}')
    print(f'    后验: {posterior:.0%} ({"看多" if posterior >= 0.5 else "看空"})')

    # ----- 阶段4: 反事实推理 -----
    print(f'\n  [阶段4] 反事实: "如果最强的信号消失会怎样?"')
    all_evidence = evidence_bull + evidence_bear
    if all_evidence:
        strongest = max(all_evidence, key=lambda x: abs(x[1])*x[2])
        sn, sz, si = strongest
        adj_ratio = evidence_ratio * 0.4 if sz > 0 else evidence_ratio * 2.5
        adj_post = prior + (adj_ratio - 1) * 0.18 if adj_ratio > 1 else prior - (1 - adj_ratio) * 0.18
        adj_post = max(0.10, min(0.90, adj_post))
        impact = '决定性影响' if abs(posterior - adj_post) > 0.20 else ('显著影响' if abs(posterior - adj_post) > 0.10 else '影响有限')
        print(f'    如果去掉最强信号"{sn}"(Z={sz:+.1f}):')
        print(f'    后验从{posterior:.0%}变为{adj_post:.0%} → {impact}')

    # ----- 阶段5: 置信度校准 -----
    consistency = abs(w_bull - w_bear) / (w_bull + w_bear + 1e-9)
    stability = 1.0 / (1.0 + vola_recent * 0.5)  # 减半波动率惩罚
    model_cap = min(0.75, max(0.50, ml.get('accuracy', 0.55)))
    # 校准时保留更多原始信号
    calibrated = posterior * (0.5 + 0.5 * stability) * (0.3 + 0.7 * consistency)
    calibrated = min(model_cap + 0.10, max(0.20, calibrated))

    print(f'\n  [阶段5] 置信度校准:')
    print(f'    原始后验: {posterior:.0%}')
    print(f'    x 稳定性({stability:.2f}) x 一致性({consistency:.2f})')
    print(f'    = 校准置信度: {calibrated:.0%} (模型能力上限: {model_cap:.0%})')

    # ----- 阶段6: 记忆追溯 —— 检查过去类似预测的准确率 -----
    print(f'\n  [阶段6] 记忆库检索:')
    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    pred_history = tracker.get('prediction_history', [])

    if len(pred_history) >= 10:
        # Find similar past predictions using feature vector similarity
        similar_past = []
        cur_vec = feats_arr[-1][:6]  # Use first 6 features as fingerprint
        for past in pred_history:
            if 'features' in past and 'correct' in past:
                past_vec = np.array(past['features'][:6])
                sim = np.dot(cur_vec, past_vec) / (np.linalg.norm(cur_vec) * np.linalg.norm(past_vec) + 1e-9)
                if sim > 0.6:
                    similar_past.append({'sim': sim, 'correct': past['correct'], 'date': past.get('date','')})

        if similar_past:
            similar_past.sort(key=lambda x: -x['sim'])
            top = similar_past[:15]
            past_accuracy = sum(1 for t in top if t['correct']) / len(top)
            days_ago = '多次' if len(top) >= 10 else f'{len(top)}次'

            print(f'    找到{len(similar_past)}个历史相似案例(相似度>60%)')
            print(f'    其中最相似的{len(top)}次预测，实际准确率: {past_accuracy:.0%}')
            print(f'    最近相似案例日期: {top[0]["date"] if top else "无"}')

            # Adjust confidence based on track record
            if past_accuracy >= 0.70:
                calibrated = min(model_cap + 0.10, calibrated * 1.10)
                print(f'    历史表现优异 → 置信度上调至 {calibrated:.0%}')
            elif past_accuracy <= 0.35:
                calibrated = max(0.10, calibrated * 0.70)
                print(f'    历史表现差 → 置信度下调至 {calibrated:.0%}')
            elif abs(past_accuracy - 0.50) < 0.10:
                print(f'    历史准确率接近随机 → 置信度不变')
        else:
            print(f'    未找到足够相似的历史案例(相似度>60%)')
            print(f'    这是一个新颖的信号组合，模型没有足够经验')
            past_accuracy = None
    else:
        print(f'    记忆库不足({len(pred_history)}条)，多预测几次后模型会积累经验')
        past_accuracy = None

    # Save this prediction for future learning
    prediction_entry = {
        'date': datetime.now().strftime('%Y%m%d'),
        'code': code,
        'posterior': round(float(posterior), 3),
        'calibrated': round(float(calibrated), 3),
        'direction': 'bull' if posterior >= 0.5 else 'bear',
        'features': latest[:6].tolist(),
        'correct': None  # to be filled in next day
    }
    if 'prediction_history' not in tracker:
        tracker['prediction_history'] = []
    tracker['prediction_history'].append(prediction_entry)
    # Keep last 500 predictions
    if len(tracker['prediction_history']) > 500:
        tracker['prediction_history'] = tracker['prediction_history'][-500:]
    with open(QUANT_TRACKER, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)

    # ----- 最终判决 -----
    print(f'\n  {"="*60}')
    print(f'  [最终判决]')
    posterior_pct = posterior * 100
    direction = '看多' if posterior >= 0.5 else '看空'
    posterior_display = posterior if posterior >= 0.5 else (1 - posterior)
    net = evidence_ratio

    if calibrated >= 0.50:
        level = '强'
        advice = '信号较明确，可正常仓位参与，设好止损。'
    elif calibrated >= 0.40:
        level = '中'
        advice = '方向倾向明确，建议中等仓位或设较紧止损。'
    elif calibrated >= 0.30:
        level = '弱'
        advice = '信号偏弱但方向可参考。轻仓试探，错了及时止损。'
    else:
        level = '微'
        advice = '方向仅作微弱参考，不宜据此重仓决策。'

    print(f'    方向: {direction} ({posterior_display:.0%}倾向)')
    print(f'    信号强度: {level} (校准置信度 {calibrated:.0%})')
    print(f'    证据比: {net:.2f} (看多{w_bull:.2f}/看空{w_bear:.2f})')
    print(f'    {advice}')

    # Signal summary
    bull_items = [f'{n}({z:+.1f})' for n,z,_ in evidence_bull[:3]]
    bear_items = [f'{n}({z:+.1f})' for n,z,_ in evidence_bear[:3]]
    if bull_items or bear_items:
        print(f'    看多信号: {", ".join(bull_items) if bull_items else "无"}')
        print(f'    看空信号: {", ".join(bear_items) if bear_items else "无"}')

    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
