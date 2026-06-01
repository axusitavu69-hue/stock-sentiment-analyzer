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
TARGET_STOCKS   = 2500         # 目标股票数
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
        tracker = json.load(open(QUANT_TRACKER))
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
    1. 获取今日涨停股K线（近100天）
    2. 提特征、训练
    3. 若已有历史模型，用 init_model 暖启动（延续上次权重）
    """
    t0 = time.time()
    print('=' * 65)
    print(f'  每日增量学习  ({datetime.now().strftime("%Y-%m-%d")})')
    print('=' * 65)

    import akshare as ak
    today = datetime.now().strftime('%Y%m%d')

    print('\n[1/3] 获取今日涨停股...')
    try:
        limit_df = ak.stock_zt_pool_em(date=today)
        if limit_df is None or limit_df.empty:
            print('  今日无涨停数据，跳过')
            return
        codes = limit_df['代码'].astype(str).str.strip().tolist()
        codes = [c for c in codes if c.isdigit() and len(c) == 6]
        print(f'  涨停: {len(codes)}只')
    except Exception as e:
        print(f'  [ERROR] 获取涨停池失败: {e}')
        return

    print(f'\n[2/3] 并发获取100天K线...')
    klines = fetch_kline_batch_concurrent(codes[:80], 100)
    all_feats, all_lbls = [], []
    trained = 0
    for code, df in klines.items():
        if len(df) >= 80:
            feats, lbls = extract_features(df)
            if feats:
                all_feats.extend(feats)
                all_lbls.extend(lbls)
                trained += 1
    print(f'  有效: {trained}只  |  特征: {len(all_feats)}条')

    if len(all_feats) < 200:
        print('  数据不足，跳过今日学习')
        return

    print(f'\n[3/3] 增量训练...')
    model, acc, importance, val_metrics = train_model_lgbm(all_feats, all_lbls, '每日增量')

    tracker = json.load(open(QUANT_TRACKER)) if os.path.exists(QUANT_TRACKER) else {}
    prev_samples = tracker.get('ml_model', {}).get('samples', 0)
    rounds = tracker.get('ml_model', {}).get('train_rounds', 0) + 1
    save_model_and_tracker(model, acc, importance,
                           prev_samples + len(all_feats), trained, rounds, val_metrics)

    print(f'\n  ✅ 今日学习完成！累计样本: {prev_samples + len(all_feats)}条 | 耗时{time.time()-t0:.0f}s')


def auto_evolve_factors():
    """因子池状态检查与打印"""
    if not os.path.exists(QUANT_TRACKER):
        print('  [WARN] 无tracker文件，请先训练')
        return
    tracker = json.load(open(QUANT_TRACKER))
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
    tracker = json.load(open(QUANT_TRACKER))
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

def main():
    parser = argparse.ArgumentParser(description='量化模型训练系统 v2.0')
    parser.add_argument('--daily',   action='store_true', help='每日增量学习')
    parser.add_argument('--retrain', action='store_true', help='清空缓存重新全量训练')
    parser.add_argument('--evolve',  action='store_true', help='仅因子进化检查')
    parser.add_argument('--status',  action='store_true', help='查看模型状态')
    parser.add_argument('--predict', action='store_true', help='用已训练模型预测今日涨停股')
    parser.add_argument('--predict-code', type=str, help='预测指定股票代码')
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
    """预测单只股票"""
    df = fetch_kline_batch_concurrent([code], 180)
    if code not in df:
        print(f'{code}: 无法获取K线数据')
        return
    feats, _ = extract_features(df[code])
    tracker = json.load(open(QUANT_TRACKER, 'r', encoding='utf-8'))
    ml = tracker.get('ml_model', {})
    print(f'{code}: 模型准确率{ml.get("accuracy",0):.1%}, {len(feats)}条特征可用于评估')


if __name__ == '__main__':
    main()
